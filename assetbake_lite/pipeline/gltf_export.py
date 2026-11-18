# SPDX-License-Identifier: GPL-3.0-or-later
"""GLB export of a finished asset. Runs in the user's session (Export button) and in the tests.

What the round-trip tests of the research fixed (Finding 12):
  - the mesh that was baked is the mesh that is exported: triangles as they are, with tangents. A quad-dominant
    asset was baked on a fixed split of its quads (tris.py); the file gets exactly those triangles, because a quad
    left to the exporter or the engine may be split along the other diagonal and then the tangents no longer fit
    the normal map (Finding 12);
  - the object scale is applied, so the file carries no node scale;
  - occlusion, roughness and metallic travel as ONE texture (R, G, B); the material built by material.py is
    already wired the way Blender's glTF exporter understands;
  - glTF wants OpenGL normal maps, which is what Blender bakes. The DirectX option flips the green channel in the
    exported file for engines that expect it (Unreal Engine with default import settings).
The scene is left as it was: the export works on a temporary copy.
"""
import json
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from . import ASSET_PROPERTY, material, tris, util

TEXTURE_KINDS = ("basecolor", "orm", "normal", "emission")


def available():
    return hasattr(bpy.types, "EXPORT_SCENE_OT_gltf")


def _flipped_green(image, path, name=None):
    """A copy of a normal map with the green channel inverted, saved as PNG at `path`; the image datablock is
    called `name` (the glTF exporter names the embedded texture after it)."""
    px = util.image_pixels(image).copy()
    px[..., 1] = 1.0 - px[..., 1]
    out = bpy.data.images.new(name or os.path.basename(path), image.size[0], image.size[1], alpha=False)
    out.colorspace_settings.name = "Non-Color"
    full = np.ones((px.shape[0], px.shape[1], 4), np.float32)
    full[..., :3] = px[..., :3]
    out.pixels.foreach_set(full.ravel())
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    return out


def _write_png(image, path):
    """The PNG bytes of a packed image, exactly as they were baked."""
    if image.packed_file is not None:
        with open(path, "wb") as fh:
            fh.write(image.packed_file.data)
    else:
        image.save(filepath=path)


def fixed_split(ob):
    """True when the asset was baked on the fixed split of its quads and still consists of triangles and quads."""
    try:
        mark = json.loads(ob.get(ASSET_PROPERTY) or "{}")
    except (TypeError, ValueError):
        return False
    return bool(isinstance(mark, dict) and mark.get("fixed_split")) and tris.is_fixed(ob.data)


def split_state(ob):
    """"none": not a quad-dominant asset; "ok": its faces are the ones the bake was made on; "edited": the faces
    were changed after the asset was made (more than four corners, other faces), so the baked split is lost."""
    try:
        mark = json.loads(ob.get(ASSET_PROPERTY) or "{}")
    except (TypeError, ValueError):
        return "none"
    if not (isinstance(mark, dict) and mark.get("fixed_split")):
        return "none"
    me = ob.data
    if not tris.is_fixed(me):
        return "edited"
    if mark.get("topology") is not None:
        return "ok" if mark["topology"] == tris.topology_hash(me) else "edited"
    return "ok" if mark.get("faces") == len(me.polygons) else "edited"      # an asset of 0.2.0: no hash yet


def export_asset(context, ob, filepath, directx=False, save_textures=False):
    """Write `ob` as <filepath> (GLB). Returns a dict with the file, the triangle count and the texture files;
    "warning" says when the asset was edited after it was made (the triangles are then split by a fixed rule
    and are no longer the ones the textures were baked on)."""
    if not available():
        raise RuntimeError("Blender's glTF 2.0 add-on is not enabled")
    folder, stem = os.path.dirname(filepath), os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(folder, exist_ok=True)
    layer = context.view_layer
    selected = [o for o in context.selected_objects]
    active = layer.objects.active
    name = ob.name
    temp_images, temp_mats, files = [], [], []

    state = split_state(ob)
    warning = ""
    if state == "edited":
        warning = ("%s was edited after it was made: its triangles are no longer the ones the textures were baked on, "
                   "so the normal map can show small errors where faces changed. The file is split by a fixed "
                   "rule (every quad along its shorter diagonal). For an exact result make the asset again." % ob.name)
    copy = ob.copy()
    # a quad-dominant asset: the bake's own triangles (when still intact); anything else: the mesh as it is
    copy.data = tris.triangulated_mesh(ob.data, ob.data.name + ".export") if (
        state != "none" and tris.is_fixed(ob.data)) else ob.data.copy()
    context.scene.collection.objects.link(copy)
    ob.name = name + ".export_source"
    copy.name = name                                  # the node in the file gets the asset's own name
    try:
        # scale applied (rotation and location stay on the node)
        copy.parent = None
        copy.matrix_world = ob.matrix_world
        loc, rot, scale = copy.matrix_world.decompose()
        copy.data.transform(Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0)))
        copy.scale = (1.0, 1.0, 1.0)
        if scale.x * scale.y * scale.z < 0:
            copy.data.flip_normals()                  # a mirrored object: keep the faces pointing outwards
        if any(p.loop_total != 3 for p in copy.data.polygons):
            bm = bmesh.new()
            bm.from_mesh(copy.data)
            bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="SHORT_EDGE", ngon_method="EAR_CLIP")
            bm.to_mesh(copy.data)
            bm.free()

        mat = copy.data.materials[0] if copy.data.materials else None
        normal_node = material.image_node(mat, "normal")
        # the flipped map goes to the target folder only when texture files were asked for; otherwise to
        # Blender's temporary folder, so no file of the user's is overwritten or deleted
        normal_path = os.path.join(folder if save_textures else (bpy.app.tempdir or folder),
                                   stem + ("_normal.png" if save_textures else "_normal_directx_tmp.png"))
        if directx and normal_node is not None and normal_node.image is not None:
            mat = mat.copy()
            temp_mats.append(mat)
            copy.data.materials[0] = mat
            flipped = _flipped_green(normal_node.image, normal_path, stem + "_normal")
            temp_images.append(flipped)
            material.image_node(mat, "normal").image = flipped
        if save_textures and mat is not None:
            for kind in TEXTURE_KINDS:
                node = material.image_node(mat, kind)
                if node is None or node.image is None:
                    continue
                path = os.path.join(folder, "%s_%s.png" % (stem, kind))
                if not (kind == "normal" and directx):         # the flipped normal map is already there
                    _write_png(node.image, path)
                files.append(path)

        for o in selected:
            o.select_set(False)
        copy.hide_set(False)
        copy.select_set(True)
        layer.objects.active = copy
        bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB", use_selection=True, export_tangents=True,
                                  export_yup=True, export_apply=False)
        triangles = len(copy.data.polygons)
    finally:
        mesh = copy.data
        bpy.data.objects.remove(copy)
        bpy.data.meshes.remove(mesh)
        ob.name = name
        for m in temp_mats:
            bpy.data.materials.remove(m)
        for im in temp_images:
            bpy.data.images.remove(im)
        if directx and not save_textures and os.path.exists(normal_path):
            os.remove(normal_path)
        for o in selected:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if active is not None and active.name in layer.objects:
            layer.objects.active = active
    return dict(file=filepath, triangles=triangles, textures=files, normal_map="DirectX" if directx else "OpenGL",
                warning=warning)
