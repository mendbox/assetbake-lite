# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 5: bake the source's material onto the low mesh (Cycles, CPU, selected to active).

What the research settled (Findings 7, 9, 12):
  - colour, roughness, metallic, emission and alpha are baked through a temporary Emission shader (type EMIT):
    the DIFFUSE colour pass comes out black wherever the source is metallic;
  - emission is baked as colour x strength / largest strength, and the largest strength goes on the result
    material as one number (a file where one material glows must not make the whole asset glow);
  - the cage follows the measured deviation of the low mesh, with no fixed floor: a cage wider than the gap between
    two parts starts the ray behind the neighbouring surface and bakes its back;
  - texels whose ray missed are baked a second time with a wide cage, and only those texels are taken from it;
  - only the bake source and the low mesh may be visible to the renderer;
  - the low mesh is triangles before the bake, so bake, viewport and export use the same triangulation.
"""
import os
import re
import time

import bmesh
import bpy
import numpy as np

from . import util

MAPS = ("basecolor", "roughness", "metallic", "emission", "alpha", "ao")
SOCKET = dict(basecolor="Base Color", roughness="Roughness", metallic="Metallic", emission="Emission Color",
              alpha="Alpha")
UNLIT = dict(roughness=1.0, metallic=0.0, emission=(0.0, 0.0, 0.0), alpha=1.0)   # an Emission-only material
SAMPLES = dict(mask=1, normal=16, ao=64, emit=4)
SCALARS = ("roughness", "metallic", "alpha")      # one value per texel: baked together, one channel each
AO_FULL_SIZE_MAX = 1024        # larger textures get their AO baked at half the size (AO is soft; it is the slowest map)
RETRY_PATCH_SHARE_MAX = 0.5    # second pass: bake onto a cut-out of the low mesh unless it would be most of the mesh
RETRY_MIN_TEXELS = 8
RETRY_MIN_SHARE = 1e-5


# ---------------------------------------------------------------------------------------------- source materials

def ensure_nodes(mat):
    """A node tree with the default Principled BSDF (Material.use_nodes is deprecated from Blender 5.0)."""
    if mat.node_tree is None or not mat.node_tree.nodes:
        mat.use_nodes = True
    return mat


def _nodes_in_use(mat):
    if mat is None or mat.node_tree is None:
        return False
    return bpy.app.version >= (5, 0, 0) or mat.use_nodes


def find_shader(mat):
    """(output node, shader node, kind) of a material. kind is "PRINCIPLED" or "EMISSION" (an unlit material, as
    glTF files without lighting import), found upstream of the surface through mix and add shaders; a Principled
    BSDF wins over an Emission shader. (output, None, None) when the material has neither."""
    if not _nodes_in_use(mat):
        return None, None, None
    out = mat.node_tree.get_output_node("CYCLES")
    if out is None or not out.inputs["Surface"].is_linked:
        return out, None, None
    queue = [out.inputs["Surface"].links[0].from_node]
    seen = set()
    emission = None
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        if node.type == "BSDF_PRINCIPLED":
            return out, node, "PRINCIPLED"
        if node.type == "EMISSION" and emission is None:
            emission = node
        for sock in node.inputs:
            if sock.type == "SHADER" and sock.is_linked:
                queue.append(sock.links[0].from_node)
    return (out, emission, "EMISSION") if emission is not None else (out, None, None)


def find_principled(mat):
    out, node, kind = find_shader(mat)
    return out, node if kind == "PRINCIPLED" else None


def source_of(mat, what):
    """Where one map of a material comes from: a node socket (linked or with a default value) or a constant."""
    _, node, kind = find_shader(mat)
    if kind == "PRINCIPLED":
        return node.inputs[SOCKET[what]]
    # unlit: the emitted colour is the look of the surface, so it becomes the base colour
    return node.inputs["Color"] if what == "basecolor" else UNLIT[what]


def emission_strength(mat):
    _, node, kind = find_shader(mat)
    if kind != "PRINCIPLED":
        return 0.0
    colour = node.inputs["Emission Color"]
    lit = colour.is_linked or max(colour.default_value[:3]) > 0
    return float(node.inputs["Emission Strength"].default_value) if lit else 0.0


def _alpha_is_real(mat):
    """Does the material's alpha change anything? FBX and OBJ imports link the texture's alpha on opaque
    materials too; an image whose alpha channel is 1 everywhere must not turn the asset into a cut-out."""
    sock = source_of(mat, "alpha")
    if not hasattr(sock, "is_linked"):
        return sock < 0.999
    if not sock.is_linked:
        return sock.default_value < 0.999
    link = sock.links[0]
    image = getattr(link.from_node, "image", None) if link.from_node.type == "TEX_IMAGE" else None
    if image is None or link.from_socket.name != "Alpha":
        return True
    if image.channels < 4 or image.size[0] == 0:
        return False
    return float(util.image_pixels(image)[..., 3].min()) < 0.99


def _fallback_material(me, state):
    """For a slot the bake cannot read: the mesh's vertex colours when it has them, plain grey otherwise."""
    colours = me.color_attributes.active_color or (me.color_attributes[0] if me.color_attributes else None)
    key = "vertex" if colours is not None else "grey"
    if key not in state:
        mat = ensure_nodes(bpy.data.materials.new("source_" + key))
        if colours is not None:
            tree = mat.node_tree
            node = tree.nodes.new("ShaderNodeVertexColor")
            node.layer_name = colours.name
            shader = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
            tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
        state[key] = mat
    return state[key], key


def prepare_source_materials(src):
    """Every material slot of the source gets a material the bake can read. A slot that is empty or has neither a
    Principled BSDF nor an Emission shader shows the mesh's vertex colours, or plain grey when there are none.
    Returns a dict: mats, has (which maps the source really has), emission_max, replaced (names of materials
    baked grey), vertex_colours (bool), unlit (names), missing (texture files not found), constants."""
    me = src.data
    state, replaced, vertex = {}, [], False
    if not me.materials:
        me.materials.append(None)
    for i, mat in enumerate(me.materials):
        if find_shader(mat)[1] is None:
            me.materials[i], key = _fallback_material(me, state)
            if key == "vertex":
                vertex = True
            else:
                replaced.append(mat.name if mat else "(empty slot)")
    mats = list(dict.fromkeys(me.materials))
    strengths = [emission_strength(m) for m in mats]
    has = dict(basecolor=True, roughness=True, metallic=True, ao=True,
               emission=any(x > 0 for x in strengths), alpha=any(_alpha_is_real(m) for m in mats))
    missing = sorted({node.image.name for m in mats for node in m.node_tree.nodes
                      if node.type == "TEX_IMAGE" and node.image is not None and not node.image.has_data
                      and node.image.size[0] == 0})

    def mean_default(what, fallback):
        values = []
        for m in mats:
            sock = source_of(m, what)
            if not hasattr(sock, "is_linked"):
                values.append(sock)
            elif not sock.is_linked:
                values.append(sock.default_value if isinstance(sock.default_value, float) else tuple(sock.default_value)[:3])
        return np.mean(np.array(values, np.float64), 0).tolist() if values else fallback

    constants = dict(roughness=mean_default("roughness", 0.5), metallic=mean_default("metallic", 0.0),
                     basecolor=mean_default("basecolor", [0.8, 0.8, 0.8]))
    return dict(mats=mats, has=has, emission_max=max(strengths + [0.0]) or 1.0, replaced=replaced,
                vertex_colours=vertex, unlit=[m.name for m in mats if find_shader(m)[2] == "EMISSION"],
                missing=missing, constants=constants)


class EmitOverride:
    """Temporarily drive the surface of every material with an Emission shader that shows one map of it
    ("WHITE" = plain white, for the hit masks). A tuple of up to three one-value maps shows them together, one in
    each colour channel."""

    def __init__(self, mats, what, emission_max=1.0):
        self.mats, self.what, self.emission_max, self.undo = mats, what, emission_max, []

    def __enter__(self):
        for mat in self.mats:
            tree = mat.node_tree
            out = find_shader(mat)[0]
            old = out.inputs["Surface"].links[0].from_socket
            em = tree.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            if isinstance(self.what, tuple):
                mix = tree.nodes.new("ShaderNodeCombineColor")
                for socket, what in zip(mix.inputs[:3], self.what):   # Combine Color: 0 R, 1 G, 2 B
                    value = source_of(mat, what)
                    if hasattr(value, "is_linked"):
                        if value.is_linked:
                            tree.links.new(value.links[0].from_socket, socket)
                            continue
                        value = value.default_value
                    socket.default_value = value if isinstance(value, float) else float(np.mean(tuple(value)[:3]))
                tree.links.new(mix.outputs[0], em.inputs["Color"])
                tree.links.new(em.outputs[0], out.inputs["Surface"])
                self.undo.append((tree, out, old, em, mix))
                continue
            value = (1.0, 1.0, 1.0) if self.what == "WHITE" else source_of(mat, self.what)
            if self.what == "emission":
                em.inputs["Strength"].default_value = emission_strength(mat) / self.emission_max
            if hasattr(value, "is_linked"):
                if value.is_linked:
                    tree.links.new(value.links[0].from_socket, em.inputs["Color"])
                    value = None
                else:
                    value = value.default_value
            if value is not None:
                rgb = (value,) * 3 if isinstance(value, float) else tuple(value)[:3]
                em.inputs["Color"].default_value = rgb + (1.0,)
            tree.links.new(em.outputs[0], out.inputs["Surface"])
            self.undo.append((tree, out, old, em, None))
        return self

    def __exit__(self, *exc):
        for tree, out, old, em, mix in self.undo:
            tree.links.new(old, out.inputs["Surface"])
            tree.nodes.remove(em)
            if mix is not None:
                tree.nodes.remove(mix)


# ---------------------------------------------------------------------------------------------- baking

def cage_distances(dev):
    """(first cage, retry cage) in working units from the measured deviation."""
    first = min(1.5, max(0.02, 1.5 * dev["p99"]))
    retry = min(3.0, max(2.0 * dev["max"], 4.0 * first))
    return first, retry


def _double(a):
    """A one-channel uint8 image at twice the size, bilinear."""
    a = a.astype(np.float32)
    for axis in (0, 1):
        a = np.moveaxis(a, axis, 0)
        p = np.concatenate([a[:1], a, a[-1:]])
        b = np.empty((2 * len(a),) + a.shape[1:], np.float32)
        b[0::2] = 0.25 * p[:-2] + 0.75 * p[1:-1]
        b[1::2] = 0.75 * p[1:-1] + 0.25 * p[2:]
        a = np.moveaxis(b, 0, axis)
    return np.round(a).astype(np.uint8)


def _to_bytes(a):
    return np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


class Baker:
    """Bakes `src` onto `low`. Everything else in the scene must be hidden from render by the caller."""

    def __init__(self, src, low, size, source_mats, emission_max, progress=lambda text: None):
        self.src, self.low, self.size = src, low, size
        self.source_mats, self.emission_max, self.progress = source_mats, emission_max, progress
        self.times = {}
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.view_settings.view_transform = "Standard"
        self.scene, self.settings = scene, scene.render.bake
        self.settings.margin_type = "EXTEND"
        self.settings.use_clear = True
        self.settings.use_cage = False
        self.settings.target = "IMAGE_TEXTURES"
        self.margin = max(2, size // 128)
        self.mat = ensure_nodes(bpy.data.materials.new("bake_target"))
        low.data.materials.clear()
        low.data.materials.append(self.mat)

    def _bake(self, tag, colorspace, samples, cage, emit=None, kind=None, margin=None, self_bake=False, size=None,
              **kw):
        """One bake into a fresh image; returns its RGB as uint8 (height, width, 3)."""
        tree = self.mat.node_tree
        size = size or self.size
        im = bpy.data.images.new("bake_" + tag, size, size, alpha=False)
        im.colorspace_settings.name = colorspace
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = im
        tree.nodes.active = node
        self.scene.cycles.samples = samples
        s = self.settings
        s.margin = self.margin if margin is None else margin
        s.use_selected_to_active = not self_bake
        s.cage_extrusion = cage
        s.max_ray_distance = 2.0 * cage          # the ray starts on the cage: this covers +-cage around the low mesh
        if self_bake:
            util.activate(self.low)
        else:
            util.activate(self.src, self.low)
        t = time.perf_counter()
        if emit is not None:
            with EmitOverride([self.mat] if self_bake else self.source_mats, emit, self.emission_max):
                bpy.ops.object.bake(type="EMIT")
        else:
            bpy.ops.object.bake(type=kind, **kw)
        self.times[tag] = round(self.times.get(tag, 0.0) + time.perf_counter() - t, 2)
        out = _to_bytes(util.image_pixels(im)[..., :3])
        tree.nodes.remove(node)
        bpy.data.images.remove(im)
        return out

    def _mask(self, tag, cage, self_bake=False):
        return self._bake(tag, "Non-Color", SAMPLES["mask"], cage, emit="WHITE", margin=0, self_bake=self_bake)[..., 0] > 127

    def _map(self, name, cage):
        if name == "normal":
            return self._bake(name, "Non-Color", SAMPLES["normal"], cage, kind="NORMAL", normal_space="TANGENT")
        if name == "ao":
            if self.size > AO_FULL_SIZE_MAX:
                half = self.size // 2
                return _double(self._bake(name, "Non-Color", SAMPLES["ao"], cage, kind="AO", size=half,
                                          margin=max(2, half // 128))[..., 0])
            return self._bake(name, "Non-Color", SAMPLES["ao"], cage, kind="AO")[..., 0]
        colour = name in ("basecolor", "emission")
        a = self._bake(name, "sRGB" if colour else "Non-Color", SAMPLES["emit"], cage, emit=name)
        return a if colour else a[..., 0]

    def _maps(self, names, cage, say=True):
        """Every map in `names`. The one-value maps (roughness, metallic, alpha) share one bake."""
        out = {}
        together = tuple(n for n in names if n in SCALARS)
        for name in names:
            if name in out:
                continue
            if name in together and len(together) > 1:
                if say:
                    self.progress("Baking: " + ", ".join(together))
                a = self._bake("+".join(together), "Non-Color", SAMPLES["emit"], cage, emit=together)
                for channel, key in enumerate(together):
                    out[key] = np.ascontiguousarray(a[..., channel])
                continue
            if say:
                self.progress("Baking: " + name.replace("basecolor", "base colour"))
            out[name] = self._map(name, cage)
        return out

    def _retry_patch(self, missed):
        """A copy of the low mesh with only the triangles that hold missed texels, plus every triangle that shares
        a vertex with one of them (so normals and tangents of the kept triangles stay what they are on the whole
        mesh). The second pass bakes onto it: its cost follows the missed texels, not the whole texture.
        None when the cut-out would be most of the mesh."""
        me = self.low.data
        n = len(me.polygons)
        if n == 0 or len(me.loops) != 3 * n:
            return None
        uv = np.empty(len(me.loops) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(n, 3, 2) * self.size
        lo = np.clip(np.floor(uv.min(1)).astype(np.int64) - 1, 0, self.size)
        hi = np.clip(np.ceil(uv.max(1)).astype(np.int64) + 1, 0, self.size)
        total = np.zeros((self.size + 1, self.size + 1), np.int32)      # up to 16.7 M texels fit
        total[1:, 1:] = missed.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)                 # [row = v, column = u], row 0 at the bottom
        count = total[hi[:, 1], hi[:, 0]] - total[lo[:, 1], hi[:, 0]] - total[hi[:, 1], lo[:, 0]] + total[lo[:, 1], lo[:, 0]]
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("vertex_index", lv)
        lv = lv.reshape(n, 3)
        touched = np.zeros(len(me.vertices), bool)
        touched[lv[count > 0]] = True
        keep = touched[lv].any(1)
        if not keep.any() or keep.mean() > RETRY_PATCH_SHARE_MAX:
            return None
        patch = util.copy_object(self.low, "bake_retry")
        bm = bmesh.new()
        bm.from_mesh(patch.data)
        bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, keep) if not k], context="FACES")
        bm.to_mesh(patch.data)
        bm.free()
        return patch

    def _second_pass(self, names, missed, retry_cage):
        """Bake the missed texels again with the wide cage. Returns (texels that now have a source, their maps)."""
        low, patch = self.low, self._retry_patch(missed)
        if patch is not None:
            low.hide_render = True                 # the whole low mesh must not stand between the cut-out and the source
            patch.hide_render = False
            self.low = patch
        try:
            fix = missed & self._mask("hit_retry", retry_cage)
            maps = self._maps(names, retry_cage, say=False) if fix.any() else {}
        finally:
            self.low = low
            if patch is not None:
                util.remove_object(patch)
                low.hide_render = False
        return fix, maps

    def run(self, wanted, cage, retry_cage):
        """Bake the normal map and every map in `wanted`. Returns (arrays by name, texel statistics)."""
        names = ["normal"] + [m for m in MAPS if m in wanted]
        self.progress("Baking: coverage")
        cover = self._mask("coverage", cage, self_bake=True)
        hit = self._mask("hit", cage)
        out = self._maps(names, cage)
        missed = cover & ~hit
        n_cover, n_missed = int(cover.sum()), int(missed.sum())
        stats = dict(texels=n_cover, fill=round(n_cover / float(self.size * self.size), 4),
                     missed_first=n_missed, missed=n_missed, retried=False,
                     cage=round(cage, 4), retry_cage=round(retry_cage, 4))
        if n_cover == 0 or not (cover & hit).any():
            stats["empty"] = True
            return out, stats
        if n_missed >= max(RETRY_MIN_TEXELS, RETRY_MIN_SHARE * n_cover):
            # second chance for the missed texels only: a wide cage. AO keeps the neighbours' value (the margin
            # filled it), a second AO pass would cost as much as all other maps together.
            self.progress("Baking: second pass for %d missed texels" % n_missed)
            fix, again = self._second_pass([n for n in names if n != "ao"], missed, retry_cage)
            for name, pixels in again.items():
                out[name][fix] = pixels[fix]
            stats.update(retried=True, missed=int((missed & ~fix).sum()))
        nz = out["normal"][..., 2]
        stats["back_normal"] = int((cover & (nz < 128)).sum())
        if "ao" in out:
            stats["black_ao"] = int((cover & (out["ao"] < 6)).sum())
        if "alpha" in out and int(out["alpha"][cover].min()) >= 252:
            del out["alpha"]                       # opaque everywhere: no cut-out material for nothing
        if "alpha" in out:
            stats["alpha_under_half_share"] = round(float((out["alpha"][cover] < 128).mean()), 5)
        if "emission" in out:
            stats["emission_lit_share"] = round(float((out["emission"][cover].max(1) > 8).mean()), 5)
        stats["seconds"] = dict(self.times)
        return out, stats

    def finish(self):
        self.low.data.materials.clear()
        bpy.data.materials.remove(self.mat)


# ---------------------------------------------------------------------------------------------- textures

def save_image(name, pixels, colorspace, folder):
    """uint8 (h, w, 3 or 4) -> a PNG-packed image datablock. The packed bytes travel inside the result file, so
    nothing depends on the temporary folder afterwards."""
    h, w, channels = pixels.shape
    alpha = channels == 4
    im = bpy.data.images.new(name, w, h, alpha=alpha)
    im.colorspace_settings.name = colorspace
    if alpha:
        im.alpha_mode = "CHANNEL_PACKED"
    full = np.ones((h, w, 4), np.float32)
    full[..., :channels] = pixels.astype(np.float32) / 255.0
    im.pixels.foreach_set(full.ravel())
    del full
    safe = re.sub(r'[:*?"<>|/\\]', "_", name).rstrip(" .") or "texture"      # Windows-safe file name
    im.filepath_raw = os.path.join(folder, safe + ".png")
    im.file_format = "PNG"
    im.save()
    im.pack()
    im.filepath_raw = "//textures/" + safe + ".png"
    for pf in im.packed_files:
        pf.filepath = im.filepath_raw
    return im


def build_textures(baked, name, folder, constants):
    """Pack the baked arrays into the texture set of the asset:
         <name>_normal     tangent space, OpenGL (+Y), as Blender bakes and glTF wants it
         <name>_basecolor  sRGB, alpha in A when the source has alpha
         <name>_orm        R = ambient occlusion, G = roughness, B = metallic (glTF / Unreal packing)
         <name>_emission   sRGB, scaled so that the strength on the material restores the source
    `constants` gives the value of a channel whose map was switched off. Returns {kind: image}."""
    images = {"normal": save_image(name + "_normal", baked["normal"], "Non-Color", folder)}
    if "basecolor" in baked or "alpha" in baked:
        rgb = baked.get("basecolor")
        if rgb is None:
            linear = np.array(constants["basecolor"], np.float32)
            srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
            rgb = np.empty(baked["normal"].shape, np.uint8)
            rgb[...] = _to_bytes(srgb)
        if "alpha" in baked:
            rgb = np.concatenate([rgb, baked["alpha"][..., None]], 2)
        images["basecolor"] = save_image(name + "_basecolor", rgb, "sRGB", folder)
    if any(k in baked for k in ("ao", "roughness", "metallic")):
        orm = np.empty(baked["normal"].shape, np.uint8)
        orm[..., 0] = baked["ao"] if "ao" in baked else 255
        orm[..., 1] = baked["roughness"] if "roughness" in baked else _to_bytes(np.float32(constants["roughness"]))
        orm[..., 2] = baked["metallic"] if "metallic" in baked else _to_bytes(np.float32(constants["metallic"]))
        images["orm"] = save_image(name + "_orm", orm, "Non-Color", folder)
    if "emission" in baked:
        images["emission"] = save_image(name + "_emission", baked["emission"], "sRGB", folder)
    return images
