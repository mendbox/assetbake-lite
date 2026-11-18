# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 1: bring the selected objects into the empty worker scene and make one clean working mesh of them.

Rules (Findings 9 and 11 of the research):
  - the parent chain's transform is kept: the add-on passes every object's world matrix in the job file;
  - helper meshes that are bone custom shapes are not part of the asset;
  - a rigged source is used in its rest shape; armature, weights, shape keys and animation are not carried over,
    and the report says so;
  - the join must really happen (a cancelled join once left the chain working on an 80-triangle helper).
"""
import bpy
import numpy as np
from mathutils import Matrix

from . import util

WORK_DIAGONAL = 100.0
UV_NAME = "UVMap"


def bone_shape_objects():
    return {pb.custom_shape for ob in bpy.data.objects if ob.type == "ARMATURE"
            for pb in ob.pose.bones if pb.custom_shape is not None}


def _is_rigged(ob):
    if any(m.type == "ARMATURE" for m in ob.modifiers):
        return True
    parent = ob.parent
    while parent is not None:
        if parent.type == "ARMATURE":
            return True
        parent = parent.parent
    return False


def _is_animated(ob):
    while ob is not None:
        if ob.animation_data is not None and ob.animation_data.action is not None:
            return True
        ob = ob.parent
    return False


def load_sources(job):
    """Append the job's objects from the source file. Returns (mesh objects, notes)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    wanted = {o["name"]: o["matrix"] for o in job["objects"]}
    with bpy.data.libraries.load(job["source_blend"], link=False) as (data_from, data_to):
        data_to.objects = [n for n in data_from.objects if n in wanted]
    shapes = bone_shape_objects()
    notes = dict(objects=0, rigged=False, animated=False, shape_keys=False, modifiers_applied=False,
                 bone_shapes_skipped=0)
    scene = bpy.context.scene
    meshes = []
    for ob in data_to.objects:
        if ob is None or ob.type != "MESH":
            continue
        if ob in shapes:
            notes["bone_shapes_skipped"] += 1
            continue
        notes["rigged"] |= _is_rigged(ob)
        notes["animated"] |= _is_animated(ob)
        meshes.append((ob, Matrix([wanted[ob.name][i * 4:i * 4 + 4] for i in range(4)])))

    needs_evaluation = []
    for ob, matrix in meshes:
        ob.data = ob.data.copy()                       # instances share mesh data: every object gets its own
        ob.animation_data_clear()
        ob.constraints.clear()
        for mod in list(ob.modifiers):
            if mod.type == "ARMATURE":
                ob.modifiers.remove(mod)               # rest shape on purpose
            elif mod.show_viewport:
                needs_evaluation.append(ob)
        if ob.data.shape_keys is not None:
            notes["shape_keys"] = True
            if ob not in needs_evaluation:
                ob.shape_key_clear()
        scene.collection.objects.link(ob)
        ob.parent = None
        ob.matrix_world = matrix
        ob.hide_viewport = ob.hide_render = False
        ob.hide_set(False)
    if needs_evaluation:
        # other modifiers (Mirror, Subdivision, ...) are part of what the user sees: take the evaluated mesh
        notes["modifiers_applied"] = True
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for ob in set(needs_evaluation):
            me = bpy.data.meshes.new_from_object(ob.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                                 depsgraph=depsgraph)
            ob.modifiers.clear()
            if ob.data.shape_keys is not None:
                ob.shape_key_clear()
            ob.data = me
    objects = [ob for ob, _ in meshes]
    notes["objects"] = len(objects)
    return objects, notes


def unify_uv_names(objects):
    """Give the UV layer that the renderer uses the same name on every object.

    object.join merges UV layers BY NAME: "UVMap" on one object and "map1" on another become two layers, each
    empty (0, 0) on the other object's faces, and one of the objects bakes in a single flat colour. Nodes that
    name the layer (UV Map, Normal Map) follow the rename.
    """
    renamed = 0
    for ob in objects:
        layers = ob.data.uv_layers
        render = next((layer for layer in layers if layer.active_render), None)
        if render is None or render.name == UV_NAME or UV_NAME in layers:
            continue
        old = render.name
        render.name = UV_NAME
        renamed += 1
        for slot in ob.material_slots:
            tree = slot.material.node_tree if slot.material else None
            for node in tree.nodes if tree else ():
                if node.type in ("UVMAP", "NORMAL_MAP") and node.uv_map == old:
                    node.uv_map = UV_NAME
    return renamed


def join(objects, name):
    """Join into one object with every transform applied. Raises when the join did not happen."""
    unify_uv_names(objects)
    util.activate(*objects)
    if len(objects) > 1:
        bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    left = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
    if left != [src]:
        raise RuntimeError("could not join the selected objects into one mesh")
    for ob in list(bpy.data.objects):
        if ob is not src:
            bpy.data.objects.remove(ob)
    util.activate(src)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    src.name = name
    if UV_NAME in src.data.uv_layers:
        src.data.uv_layers[UV_NAME].active_render = True
    return src


def normalise(ob):
    """Scale to a bounding-box diagonal of WORK_DIAGONAL and move the box centre to the origin.
    Returns (scale, centre): world = working / scale + centre."""
    co = util.vert_coords(ob.data)
    lo, hi = co.min(0), co.max(0)
    diagonal = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise RuntimeError("the selection has no size")
    scale = WORK_DIAGONAL / diagonal
    centre = (lo + hi) / 2.0
    util.set_vert_coords(ob.data, (co - centre) * scale)
    return scale, centre
