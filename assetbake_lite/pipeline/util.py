# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helpers shared by the pipeline modules."""
import bpy
import numpy as np


def activate(*objects):
    """Select exactly these objects; the last one becomes active."""
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    for ob in objects:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objects[-1]


def tri_count(me):
    """Triangles of a mesh after fan triangulation (no loop over the faces)."""
    return len(me.loops) - 2 * len(me.polygons)


def vert_coords(me):
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def set_vert_coords(me, co):
    me.vertices.foreach_set("co", np.ascontiguousarray(co, np.float64).ravel())
    me.update()


def poly_array(me, attr, dtype, width=1):
    a = np.empty(len(me.polygons) * width, dtype)
    me.polygons.foreach_get(attr, a)
    return a.reshape(-1, width) if width > 1 else a


def copy_object(ob, name):
    """Independent copy (object and mesh) linked to the scene."""
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = name
    new.data.name = name
    bpy.context.scene.collection.objects.link(new)
    return new


def remove_object(ob):
    me = ob.data
    bpy.data.objects.remove(ob)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def shade_smooth(me):
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))


def clear_custom_normals(ob):
    if ob.data.has_custom_normals:
        activate(ob)
        bpy.ops.mesh.customdata_custom_splitnormals_clear()


def vertex_components(n_verts, e0, e1):
    """Connected components of a graph given as two vertex index arrays (one entry per edge).
    Returns one label per vertex, 0..k-1. Vectorised label propagation with pointer jumping."""
    lab = np.arange(n_verts, dtype=np.int64)
    if len(e0):
        src = np.concatenate([e0, e1]).astype(np.int64)
        dst = np.concatenate([e1, e0]).astype(np.int64)
        order = np.argsort(dst, kind="stable")
        src, dst = src[order], dst[order]
        starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
        targets = dst[starts]
        while True:
            new = lab.copy()
            new[targets] = np.minimum(new[targets], np.minimum.reduceat(lab[src], starts))
            new = new[new]
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
    _, comp = np.unique(lab, return_inverse=True)
    return comp.reshape(-1)


def image_pixels(im):
    """Pixels as float32 (height, width, channels), row 0 at the bottom."""
    a = np.empty(im.size[0] * im.size[1] * im.channels, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(im.size[1], im.size[0], im.channels)
