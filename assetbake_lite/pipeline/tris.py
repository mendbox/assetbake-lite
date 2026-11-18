# SPDX-License-Identifier: GPL-3.0-or-later
"""The fixed triangulation of a quad-dominant asset.

A normal map is only right on the triangles it was baked on (Finding 12: a quad asset exported as quads lets
every program pick its own diagonal, and the tangents no longer fit the map). So the diagonal of every quad is
decided once, when the low mesh is built, and stored in the corner order of the face: the split always runs from
corner 0 to corner 2. The bake, the GLB export and this module's triangulated copy all use exactly that split,
and so does Blender's own display of a convex quad.

Used in the worker (bake) and in the user's session (Export button). No compiled module is needed here.
"""
import bpy
import numpy as np


def order_quads(co, faces):
    """Rotate the corners of every quad so that the better diagonal runs from corner 0 to corner 2.

    Better = the shorter one (as in the research bakes), unless that split folds the quad over or leaves a
    triangle without area while the other split does neither. `co` is (n, 3), `faces` a list of index lists
    (3 or 4 corners). Returns a new list; triangles are passed through."""
    quads = [i for i, f in enumerate(faces) if len(f) == 4]
    if not quads:
        return list(faces)
    q = np.array([faces[i] for i in quads], np.int64)
    p = co[q]                                                    # (m, 4, 3)

    def split(a, b, c, d):
        """Triangles (a, b, c) and (a, c, d): fold flag and the smaller of the two areas."""
        n1 = np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a])
        n2 = np.cross(p[:, c] - p[:, a], p[:, d] - p[:, a])
        a1, a2 = np.linalg.norm(n1, axis=1), np.linalg.norm(n2, axis=1)
        return np.einsum("ij,ij->i", n1, n2) < 0, np.minimum(a1, a2), a1 + a2

    fold02, small02, total = split(0, 1, 2, 3)
    fold13, small13, _ = split(1, 2, 3, 0)
    bad02 = fold02 | (small02 < 1e-4 * total)
    bad13 = fold13 | (small13 < 1e-4 * total)
    d02 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    d13 = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
    use13 = np.where(bad02 != bad13, bad02, d13 < d02)
    q[use13] = np.roll(q[use13], -1, axis=1)
    out = list(faces)
    for i, row in zip(quads, q.tolist()):
        out[i] = row
    # Two faces that share three corners (a small fold in the mesh) can give the same triangle twice, which
    # breaks tangents on export (Finding 12): the quad among them takes its other diagonal.
    for _ in range(4):
        owners = {}
        for i, f in enumerate(out):
            for tri in ((f[0], f[1], f[2]), (f[0], f[2], f[3])) if len(f) == 4 else (tuple(f),):
                owners.setdefault(tuple(sorted(tri)), []).append(i)
        turn = sorted({max((i for i in group if len(out[i]) == 4), default=-1)
                       for group in owners.values() if len(group) > 1} - {-1})
        if not turn:
            break
        for i in turn:
            out[i] = out[i][1:] + out[i][:1]
    return out


def double_triangles(faces):
    """How many triangles of the fixed split lie on the same three vertices as another one."""
    seen, doubles = set(), 0
    for f in faces:
        for tri in ((f[0], f[1], f[2]), (f[0], f[2], f[3])) if len(f) == 4 else (tuple(f),):
            key = tuple(sorted(tri))
            doubles += key in seen
            seen.add(key)
    return doubles


def mesh_from_faces(name, co, faces):
    """A new mesh from coordinates and faces of three or four corners, corner order kept as given."""
    me = bpy.data.meshes.new(name)
    sizes = np.fromiter((len(f) for f in faces), np.int32, len(faces))
    corners = np.fromiter((i for f in faces for i in f), np.int32, int(sizes.sum()))
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", np.ascontiguousarray(co, np.float64).ravel())
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", corners)
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.cumsum(sizes) - sizes).astype(np.int32))
    me.polygons.foreach_set("loop_total", sizes)
    me.update(calc_edges=True)
    me.validate()
    return me


def is_fixed(me):
    """True when the mesh can take the fixed split: triangles and quads only."""
    n = len(me.polygons)
    if not n:
        return False
    total = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_total", total)
    return bool(((total == 3) | (total == 4)).all())


def topology_hash(me):
    """A short hash of the faces of a mesh (corner order and face sizes, vertex count; no coordinates): it changes
    when the faces are edited, not when vertices are moved. The asset mark keeps it to tell an edited asset."""
    import hashlib
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    size = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", size)
    return hashlib.sha1(lv.tobytes() + size.tobytes() + np.int64(len(me.vertices)).tobytes()).hexdigest()[:16]


def triangulated_mesh(me, name):
    """A triangle mesh of `me` (triangles and quads): every quad is split from corner 0 to corner 2. Vertices keep
    their index, every corner keeps its UVs, faces keep material and smooth flag; triangles come in face order."""
    n = len(me.polygons)
    start = np.empty(n, np.int32)
    total = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_start", start)
    me.polygons.foreach_get("loop_total", total)
    quad = total == 4
    # corners of the first triangle of every face, then one more row for the second triangle of every quad
    first = start[:, None] + np.arange(3)
    second = start[quad][:, None] + np.array([0, 2, 3])
    order = np.argsort(np.concatenate([np.arange(n), np.flatnonzero(quad)]), kind="stable")
    loops = np.concatenate([first, second])[order]               # (triangles, 3) corner indices into `me`
    face = np.concatenate([np.arange(n), np.flatnonzero(quad)])[order]
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)

    out = bpy.data.meshes.new(name)
    nt = len(loops)
    out.vertices.add(len(me.vertices))
    out.vertices.foreach_set("co", co)
    out.loops.add(3 * nt)
    out.loops.foreach_set("vertex_index", lv[loops].ravel().astype(np.int32))
    out.polygons.add(nt)
    out.polygons.foreach_set("loop_start", np.arange(0, 3 * nt, 3, dtype=np.int32))
    out.polygons.foreach_set("loop_total", np.full(nt, 3, np.int32))
    out.update(calc_edges=True)
    smooth = np.empty(n, bool)
    me.polygons.foreach_get("use_smooth", smooth)
    out.polygons.foreach_set("use_smooth", smooth[face])
    index = np.empty(n, np.int32)
    me.polygons.foreach_get("material_index", index)
    out.polygons.foreach_set("material_index", index[face])
    for layer in me.uv_layers:
        uv = np.empty(len(me.loops) * 2, np.float32)
        layer.data.foreach_get("uv", uv)
        new = out.uv_layers.new(name=layer.name)
        new.data.foreach_set("uv", uv.reshape(-1, 2)[loops.ravel()].ravel())
        new.active_render = layer.active_render
    if me.uv_layers.active is not None:
        out.uv_layers.active = out.uv_layers[me.uv_layers.active.name]
    for mat in me.materials:
        out.materials.append(mat)
    out.update()
    return out


def triangulated_copy(ob, name):
    """A new object next to `ob` with the fixed triangulation of its mesh (linked to the scene collection)."""
    new = bpy.data.objects.new(name, triangulated_mesh(ob.data, name))
    new.matrix_world = ob.matrix_world
    bpy.context.scene.collection.objects.link(new)
    return new
