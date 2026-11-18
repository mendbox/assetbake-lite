# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 2: preclean the working mesh (bounding-box diagonal 100).

Order matters (Findings 6 and 9): AI meshes arrive cut into pieces with doubled vertices on every UV seam, so
vertices are merged FIRST, across all joined objects; anything that looks at the mesh before that sees garbage.

Orientation is made safe: recalc_face_normals can turn a whole island inside out. After it,
  - an open island that was turned over as a whole goes back to the winding the file had;
  - a closed island is oriented by the sign of its volume.
"""
import math

import bmesh
import bpy
import numpy as np

from . import util

MERGE_DIST = 3e-4          # working units: 0.0003 % of the model size
SHARP_ANGLE = math.radians(30.0)
FLAT_SOURCE_SHARE = 0.5      # the by-angle rule runs only when more than this share of the faces is flat


def _face_normals(me):
    return util.poly_array(me, "normal", np.float32, 3)


def analyse(me):
    """Islands of a mesh: label per face, number of islands, open flag per island, edge counts."""
    n_verts, n_edges = len(me.vertices), len(me.edges)
    ev = np.empty(n_edges * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    vert_comp = util.vertex_components(n_verts, ev[:, 0], ev[:, 1])
    loop_start = util.poly_array(me, "loop_start", np.int32)
    loop_vert = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_vert)
    loop_edge = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", loop_edge)
    face_comp = vert_comp[loop_vert[loop_start]] if len(loop_start) else np.zeros(0, np.int64)
    # islands are numbered over vertices; loose vertices would add empty islands, so renumber over faces
    used, face_comp = np.unique(face_comp, return_inverse=True)
    face_comp = face_comp.reshape(-1)
    n_comp = len(used)
    remap = np.full(int(vert_comp.max()) + 1 if n_verts else 1, -1, np.int64)
    remap[used] = np.arange(n_comp)
    faces_per_edge = np.bincount(loop_edge, minlength=n_edges)
    open_edges = faces_per_edge != 2
    is_open = np.zeros(n_comp, bool)
    edge_comp = remap[vert_comp[ev[:, 0]]] if n_edges else np.zeros(0, np.int64)
    sel = open_edges & (edge_comp >= 0)
    is_open[edge_comp[sel]] = True
    return dict(face_comp=face_comp, n_comp=n_comp, is_open=is_open,
                boundary_edges=int((faces_per_edge == 1).sum()), nonmanifold_edges=int((faces_per_edge > 2).sum()))


def _island_volumes(me, face_comp, n_comp):
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    tri = np.empty(n * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    poly = np.empty(n, np.int32)
    me.loop_triangles.foreach_get("polygon_index", poly)
    co = util.vert_coords(me)[tri.reshape(-1, 3)]
    six_vol = np.einsum("ij,ij->i", co[:, 0], np.cross(co[:, 1], co[:, 2]))
    return np.bincount(face_comp[poly], weights=six_vol, minlength=n_comp)


def _face_order(bm):
    """(order, duplicates): the canonical order of the faces and the faces to drop. A face is keyed by its sorted
    vertex indices (so the key does not depend on the corner order of the file); faces on the same vertex set are
    duplicates, and the one that stays is the one whose corners, rotated to start at the smallest index, are
    lexicographically smallest."""
    rows = [[v.index for v in f.verts] for f in bm.faces]
    size = np.fromiter((len(r) for r in rows), np.int64, len(rows))
    width = int(size.max())
    big = len(bm.verts) + 1
    arr = np.full((len(rows), width), big, np.int64)
    if (size == width).all():
        arr[:] = np.array(rows, np.int64)
    else:
        for k in range(width):
            has = size > k
            arr[has, k] = [r[k] for r, h in zip(rows, has.tolist()) if h]
    srt = np.sort(arr, axis=1)
    start = np.argmin(arr, axis=1)
    idx = (np.arange(width)[None, :] + start[:, None]) % size[:, None]
    rot = np.take_along_axis(arr, np.minimum(idx, width - 1), axis=1)
    rot[np.arange(width)[None, :] >= size[:, None]] = big
    keys = np.concatenate([rot[:, ::-1], srt[:, ::-1]], axis=1)          # lexsort: the last key is the main one
    order = np.lexsort(keys.T)
    same = (srt[order][1:] == srt[order][:-1]).all(1)
    return order, order[1:][same]


def canonical_order(bm):
    """Put the elements of a bmesh into an order that depends on the geometry only: vertices by coordinates
    (x, then y, then z), faces by their vertex indices. The glTF importer ends with mesh.validate(), which drops
    one face of every pair of duplicate faces and does not drop the same one in every Blender process; so the
    same file gives corners in another order from run to run. Everything after this call sees the same arrays
    whatever the order of the source was. Returns the number of duplicate faces dropped."""
    if not len(bm.verts):
        return 0
    bm.verts.index_update()
    co = np.array([v.co[:] for v in bm.verts], np.float64).reshape(-1, 3)
    order = np.lexsort((co[:, 2], co[:, 1], co[:, 0]))            # equal coordinates stay in their order (stable)
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    rank = rank.tolist()
    bm.verts.sort(key=lambda v: rank[v.index])
    bm.verts.index_update()
    dropped = 0
    if len(bm.faces):
        bm.faces.index_update()
        order, duplicates = _face_order(bm)
        if len(duplicates):
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[bm.faces[i] for i in duplicates.tolist()], context="FACES_ONLY")
            dropped = len(duplicates)
            bm.faces.index_update()
            order, _ = _face_order(bm)
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        rank = rank.tolist()
        bm.faces.sort(key=lambda f: rank[f.index])
        bm.faces.index_update()
    return dropped


# corner attributes that rotate_corners can carry along: type -> (property of the attribute value, numpy type, width)
_CORNER_TYPES = dict(FLOAT=("value", np.float32, 1), INT=("value", np.int32, 1), BOOLEAN=("value", np.bool_, 1),
                     INT8=("value", np.int8, 1), FLOAT_VECTOR=("vector", np.float32, 3), FLOAT2=("vector", np.float32, 2),
                     FLOAT_COLOR=("color", np.float32, 4), BYTE_COLOR=("color_srgb", np.float32, 4))


def rotate_corners(me):
    """Start every face at its lowest vertex by coordinates (z, then y, then x; winding kept). Not "smallest
    index": the vertices are sorted by x first, and that start leaves Blender's UV packer a worse layout
    (kiosk atlas fill 0.24 against 0.53 here). The rule depends on the coordinates only.
    Which of two duplicate faces the importer or the merge keeps, and recalc_face_normals, leave faces with the same corners in another rotation from run to
    run; this makes the corner arrays one thing. Corner data (UVs, colours, other corner attributes) moves with
    the corners. Returns the number of faces rotated; -1 when a corner attribute of a type that cannot be carried
    is present (nothing is changed then)."""
    n = len(me.polygons)
    if not n:
        return 0
    start = util.poly_array(me, "loop_start", np.int64)
    total = util.poly_array(me, "loop_total", np.int64)
    n_loops = len(me.loops)
    carried = [a for a in me.attributes if a.domain == "CORNER" and not a.name.startswith(".")
               and a.name not in me.uv_layers]
    if any(a.data_type not in _CORNER_TYPES for a in carried):
        return -1
    lv = np.empty(n_loops, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    pos = np.arange(n_loops, dtype=np.int64)
    face = np.repeat(np.arange(n), total)
    co = util.vert_coords(me)
    order = np.lexsort((co[:, 0], co[:, 1], co[:, 2]))          # by z, then y, then x: depends on coordinates only
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    lr = rank[lv]
    low = np.minimum.reduceat(lr, start)
    first = np.minimum.reduceat(np.where(lr == low[face], pos, n_loops), start)
    shift = first - start
    if not shift.any():
        return 0
    take = start[face] + (pos - start[face] + shift[face]) % total[face]       # new corner i is the old corner take[i]
    me.loops.foreach_set("vertex_index", lv[take])
    for layer in me.uv_layers:
        uv = np.empty(n_loops * 2, np.float32)
        layer.data.foreach_get("uv", uv)
        layer.data.foreach_set("uv", uv.reshape(-1, 2)[take].ravel())
    for a in carried:
        prop, dtype, width = _CORNER_TYPES[a.data_type]
        values = np.empty(n_loops * width, dtype)
        a.data.foreach_get(prop, values)
        a.data.foreach_set(prop, values.reshape(n_loops, width)[take].ravel())
    me.update(calc_edges=True)
    return int((shift > 0).sum())


def preclean(ob):
    """Merge, drop degenerate faces, make the winding consistent and safe. Returns statistics and the island
    label of every face (face order of ob.data after the call). The result does not depend on the order of the
    vertices and faces of the source (canonical_order)."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    stats = dict(verts_in=len(bm.verts), faces_in=len(bm.faces))
    stats["duplicate_faces_dropped"] = canonical_order(bm)       # before the merge: it picks survivors in order
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=MERGE_DIST)
    bmesh.ops.dissolve_degenerate(bm, dist=MERGE_DIST, edges=bm.edges)
    stats["duplicate_faces_dropped"] += canonical_order(bm)
    bm.to_mesh(me)
    before = _face_normals(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    turned = np.einsum("ij,ij->i", before, _face_normals(me)) < 0
    del before

    info = analyse(me)
    face_comp, n_comp = info["face_comp"], info["n_comp"]
    faces_per = np.maximum(np.bincount(face_comp, minlength=n_comp), 1)
    turned_whole = np.bincount(face_comp, weights=turned, minlength=n_comp) / faces_per > 0.5
    inside_out = _island_volumes(me, face_comp, n_comp) < 0
    reverse = np.where(info["is_open"], turned_whole, inside_out)
    if reverse.any():
        bm.faces.ensure_lookup_table()
        faces = [bm.faces[i] for i in np.flatnonzero(reverse[face_comp]).tolist()]
        bmesh.ops.reverse_faces(bm, faces=faces)
        bm.to_mesh(me)
    bm.free()

    # imported custom normals survive the merge and would win over the winding when Cycles shades and bakes
    util.clear_custom_normals(ob)
    stats["faces_rotated"] = rotate_corners(me)
    # Shading: a smooth source stays as it is. A source with flat faces keeps its hard edges (a cube must not
    # come out as a blob) but loses the facets of a dense flat-shaded mesh, which AI files without normals have:
    # smooth everywhere, sharp where two faces meet at more than 30 degrees. Sharp edges of the file are kept.
    # Only when most faces are flat: a few planar faces in an otherwise smooth file must not turn every 30+ degree
    # edge of that file into a hard crease.
    flat = bool(len(me.polygons)) and float((~util.poly_array(me, "use_smooth", bool)).mean()) > FLAT_SOURCE_SHARE
    if flat:
        util.activate(ob)
        bpy.ops.object.shade_smooth_by_angle(angle=SHARP_ANGLE, keep_sharp_edges=True)
    me.update()
    stats.update(
        verts=len(me.vertices), faces=len(me.polygons), merged_verts=stats["verts_in"] - len(me.vertices),
        islands=int(n_comp), flat_source=flat, faces_turned_by_recalc=int(turned.sum()),
        open_islands_turned_back=int((reverse & info["is_open"]).sum()),
        closed_islands_flipped_by_volume=int((reverse & ~info["is_open"]).sum()),
        boundary_edges=info["boundary_edges"], nonmanifold_edges=info["nonmanifold_edges"])
    return stats, face_comp
