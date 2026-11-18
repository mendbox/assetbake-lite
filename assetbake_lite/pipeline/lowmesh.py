# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 3: the low mesh.

Game mode = Decimate (collapse) to the triangle budget. It runs in two stages:
  source -> "mid" mesh (4 x budget, at least 60k triangles) -> low mesh.
The mid mesh is cheap to work on: surface that nothing outside can see is removed from it before the second
stage (Finding 12: 2-5 % of a multi-part AI model is shut in between parts and only wastes triangles and texels),
and the corrective passes that bring the result to the budget start from it, not from the heavy source.

Decimate cannot collapse every mesh. Edges shared by three or more faces, and faces that touch at single points,
stay as they are (TRELLIS files have tens of thousands of them): on such a source Decimate stops far above the
target, and before it stops it spends every triangle of the healthy surface. reduce() therefore measures the
floor (what is left when Decimate has nothing more to collapse) and, when the floor is a large part of the
target, first merges the stuck vertices to the vertex spacing the target asks for.

LOW_MESH_BUILDERS is the hook for stage 2: a "QUADS" builder returns the same kind of result.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import util

MID_FACTOR = 4
MID_MIN_TRIS = 60000
BUDGET_TOLERANCE = 0.01        # the corrective loop stops inside +-1 % (the promise to the user is +-5 %)
BUDGET_PROMISE = 0.05          # what the user is told: the asset is within 5 % of the budget
MID_TOLERANCE = 0.25           # the mid mesh only has to be roughly its size
FLOOR_SHARE = 0.5              # geometry that Decimate cannot collapse may take up to half of the target
UNLOCK_ROUNDS = 4              # merge passes over the stuck vertices; the distance grows 1.5 x per pass
SHARE_ROUNDS = 3               # ... of which the first three may run only to bring the floor down (the target is met)
PROBE_RATIO = 1e-6             # Decimate ratio that collapses everything it can
MERGE_MAX_EDGES = 4.0          # last resort: merge distance up to this many target edge lengths
MID_BOOST_MAX = 8.0            # a second, larger mid mesh when the hidden surface took too much of the first
HIDDEN_MIN_SHARE = 0.005       # hidden surface under 0.5 % of the area is left alone
RAY_START = 2e-3               # working units; rays start this far off the source surface
DENSE_RAYS = 256               # second pass per sample point: an opening of 1 % of the hemisphere is found


def _strip_for_decimate(ob):
    """UV layers, colours and vertex groups only slow Decimate down: the low mesh gets new UVs anyway."""
    me = ob.data
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    while me.color_attributes:
        me.color_attributes.remove(me.color_attributes[0])
    ob.vertex_groups.clear()


def _triangulate(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()


def decimated_copy(ob, ratio, name):
    """Copy of `ob` with a collapse Decimate applied, triangles only."""
    low = util.copy_object(ob, name)
    util.activate(low)
    if ratio < 1.0:
        mod = low.modifiers.new("decimate", "DECIMATE")
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)
    _triangulate(low.data)
    return low


def decimate_to(ob, target, name, tolerance=BUDGET_TOLERANCE):
    """Decimate to `target` triangles, with corrective passes: the modifier's ratio is only approximately kept
    on meshes with many parts and open borders. Returns (object, passes)."""
    n_src = util.tri_count(ob.data)
    ratio = min(1.0, target / max(1, n_src))
    best = None
    before = None
    for attempt in range(4):
        low = decimated_copy(ob, ratio, name)
        n = util.tri_count(low.data)
        if best is None or abs(n - target) < abs(util.tri_count(best.data) - target):
            if best is not None:
                util.remove_object(best)
            best = low
        else:
            util.remove_object(low)
        if abs(n - target) <= tolerance * target or ratio >= 1.0 or n == 0:
            break
        if before is not None and n > target and n >= 0.995 * before:
            break                                  # a lower ratio changed nothing: Decimate has nothing left to collapse
        before = n
        ratio = min(1.0, ratio * target / n)
    best.name = name
    return best, attempt + 1


def _edge_length(me, target):
    """Edge length of an even mesh of `target` triangles over the surface of `me`."""
    area = float(util.poly_array(me, "area", np.float64).sum())
    return math.sqrt(4.0 * area / (math.sqrt(3.0) * max(1, target)))


def _coord_keys(me):
    """One number per vertex that is equal for vertices with bit-identical coordinates."""
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    bits = co.view(np.uint32).reshape(-1, 3).astype(np.uint64)
    return (bits[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (bits[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) ^ (
        bits[:, 2] * np.uint64(0x165667B19E3779F9))


def _drop_bad(bm):
    """Zero-area faces, faces on the same three vertices as another one, and what is left without a face."""
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    seen = set()
    doubles = []
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen:
            doubles.append(f)
        else:
            seen.add(key)
    if doubles:
        bmesh.ops.delete(bm, geom=doubles, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.triangulate(bm, faces=bm.faces)


def _merge(ob, dist, keys=None):
    """Merge vertices closer than `dist`: all of them, or only those whose coordinates are in `keys`."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    verts = bm.verts
    if keys is not None:
        bm.verts.ensure_lookup_table()
        verts = [bm.verts[i] for i in np.flatnonzero(np.isin(_coord_keys(me), keys)).tolist()]
    bmesh.ops.remove_doubles(bm, verts=verts, dist=dist)
    bm.verts.index_update()
    _drop_bad(bm)
    bm.to_mesh(me)
    bm.free()


def merge_to(ob, target, name):
    """Last resort when Decimate stays above the target: merge ALL vertices by distance, the distance found by
    bisection. Works on any topology; the shape suffers more than with Decimate. Returns a new object."""
    lo, hi = 0.0, MERGE_MAX_EDGES * _edge_length(ob.data, target)
    best = None
    for _ in range(14):
        dist = 0.5 * (lo + hi)
        trial = util.copy_object(ob, name)
        _merge(trial, dist)
        n = util.tri_count(trial.data)
        if best is None or abs(n - target) < abs(util.tri_count(best.data) - target):
            if best is not None:
                util.remove_object(best)
            best = trial
        else:
            util.remove_object(trial)
        if abs(n - target) <= BUDGET_TOLERANCE * target:
            break
        if n > target:
            lo = dist
        else:
            hi = dist
    best.name = name
    return best


def reduce(ob, target, name, tolerance=BUDGET_TOLERANCE, force=False):
    """Decimate `ob` (not changed) to `target` triangles, also when part of it cannot be collapsed.

    After every Decimate the floor is measured: the triangles left when Decimate collapses all it can. A floor
    above FLOOR_SHARE of the target means stuck geometry holds the triangles and the healthy surface pays for it
    (or the target is not reached at all). The vertices that survive in the floor are the stuck ones (a collapse
    makes a new vertex, a stuck vertex keeps its coordinates): they are merged by distance, starting at the edge
    length the target stands for, and Decimate runs again.
    force: when the result is still more than BUDGET_PROMISE above the target, merge all vertices by distance.
    Returns (object, info)."""
    info = dict(passes=0, floor=0, unlock_rounds=0, merged_all=False)
    step = _edge_length(ob.data, target)
    near, far = 0.0, None                # merge distances: the largest one used, the smallest one that was too much
    work = ob
    for rnd in range(UNLOCK_ROUNDS + 1):
        low, passes = decimate_to(work, target, name, tolerance)
        info["passes"] += passes
        probe = decimated_copy(low, PROBE_RATIO, name + "_probe")
        floor = util.tri_count(probe.data)
        if rnd == 0:
            info["floor"] = floor
        over = util.tri_count(low.data) > target * (1 + max(tolerance, BUDGET_PROMISE))
        unlock = floor > FLOOR_SHARE * target and rnd < (UNLOCK_ROUNDS if over else SHARE_ROUNDS)
        keys = _coord_keys(probe.data) if unlock else None
        util.remove_object(probe)
        if not unlock:
            break
        # merge the stuck vertices, but never so far that fewer triangles than the target are left
        merged = None
        for _ in range(5):
            dist = max(step, 1.5 * near) if far is None else 0.5 * (near + far)
            trial = util.copy_object(work, name + "_unlock")
            _merge(trial, dist, keys)
            if util.tri_count(trial.data) >= target:
                merged, near = trial, dist
                break
            util.remove_object(trial)
            far = dist
        if merged is None:
            break
        util.remove_object(low)
        if work is not ob:
            util.remove_object(work)
        work = merged
        info["unlock_rounds"] += 1
    if work is not ob:
        util.remove_object(work)
    if force and util.tri_count(low.data) > target * (1 + BUDGET_PROMISE):
        merged = merge_to(low, target, name + "_merged")
        util.remove_object(low)
        low = merged
        info["merged_all"] = True
    low.name = name
    return low, info


def remove_bad_triangles(me):
    """Triangles on the same three vertices as another one and zero-area triangles: both break tangents on export
    (Finding 12). Returns how many were removed."""
    bm = bmesh.new()
    bm.from_mesh(me)
    n0 = len(bm.faces)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    seen = set()
    doubles = []
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen:
            doubles.append(f)
        else:
            seen.add(key)
    if doubles:
        bmesh.ops.delete(bm, geom=doubles, context="FACES")
    bmesh.ops.triangulate(bm, faces=bm.faces)
    removed = n0 - len(bm.faces)
    bm.to_mesh(me)
    bm.free()
    return removed


# ---------------------------------------------------------------------------------------------- hidden surface

def _ray_directions():
    """13 directions in the hemisphere around +Z: the pole, six at 45 degrees, six at 75 degrees."""
    out = [(0.0, 0.0, 1.0)]
    for tilt, phase in ((45.0, 0.0), (75.0, 30.0)):
        s, c = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
        for k in range(6):
            a = math.radians(phase + 60.0 * k)
            out.append((s * math.cos(a), s * math.sin(a), c))
    return out


def _fibonacci_hemisphere(n):
    """n directions spread evenly over the hemisphere around +Z."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = (i + 0.5) / n
        r = math.sqrt(1.0 - z * z)
        out.append((r * math.cos(golden * i), r * math.sin(golden * i), z))
    return out


_DIRECTIONS = _ray_directions()                    # quick first pass: most visible faces leave on the first ray
_DENSE_DIRECTIONS = _fibonacci_hemisphere(DENSE_RAYS)


def _shut_in(bvh, start, normal, directions=_DIRECTIONS):
    """True when every ray from `start` into the hemisphere around `normal` hits the source."""
    helper = Vector((1.0, 0.0, 0.0)) if abs(normal.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    t = normal.cross(helper).normalized()
    b = normal.cross(t)
    for x, y, z in directions:
        if bvh.ray_cast(start, t * x + b * y + normal * z)[0] is None:
            return False
    return True


def hidden_faces(low, src_bvh):
    """Mask of the faces of `low` (triangles) that nothing outside can see.

    Four points per face are tested. A point is hidden only when the source blocks every ray on BOTH sides of the
    surface: a face that merely points the wrong way has an open back and is kept.
    Two passes: 13 rays per side find the candidates (a few percent of the faces), then every candidate is tested
    again with DENSE_RAYS directions on its front side, so surface seen through a small opening (a lantern, a
    helmet, a room with a door) is kept.
    """
    me = low.data
    co = util.vert_coords(me)
    tri = np.empty(len(me.polygons) * 3, np.int32)
    me.polygons.foreach_get("vertices", tri)
    corners = co[tri.reshape(-1, 3)]
    centre = corners.mean(1)
    normals = util.poly_array(me, "normal", np.float64, 3)
    samples = np.concatenate([centre[:, None], 0.6 * centre[:, None] + 0.4 * corners], 1)
    hidden = np.zeros(len(centre), bool)
    for i in range(len(centre)):
        n = Vector(normals[i])
        if n.length_squared < 0.5:
            continue
        starts = []
        for p in samples[i]:
            on_source = src_bvh.find_nearest(Vector(p))[0]
            if on_source is None:
                break
            front = on_source + n * RAY_START
            if not (_shut_in(src_bvh, front, n) and _shut_in(src_bvh, on_source - n * RAY_START, -n)):
                break
            starts.append(front)
        else:
            hidden[i] = all(_shut_in(src_bvh, front, n, _DENSE_DIRECTIONS) for front in starts)
    return hidden


def _core(me, mask):
    """Faces of the mask whose edge neighbours are all in the mask too: one ring of hidden faces stays as a
    safety margin towards the visible surface."""
    n_edges = len(me.edges)
    loop_edge = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", loop_edge)
    loop_face = np.repeat(np.arange(len(me.polygons)), util.poly_array(me, "loop_total", np.int32))
    edge_open = np.bincount(loop_edge, weights=~mask[loop_face], minlength=n_edges) > 0
    touches_open = np.bincount(loop_face, weights=edge_open[loop_edge], minlength=len(me.polygons)) > 0
    return mask & ~touches_open


def remove_hidden(mid, src_bvh):
    """Delete the hidden core of `mid`. Returns (share of the area removed, removed triangles as a (n, 3, 3)
    coordinate array). The removed triangles are kept for the deviation measure: the source still has surface
    there, and it must not count as "the low mesh is far from the source"."""
    me = mid.data
    core = _core(me, hidden_faces(mid, src_bvh))
    area = util.poly_array(me, "area", np.float64)
    share = float(area[core].sum() / max(area.sum(), 1e-30))
    if share < HIDDEN_MIN_SHARE:
        return 0.0, None
    tri = np.empty(len(me.polygons) * 3, np.int32)
    me.polygons.foreach_get("vertices", tri)
    ghost = util.vert_coords(me)[tri.reshape(-1, 3)[core]]
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.flatnonzero(core).tolist()], context="FACES")
    bm.to_mesh(me)
    bm.free()
    return share, ghost


# ---------------------------------------------------------------------------------------------- builders

def build_game(src, budget, name, src_bvh=None, drop_hidden=True, progress=lambda text: None):
    """Game mode. Returns (asset object, bake object, stats). The asset is all triangles, so the mesh that is
    baked and the mesh the user gets are the same object. stats["ghost"] holds the removed hidden triangles."""
    n_src = util.tri_count(src.data)
    stats = dict(tris_in=n_src, budget=int(budget), under_budget=False, hidden_removed_share=0.0, passes=0,
                 ghost=None, floor=0, unlock_rounds=0, merged_all=False, mid_boost=1.0, short="")
    work = util.copy_object(src, name + "_mid")
    _strip_for_decimate(work)
    if n_src <= budget * (1 + BUDGET_TOLERANCE):
        stats["under_budget"] = True
        _triangulate(work.data)
        low = work
    else:
        boost = 1.0
        while True:
            mid_target = int(max(MID_FACTOR * budget, MID_MIN_TRIS) * boost)
            reduced = n_src > mid_target
            unlocked = 0
            if reduced:
                progress("Reducing the source")
                mid, info = reduce(work, mid_target, name + "_mid2", tolerance=MID_TOLERANCE)
                unlocked = info["unlock_rounds"]
            else:
                mid = util.copy_object(work, name + "_mid2")
                _triangulate(mid.data)
            stats["ghost"], stats["hidden_removed_share"] = None, 0.0
            if drop_hidden and src_bvh is not None:
                progress("Looking for hidden surface")
                share, stats["ghost"] = remove_hidden(mid, src_bvh)
                stats["hidden_removed_share"] = round(share, 4)
            progress("Decimating to the budget")
            low, info = reduce(mid, budget, name, force=True)
            util.remove_object(mid)
            stats.update(passes=info["passes"], floor=info["floor"], merged_all=info["merged_all"],
                         unlock_rounds=unlocked + info["unlock_rounds"], mid_boost=round(boost, 2))
            n = util.tri_count(low.data)
            if n >= budget * (1 - BUDGET_PROMISE) or not reduced or boost > 1.0:
                break
            # the hidden surface took so much of the mid mesh that fewer triangles than the budget are left:
            # once more from a larger mid mesh
            boost = min(MID_BOOST_MAX, max(2.0, 1.5 * budget / max(1, n)))
            util.remove_object(low)
        util.remove_object(work)
    low.name = low.data.name = name
    stats["bad_triangles_removed"] = remove_bad_triangles(low.data)
    stats["tris"] = util.tri_count(low.data)
    stats["verts"] = len(low.data.vertices)
    if not stats["under_budget"]:
        if stats["tris"] > budget * (1 + BUDGET_PROMISE):
            stats["short"] = "over"
        elif stats["tris"] < budget * (1 - BUDGET_PROMISE):
            stats["short"] = "under"
    return low, low, stats


LOW_MESH_BUILDERS = {
    "GAME": build_game,
    # "QUADS": stage 2 (quad-dominant hybrid). It returns a quad asset plus its loop_triangles copy for the bake.
}
