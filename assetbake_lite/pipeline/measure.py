# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape deviation between the source and the low mesh, in working units (= percent of the model size)."""
import numpy as np
from mathutils.bvhtree import BVHTree

from . import util

MAX_SOURCE_SAMPLES = 300000


def bvh_of(ob, depsgraph):
    return BVHTree.FromObject(ob, depsgraph)


def _low_bvh(low, ghost):
    """BVH of the low mesh (triangles) plus the hidden triangles that were removed on purpose."""
    me = low.data
    co = util.vert_coords(me)
    tri = np.empty(len(me.polygons) * 3, np.int32)
    me.polygons.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    if ghost is not None and len(ghost):
        extra = np.arange(len(ghost) * 3).reshape(-1, 3) + len(co)
        co = np.concatenate([co, ghost.reshape(-1, 3)])
        tri = np.concatenate([tri, extra])
    return BVHTree.FromPolygons(co.tolist(), tri.tolist())


def deviation(src, src_bvh, low, ghost=None):
    """Two-sided distance: source vertices -> low surface, and low vertices + face centres -> source surface.
    p99 and max are the larger of the two directions."""
    co = util.vert_coords(src.data)
    if len(co) > MAX_SOURCE_SAMPLES:
        co = co[np.linspace(0, len(co) - 1, MAX_SOURCE_SAMPLES).astype(np.int64)]
    nearest = _low_bvh(low, ghost).find_nearest
    to_low = np.array([nearest(p)[3] or 0.0 for p in co.tolist()])
    me = low.data
    points = np.concatenate([util.vert_coords(me), util.poly_array(me, "center", np.float64, 3)])
    nearest = src_bvh.find_nearest
    to_src = np.array([nearest(p)[3] or 0.0 for p in points.tolist()])
    return dict(
        p99=round(float(max(np.percentile(to_low, 99), np.percentile(to_src, 99))), 4),
        max=round(float(max(to_low.max(), to_src.max())), 4),
        source_to_low_p99=round(float(np.percentile(to_low, 99)), 4),
        low_to_source_p99=round(float(np.percentile(to_src, 99)), 4))
