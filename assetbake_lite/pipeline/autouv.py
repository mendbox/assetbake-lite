# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic UV unwrap of the low mesh.

    stats = autouv.auto_uv(obj, tex=2048, gap_px=8)

Steps:
  1. face normals smoothed with a bilateral filter (6 passes, sigma 40 degrees)
  2. charts: bottom-up face clustering (uvlab.cluster_hier), border smoothing, small charts merged
  3. seams on chart borders, bpy.ops.uv.unwrap (MINIMUM_STRETCH in Blender 4.3+, ANGLE_BASED in 4.2)
  4. every island is checked (angle and area distortion, flipped faces, self-overlap); bad ones are split and
     unwrapped again, at most `rounds` times
  5. average island scale, pack (concave shapes, any rotation, fixed pixel gap)

The winding of the mesh is taken as it is: the caller makes the normals consistent before.
"""
import time

import bpy
import numpy as np

from . import uvlab as L


def unwrap_method():
    """Blender 4.2 has no MINIMUM_STRETCH unwrap; ANGLE_BASED gives the same island count and fill there."""
    methods = [i.identifier for i in bpy.ops.uv.unwrap.get_rna_type().properties["method"].enum_items]
    return "MINIMUM_STRETCH" if "MINIMUM_STRETCH" in methods else "ANGLE_BASED"

DEFAULTS = dict(
    smooth_iters=6, smooth_sigma=40.0,   # bilateral normal smoothing
    rmin=0.55,                           # mean resultant length of RAW normals a chart must keep (no closed tubes)
    cone=160.0,                          # conservative cone bound on smoothed normals, degrees
    w_shape=0.2,                         # compactness weight in the merge cost
    border_passes=3, min_share=0.004,    # border smoothing; charts under 0.4 % of the area are merged away
    max_angle=1.3, max_area=0.35,        # island limits: mean sigma1 / sigma2, mean |log2 area ratio|
    split_min_share=0.003,               # islands under 0.3 % of the area are split only when broken
    overlap_share=0.004,                 # self-overlap above 0.4 % of the island's pixels counts as broken
    rounds=6,
)


def _edit():
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')


def _unwrap(method):
    kw = dict(method=method, fill_holes=False, correct_aspect=False, margin_method='SCALED', margin=0.001)
    if method == 'MINIMUM_STRETCH':
        kw['iterations'] = 10
    bpy.ops.uv.unwrap(**kw)


def unwrap_collapsed(me, T):
    """True when the unwrap gave one island that is a line or a point and as long as the whole layout. The
    MINIMUM_STRETCH solver does that to a chart that is pinched at a vertex (TRELLIS meshes have such charts);
    the unwrap operator then scales every other island down to nothing next to it."""
    uv = L._uv_array(me)
    if not len(uv):
        return False
    isl, _ = L.uv_islands(T, uv)
    n = int(isl.max()) + 1
    corners = T.ls[:, None] + np.arange(3)
    tri = uv[corners]
    d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.bincount(isl, weights=0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]), minlength=n)
    lo, hi = np.full((n, 2), np.inf), np.full((n, 2), -np.inf)
    which = np.repeat(isl, 3)
    np.minimum.at(lo, which, tri.reshape(-1, 2))
    np.maximum.at(hi, which, tri.reshape(-1, 2))
    extent = (hi - lo).max(1)
    whole = float((uv.max(0) - uv.min(0)).max())
    return bool(((area < 1e-7 * extent * extent) & (extent > 0.5 * whole)).any()) and n > 1


def auto_uv(ob, tex=2048, gap_px=8, method=None, pack=True, uv_name='UVMap', **over):
    """Unwrap and pack `ob` (active, object mode). Returns a dict with counts and timings."""
    p = dict(DEFAULTS)
    p.update(over)
    method = method or unwrap_method()
    tm = {}
    t0 = t = time.perf_counter()
    me = ob.data
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    if uv_name not in me.uv_layers:
        me.uv_layers.new(name=uv_name)
    me.uv_layers.active = me.uv_layers[uv_name]
    me.uv_layers[uv_name].active_render = True
    T = L.Topo(me)
    sn = T.smoothed_normals(p['smooth_iters'], p['smooth_sigma'])
    labels = L.cluster_hier(T, sn, rmin=p['rmin'], w_shape=p['w_shape'], cone=p['cone'])
    labels, _ = L.smooth_boundaries(T, labels, p['border_passes'])
    labels = L.merge_small(T, sn, labels, min_share=p['min_share'])
    tm['charts'] = time.perf_counter() - t
    t = time.perf_counter()
    splits = 0
    rounds = 0
    for rnd in range(p['rounds'] + 1):
        L.labels_to_seams(T, labels)
        _edit()
        _unwrap(method)
        bpy.ops.object.mode_set(mode='OBJECT')
        if method == 'MINIMUM_STRETCH' and unwrap_collapsed(me, T):
            method = 'ANGLE_BASED'                 # for the rest of this mesh: it has no such failure
            _edit()
            _unwrap(method)
            bpy.ops.object.mode_set(mode='OBJECT')
        rounds = rnd
        if rnd == p['rounds']:
            break
        m, ex = L.metrics(me, raster=0, T=T)
        rep = L.island_report(ex)
        isl = ex['isl']
        cnt = np.bincount(isl)
        cross = L.boundary_crossings(T, L._uv_array(me), isl, ex['cut'])
        for i in np.nonzero(cross > 0)[0].tolist():
            if L.island_overlap_share(ex['td']['Q'][ex['ti'] == i]) < p['overlap_share']:
                cross[i] = 0
        big = rep['area3'] >= p['split_min_share'] * rep['area3'].sum()
        severe = (rep['angle'] > 2 * p['max_angle'] - 1) | (rep['area'] > 2 * p['max_area']) | (rep['flip'] > 0.1) | (rep['uvarea'] <= 0) | (cross > 0)
        mild = (rep['angle'] > p['max_angle']) | (rep['area'] > p['max_area']) | (rep['flip'] > 0.03)
        if rnd >= p['rounds'] - 2:
            mild[:] = False
        bad = np.nonzero((mild & big | severe) & (cnt >= 24))[0]
        if not len(bad):
            break
        labels = isl.astype(np.int64).copy()
        nxt = int(labels.max()) + 1
        for i in bad.tolist():
            mask = isl == i
            sub = L.cluster_hier(T, sn, w_shape=p['w_shape'], target=4 if severe[i] and cnt[i] >= 96 else 2, restrict=mask)
            for x in np.unique(sub[mask]).tolist()[1:]:
                labels[mask & (sub == x)] = nxt
                nxt += 1
            splits += 1
    tm['unwrap_and_fix'] = time.perf_counter() - t
    t = time.perf_counter()
    if pack:
        _edit()
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(rotate=True, rotate_method='ANY', shape_method='CONCAVE', margin_method='ADD', margin=gap_px / (2.0 * tex))
        bpy.ops.object.mode_set(mode='OBJECT')
    tm['pack'] = time.perf_counter() - t
    tm['total'] = time.perf_counter() - t0
    return dict(method=method, faces=T.nf, rounds=rounds, splits=splits, tm={k: round(v, 2) for k, v in tm.items()})
