# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy tools for the automatic UV unwrap (used by autouv.py). No scipy, works in Blender 4.2 to 5.x.

Parts:
  Topo(me)            numpy view of a mesh: face normals and areas, manifold face pairs, edge lengths
  cluster_hier(...)   chart segmentation: bottom-up face clustering on smoothed normals
  smooth_boundaries, merge_small, labels_to_seams
  metrics(me)         UV islands, distortion per triangle, optional raster fill and overlaps
  island_report(...)  per-island quality after an unwrap

Notes on cluster_hier (the heart of the unwrap):
  every cluster keeps the sum of area * smoothed normal, the sum of area * raw normal, its area and its perimeter.
  R = |sum| / area is the mean resultant length of the normals: 1 = flat, 0.64 = half cylinder, 0.5 = hemisphere,
  0 = closed tube. Two clusters may merge while R of the RAW normals stays above `rmin` (a chart must not wrap around
  a tube) and a conservative bound on the widest angle between a member's SMOOTHED normal and the chart mean stays
  under `cone` degrees. The merge cost is (1 - R_smoothed) + w_shape * relative loss of compactness.
"""
import heapq
import math

import numpy as np


class Topo:
    """Numpy view of a mesh (object mode)."""

    def __init__(self, me):
        self.me = me
        nv, ne, nl, nf = (len(me.vertices), len(me.edges), len(me.loops), len(me.polygons))
        self.nv, self.ne, self.nl, self.nf = (nv, ne, nl, nf)
        co = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get('co', co)
        self.co = co.reshape(-1, 3)
        self.lv = np.empty(nl, np.int32)
        me.loops.foreach_get('vertex_index', self.lv)
        self.le = np.empty(nl, np.int32)
        me.loops.foreach_get('edge_index', self.le)
        self.ls = np.empty(nf, np.int32)
        me.polygons.foreach_get('loop_start', self.ls)
        self.lt = np.empty(nf, np.int32)
        me.polygons.foreach_get('loop_total', self.lt)
        self.lp = np.repeat(np.arange(nf, dtype=np.int32), self.lt)
        n = np.empty(nf * 3, np.float64)
        me.polygons.foreach_get('normal', n)
        self.fn = n.reshape(-1, 3)
        self.fa = np.empty(nf, np.float64)
        me.polygons.foreach_get('area', self.fa)
        c = np.empty(nf * 3, np.float64)
        me.polygons.foreach_get('center', c)
        self.fc = c.reshape(-1, 3)
        ev = np.empty(ne * 2, np.int32)
        me.edges.foreach_get('vertices', ev)
        self.ev = ev.reshape(-1, 2)
        self.elen = np.linalg.norm(self.co[self.ev[:, 0]] - self.co[self.ev[:, 1]], axis=1)
        order = np.argsort(self.le, kind='stable')
        cnt = np.bincount(self.le, minlength=ne)
        start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        self.ecnt = cnt
        m = np.nonzero(cnt == 2)[0]
        self.pe = m
        self.pl0 = order[start[m]]
        self.pl1 = order[start[m] + 1]
        self.pf0 = self.lp[self.pl0]
        self.pf1 = self.lp[self.pl1]
        self.plen = self.elen[m]
        d = np.einsum('ij,ij->i', self.fc[self.pf1] - self.fc[self.pf0], self.fn[self.pf0])
        self.pconcave = d > 0
        self.pcos = np.einsum('ij,ij->i', self.fn[self.pf0], self.fn[self.pf1])
        self.fper = np.bincount(self.lp, weights=self.elen[self.le], minlength=nf)

    def smoothed_normals(self, iters, sigma_deg=40.0):
        """Area-weighted neighbour averaging of face normals. Bilateral: the weight of a neighbour falls with the
        dihedral angle of the shared edge (exp(-(angle/sigma)^2)), so thin rods and hard edges are not blurred away.
        sigma_deg=0 gives plain averaging."""
        n = self.fn * self.fa[:, None]
        if sigma_deg > 0:
            ang = np.arccos(np.clip(self.pcos, -1, 1))
            w = np.exp(-(ang / math.radians(sigma_deg)) ** 2)[:, None]
        else:
            w = 1.0
        for _ in range(iters):
            acc = n.copy()
            np.add.at(acc, self.pf0, n[self.pf1] * w)
            np.add.at(acc, self.pf1, n[self.pf0] * w)
            l = np.linalg.norm(acc, axis=1)
            l[l == 0] = 1
            n = acc / l[:, None] * self.fa[:, None]
        l = np.linalg.norm(n, axis=1)
        l[l == 0] = 1
        return n / l[:, None]

    def components(self, labels=None):
        """Connected components of faces; with labels, only same-label neighbours connect."""
        parent = list(range(self.nf))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        f0, f1 = (self.pf0, self.pf1)
        if labels is not None:
            k = labels[f0] == labels[f1]
            f0, f1 = (f0[k], f1[k])
        for a, b in zip(f0.tolist(), f1.tolist()):
            ra, rb = (find(a), find(b))
            if ra != rb:
                parent[rb] = ra
        root = np.array([find(i) for i in range(self.nf)])
        _, comp = np.unique(root, return_inverse=True)
        return comp.astype(np.int32)


def cluster_hier(T, sn, rmin=0.55, w_shape=0.2, max_share=1.0, w_concave=0.0, labels=None, target=None, restrict=None, cone=120.0):
    """Bottom-up clustering (Garland-style hierarchical face clustering, simplified).
    Each cluster keeps: sum of area*smoothed normal, sum of area*raw normal, area, perimeter.
    R = |sum|/area is the mean resultant length of normals (1 = flat, 0.64 = half cylinder, 0.5 = hemisphere, 0 = closed tube).
    A merge is allowed while
      R of the RAW normals of the merged chart >= rmin  (stops charts that wrap around a tube or close on themselves), and
      a conservative bound on the largest angle between a member's SMOOTHED normal and the chart mean <= cone degrees
      (bound: dev(m) <= max(dev(a) + angle(Na, Nm), dev(b) + angle(Nb, Nm)); real cones are narrower than the bound).
    Cost = (1 - R_smoothed) + w_shape * relative loss of compactness (+ w_concave * share of the common border that is not a concave crease).
    labels: start from an existing partition. target: merge down to this many clusters, ignoring the limits.
    restrict: boolean face mask, only these faces take part (others get label -1)."""
    nf = T.nf
    lab0 = np.arange(nf, dtype=np.int64) if labels is None else labels.astype(np.int64)
    ncl = int(lab0.max()) + 1
    N = np.zeros((ncl, 3))
    np.add.at(N, lab0, sn * T.fa[:, None])
    NR = np.zeros((ncl, 3))
    np.add.at(NR, lab0, T.fn * T.fa[:, None])
    A = np.bincount(lab0, weights=T.fa, minlength=ncl)
    f0, f1, pl, conc, pc = (T.pf0, T.pf1, T.plen, T.pconcave, T.pcos)
    if restrict is not None:
        k = restrict[f0] & restrict[f1]
        f0, f1, pl, conc, pc = (f0[k], f1[k], pl[k], conc[k], pc[k])
    a, b = (lab0[f0], lab0[f1])
    k = a != b
    a, b, plk = (a[k], b[k], pl[k])
    crease_len = plk * (conc[k] & (pc[k] < 0.82))
    P = np.bincount(lab0, weights=T.fper, minlength=ncl)
    same = lab0[T.pf0] == lab0[T.pf1]
    np.subtract.at(P, lab0[T.pf0][same], 2 * T.plen[same])
    adj = [None] * ncl
    for x, y, s, w in zip(a.tolist(), b.tolist(), plk.tolist(), crease_len.tolist()):
        dx = adj[x]
        if dx is None:
            dx = adj[x] = {}
        dy = adj[y]
        if dy is None:
            dy = adj[y] = {}
        e = dx.get(y)
        if e is None:
            e = [0.0, 0.0]
            dx[y] = e
            dy[x] = e
        e[0] += s
        e[1] += w
    Nl = N.tolist()
    NRl = NR.tolist()
    Al = (A + 1e-30).tolist()
    Pl = P.tolist()
    nn = np.linalg.norm(N, axis=1)
    nn[nn == 0] = 1
    cosd = np.einsum('ij,ij->i', sn, (N / nn[:, None])[lab0])
    dmin = np.ones(ncl)
    np.minimum.at(dmin, lab0, cosd)
    Dl = np.arccos(np.clip(dmin, -1, 1)).tolist()
    conelim = math.radians(cone)
    acos = math.acos
    sqrt = math.sqrt
    Atot = float(T.fa.sum()) if restrict is None else float(T.fa[restrict].sum())
    amax = max_share * Atot
    ver = [0] * ncl
    alive = [adj[i] is not None for i in range(ncl)]
    four_pi = 4 * math.pi
    free = target is not None

    def cost(x, y, e):
        """-> (cost, allowed, cone bound of the merged cluster)"""
        nx, ny = (Nl[x], Nl[y])
        mx, my, mz = (nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2])
        am = Al[x] + Al[y]
        lm = sqrt(mx * mx + my * my + mz * mz)
        lx = sqrt(nx[0] * nx[0] + nx[1] * nx[1] + nx[2] * nx[2]) * lm + 1e-30
        ly = sqrt(ny[0] * ny[0] + ny[1] * ny[1] + ny[2] * ny[2]) * lm + 1e-30
        dx_ = Dl[x] + acos(max(-1.0, min(1.0, (nx[0] * mx + nx[1] * my + nx[2] * mz) / lx)))
        dy_ = Dl[y] + acos(max(-1.0, min(1.0, (ny[0] * mx + ny[1] * my + ny[2] * mz) / ly)))
        dev = dx_ if dx_ > dy_ else dy_
        ok = True
        if not free:
            rx, ry = (NRl[x], NRl[y])
            qx, qy, qz = (rx[0] + ry[0], rx[1] + ry[1], rx[2] + ry[2])
            ok = dev <= conelim and am <= amax and (sqrt(qx * qx + qy * qy + qz * qz) / am >= rmin)
        pm = Pl[x] + Pl[y] - 2 * e[0]
        gm = pm * pm / (four_pi * am) + 1e-30
        gx = Pl[x] * Pl[x] / (four_pi * Al[x])
        gy = Pl[y] * Pl[y] / (four_pi * Al[y])
        shape = max((gm - max(gx, gy)) / gm, -1.0)
        c = 1 - lm / am + w_shape * shape
        if w_concave:
            c += w_concave * (1 - e[1] / e[0]) if e[0] > 0 else 0.0
        return (c, ok, dev)
    heap = []
    for x in range(ncl):
        dx = adj[x]
        if dx is None:
            continue
        for y, e in dx.items():
            if y > x:
                c, ok, dv = cost(x, y, e)
                if ok:
                    heap.append((c, x, y, 0, 0))
    heapq.heapify(heap)
    nclusters = sum(alive)
    parent = list(range(ncl))
    while heap:
        if free and nclusters <= target:
            break
        c, x, y, vx, vy = heapq.heappop(heap)
        if not (alive[x] and alive[y]) or ver[x] != vx or ver[y] != vy:
            continue
        if len(adj[x]) < len(adj[y]):
            x, y = (y, x)
        e = adj[x].pop(y)
        del adj[y][x]
        Dl[x] = cost(x, y, e)[2]
        nx, ny = (Nl[x], Nl[y])
        Nl[x] = [nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2]]
        nx, ny = (NRl[x], NRl[y])
        NRl[x] = [nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2]]
        Pl[x] = Pl[x] + Pl[y] - 2 * e[0]
        Al[x] += Al[y]
        dx = adj[x]
        for z, ez in adj[y].items():
            dz = adj[z]
            del dz[y]
            ex = dx.get(z)
            if ex is None:
                dx[z] = ez
                dz[x] = ez
            else:
                ex[0] += ez[0]
                ex[1] += ez[1]
        adj[y] = None
        alive[y] = False
        parent[y] = x
        nclusters -= 1
        ver[x] += 1
        vx = ver[x]
        for z, ez in dx.items():
            c, ok, dv = cost(x, z, ez)
            if ok:
                heapq.heappush(heap, (c, x, z, vx, ver[z]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    roots = np.array([find(i) for i in range(ncl)], dtype=np.int64)
    out = roots[lab0]
    if restrict is not None:
        out = np.where(restrict, out, -1)
    return out


def relabel(labels):
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


def smooth_boundaries(T, labels, iters=3):
    """A face whose border is mostly shared with one other chart moves to that chart (shortens seams)."""
    labels = labels.copy()
    f0, f1, pl = (T.pf0, T.pf1, T.plen)
    nb = [[] for _ in range(T.nf)]
    for a, b, s in zip(f0.tolist(), f1.tolist(), pl.tolist()):
        nb[a].append((b, s))
        nb[b].append((a, s))
    moved_total = 0
    for _ in range(iters):
        diff = labels[f0] != labels[f1]
        cand = np.unique(np.concatenate((f0[diff], f1[diff])))
        L = labels.tolist()
        moved = 0
        for f in cand.tolist():
            own = L[f]
            acc = {}
            for g, s in nb[f]:
                acc[L[g]] = acc.get(L[g], 0.0) + s
            best = max(acc, key=acc.get)
            if best != own and acc[best] > 1.2 * acc.get(own, 0.0) and (acc[best] > 0.5 * sum(acc.values())):
                labels[f] = best
                L[f] = best
                moved += 1
        moved_total += moved
        if not moved:
            break
    return (labels, moved_total)


def merge_small(T, sn, labels, min_share=0.004, min_faces=8, rfloor=0.5):
    """Connected charts smaller than min_share of the area (or min_faces) join the neighbour with the
    longest common border, as long as the merged chart keeps R (raw normals) >= rfloor, so two halves of a
    thin rod are not merged back into a closed tube. Tiny separate shells stay."""
    labels = T.components(labels).astype(np.int64)
    Atot = T.fa.sum()
    for _ in range(4):
        ncl = int(labels.max()) + 1
        A = np.bincount(labels, weights=T.fa, minlength=ncl)
        cnt = np.bincount(labels, minlength=ncl)
        N = np.zeros((ncl, 3))
        np.add.at(N, labels, T.fn * T.fa[:, None])
        small = (A < min_share * Atot) | (cnt < min_faces)
        if not small.any():
            break
        a, b = (labels[T.pf0], labels[T.pf1])
        k = a != b
        a, b, s = (a[k], b[k], T.plen[k])
        border = {}
        for x, y, l in zip(a.tolist(), b.tolist(), s.tolist()):
            border.setdefault(x, {})
            border.setdefault(y, {})
            border[x][y] = border[x].get(y, 0.0) + l
            border[y][x] = border[y].get(x, 0.0) + l
        target = np.arange(ncl)
        changed = 0
        order = np.argsort(A, kind="stable")
        done = set()
        for x in order.tolist():
            if not small[x] or x not in border or x in done:
                continue
            best = None
            bl = 0.0
            for y, l in border[x].items():
                if y in done:
                    continue
                m = N[x] + N[y]
                r = np.sqrt(m @ m) / (A[x] + A[y])
                if r >= rfloor and l > bl:
                    best, bl = (y, l)
            if best is not None:
                target[x] = best
                done.add(x)
                done.add(best)
                changed += 1
        if not changed:
            break
        labels = relabel(target[labels])
    return labels


def labels_to_seams(T, labels):
    seam = np.zeros(T.ne, dtype=bool)
    seam[T.pe[labels[T.pf0] != labels[T.pf1]]] = True
    T.me.edges.foreach_set('use_seam', seam)
    return int(seam.sum())


def _uv_array(me):
    uvl = me.uv_layers.active
    uv = np.empty(len(me.loops) * 2, np.float64)
    uvl.data.foreach_get('uv', uv)
    return uv.reshape(-1, 2)


def uv_islands(T, uv):
    """Island id per face from UV connectivity, and the boolean 'cut' flag per manifold pair."""
    q = np.round(uv * 100000.0).astype(np.int64)
    key = np.stack((T.lv.astype(np.int64), q[:, 0], q[:, 1]), axis=1)
    _, uvid = np.unique(key, axis=0, return_inverse=True)
    uvid = uvid.reshape(-1)
    nxt = np.arange(T.nl) + 1
    last = T.ls + T.lt - 1
    nxt[last] = T.ls
    a = uvid
    b = uvid[nxt]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    joined = (lo[T.pl0] == lo[T.pl1]) & (hi[T.pl0] == hi[T.pl1])
    parent = list(range(T.nf))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(T.pf0[joined].tolist(), T.pf1[joined].tolist()):
        rx, ry = (find(x), find(y))
        if rx != ry:
            parent[ry] = rx
    root = np.array([find(i) for i in range(T.nf)])
    _, isl = np.unique(root, return_inverse=True)
    return (isl.reshape(-1), ~joined)


def boundary_crossings(T, uv, isl, cut):
    """Number of proper self-intersections of each island's UV boundary. With no flipped faces, an island overlaps
    itself only if its boundary crosses itself, so this is a cheap overlap test (no rasterisation)."""
    bl = np.zeros(T.nl, bool)
    bl[T.pl0[cut]] = True
    bl[T.pl1[cut]] = True
    bl[T.ecnt[T.le] != 2] = True
    l = np.nonzero(bl)[0]
    nxt = l + 1
    last = (T.ls + T.lt - 1)[T.lp[l]]
    nxt = np.where(l == last, T.ls[T.lp[l]], nxt)
    A = uv[l]
    B = uv[nxt]
    ii = isl[T.lp[l]]
    order = np.argsort(ii, kind='stable')
    A, B, ii = (A[order], B[order], ii[order])
    ni = int(isl.max()) + 1
    out = np.zeros(ni, np.int64)
    bounds = np.searchsorted(ii, np.arange(ni + 1))
    for i in range(ni):
        s0, s1 = (bounds[i], bounds[i + 1])
        n = s1 - s0
        if n < 4:
            continue
        a = A[s0:s1]
        b = B[s0:s1]
        d = b - a

        def cross(o, dv, pts):
            return dv[:, None, 0] * (pts[None, :, 1] - o[:, None, 1]) - dv[:, None, 1] * (pts[None, :, 0] - o[:, None, 0])
        c1 = cross(a, d, a)
        c2 = cross(a, d, b)
        eps = 1e-12
        straddle = (c1 > eps) & (c2 < -eps) | (c1 < -eps) & (c2 > eps)
        hit = straddle & straddle.T
        out[i] = int(np.triu(hit, 1).sum())
    return out


def tri_data(T, uv):
    me = T.me
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get('loops', tl)
    tl = tl.reshape(-1, 3)
    tp = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get('polygon_index', tp)
    P = T.co[T.lv[tl]]
    Q = uv[tl]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    x1 = np.linalg.norm(e1, axis=1)
    x1s = np.where(x1 > 0, x1, 1)
    x2 = np.einsum('ij,ij->i', e2, e1) / x1s
    y2 = np.linalg.norm(np.cross(e1, e2), axis=1) / x1s
    ok = (x1 > 1e-12) & (y2 > 1e-12)
    y2s = np.where(ok, y2, 1)
    d1 = Q[:, 1] - Q[:, 0]
    d2 = Q[:, 2] - Q[:, 0]
    ja = d1 / x1s[:, None]
    jb = (d2 - d1 * (x2 / x1s)[:, None]) / y2s[:, None]
    E = (ja * ja).sum(1)
    F = (ja * jb).sum(1)
    G = (jb * jb).sum(1)
    disc = np.sqrt(np.maximum((E - G) ** 2 + 4 * F * F, 0))
    s1 = np.sqrt(np.maximum((E + G + disc) / 2, 0))
    s2 = np.sqrt(np.maximum((E + G - disc) / 2, 0))
    det = ja[:, 0] * jb[:, 1] - ja[:, 1] * jb[:, 0]
    area3 = x1 * y2 / 2
    uva = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 2
    return dict(tl=tl, tp=tp, Q=Q, s1=s1, s2=s2, det=det, area3=area3, uva=uva, ok=ok)


def rasterize(Q, isl_t, n=1024):
    """Pixel-centre rasterisation. Returns count map, island-id map, inter-island conflict map."""
    cnt = np.zeros((n, n), np.uint8)
    ids = np.full((n, n), -1, np.int32)
    conflict = np.zeros((n, n), bool)
    q = Q * n
    mn = np.floor(q.min(1) - 0.5).astype(np.int64) + 1
    mx = np.floor(q.max(1) - 0.5).astype(np.int64)
    mn = np.clip(mn, 0, n - 1)
    mx = np.clip(mx, -1, n - 1)
    x0 = q[:, 0, 0]
    y0 = q[:, 0, 1]
    x1 = q[:, 1, 0]
    y1 = q[:, 1, 1]
    x2 = q[:, 2, 0]
    y2 = q[:, 2, 1]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    ar = np.arange(n) + 0.5
    for i in np.nonzero((mx[:, 0] >= mn[:, 0]) & (mx[:, 1] >= mn[:, 1]) & (np.abs(den) > 1e-12))[0].tolist():
        ax, bx = (mn[i, 0], mx[i, 0] + 1)
        ay, by = (mn[i, 1], mx[i, 1] + 1)
        px = ar[ax:bx][None, :]
        py = ar[ay:by][:, None]
        d = den[i]
        w0 = ((y1[i] - y2[i]) * (px - x2[i]) + (x2[i] - x1[i]) * (py - y2[i])) / d
        w1 = ((y2[i] - y0[i]) * (px - x2[i]) + (x0[i] - x2[i]) * (py - y2[i])) / d
        m = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not m.any():
            continue
        sub_ids = ids[ay:by, ax:bx]
        sub_cnt = cnt[ay:by, ax:bx]
        conflict[ay:by, ax:bx] |= m & (sub_ids >= 0) & (sub_ids != isl_t[i])
        sub_cnt[m] += 1
        sub_ids[m] = isl_t[i]
    return (cnt, ids, conflict)


def island_overlap_share(Q, n=192):
    """Self-overlap of ONE island: share of its covered pixels that are covered twice (island scaled to its bbox)."""
    lo = Q.reshape(-1, 2).min(0)
    hi = Q.reshape(-1, 2).max(0)
    size = max(float((hi - lo).max()), 1e-30)
    q = (Q - lo) / size * 0.98 + 0.01
    cnt, _, _ = rasterize(q, np.zeros(len(Q), np.int32), n)
    cov = int((cnt > 0).sum())
    return float((cnt > 1).sum()) / max(cov, 1)


def wpercentile(x, w, p):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float('nan')
    return float(x[o][min(np.searchsorted(cw, p / 100 * cw[-1]), len(x) - 1)])


def island_gap(ids, kmax=8):
    """Smallest pixel distance (Chebyshev) at which two different islands meet, up to kmax."""
    n = ids.shape[0]
    for k in range(1, kmax + 1):
        for dx, dy in ((k, 0), (0, k), (k, k), (k, -k)):
            a = ids[max(dy, 0):n + min(dy, 0), dx:]
            b = ids[max(-dy, 0):n + min(-dy, 0), :n - dx]
            if ((a >= 0) & (b >= 0) & (a != b)).any():
                return k
    return kmax + 1


def metrics(me, raster=1024, T=None):
    T = T or Topo(me)
    uv = _uv_array(me)
    isl, cut = uv_islands(T, uv)
    ni = int(isl.max()) + 1
    td = tri_data(T, uv)
    ok = td['ok'] & (td['s2'] > 1e-12)
    w = td['area3']
    ti = isl[td['tp']]
    ia3 = np.bincount(ti, weights=td['area3'], minlength=ni)
    iuv = np.bincount(ti, weights=td['uva'], minlength=ni)
    iscale = iuv / np.maximum(ia3, 1e-30)
    gscale = iuv.sum() / ia3.sum()
    s = td['s1'] * td['s2']
    with np.errstate(divide='ignore', invalid='ignore'):
        la = np.abs(np.log2(np.where(ok, s / iscale[ti], 1)))
        lg = np.abs(np.log2(np.where(ok, s / gscale, 1)))
        conf = np.where(ok, td['s1'] / np.where(ok, td['s2'], 1), 1)
        dens = np.abs(np.log2(np.maximum(iscale, 1e-30) / gscale))
    W = w.sum()
    sign = np.sign(td['det'])
    isum = np.bincount(ti, weights=sign * w, minlength=ni)
    flipped = ok & (sign * np.sign(isum[ti]) < 0)
    fcount = np.bincount(isl, minlength=ni)
    out = dict(faces=T.nf, islands=ni, islands_tiny=int((fcount < 8).sum()), seam_share=round(float(cut.sum() / T.ne), 4), fill_sum=round(float(td['uva'].sum()), 4), angle_mean=round(float((conf * w).sum() / W), 4), angle_p95=round(wpercentile(conf, w, 95), 3), area_mean=round(float((la * w).sum() / W), 4), area_p95=round(wpercentile(la, w, 95), 3), texel_mean=round(float((lg * w).sum() / W), 4), texel_p95=round(wpercentile(lg, w, 95), 3), density_spread=round(float((dens * ia3).sum() / ia3.sum()), 4), flipped_share=round(float((w * flipped).sum() / W), 5), uv_min=round(float(uv.min()), 4), uv_max=round(float(uv.max()), 4))
    extra = dict(isl=isl, ti=ti, td=td, conf=conf, la=la, T=T, cut=cut)
    if raster:
        cnt, ids, conflict = rasterize(td['Q'], ti, raster)
        cov = cnt > 0
        nc = max(int(cov.sum()), 1)
        out['fill'] = round(float(cov.mean()), 4)
        out['overlap_inter'] = round(float(conflict.sum() / nc), 5)
        out['overlap_self'] = round(float(((cnt > 1) & ~conflict).sum() / nc), 5)
        out['gap_px'] = island_gap(ids)
        extra.update(cnt=cnt, ids=ids, conflict=conflict)
    return (out, extra)


def island_report(extra):
    """Per-island quality after an unwrap: area-weighted mean angle distortion, area p95, flipped share."""
    td = extra['td']
    ti = extra['ti']
    ni = int(ti.max()) + 1
    w = td['area3']
    a3 = np.bincount(ti, weights=w, minlength=ni)
    a3s = np.maximum(a3, 1e-30)
    ang = np.bincount(ti, weights=extra['conf'] * w, minlength=ni) / a3s
    are = np.bincount(ti, weights=extra['la'] * w, minlength=ni) / a3s
    sign = np.sign(td['det'])
    isum = np.bincount(ti, weights=sign * w, minlength=ni)
    fl = (sign * np.sign(isum[ti]) < 0) | (td['det'] == 0)
    flip = np.bincount(ti, weights=fl * w, minlength=ni) / a3s
    uva = np.bincount(ti, weights=td['uva'], minlength=ni)
    return dict(area3=a3, angle=ang, area=are, flip=flip, uvarea=uva)
