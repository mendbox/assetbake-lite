# SPDX-License-Identifier: GPL-3.0-or-later
"""Scope "one asset per part": split the precleaned mesh into its connected parts.

A part is what stays connected AFTER the merge, not an object of the file: AI files are cut into arbitrary pieces
(eight chairs arrive as 18 objects). Small loose pieces (eyes, buttons, debris) do not become assets of their own:
each one joins the nearest large part.
"""
import bpy
import numpy as np

from . import util

MAIN_PART_MIN_SIZE = 0.3     # a part is "large" when its diagonal is at least this share of the largest one
MAX_ASSETS = 24              # more large parts than this is a scene, not a set of assets
MAX_LOOSE_PIECES = 2000      # separating more pieces than this takes minutes


def _bbox(ob):
    co = util.vert_coords(ob.data)
    return co.min(0), co.max(0)


def split(src, n_islands, base_name):
    """Returns (list of part objects, notes). With one island, or when the split is refused, the list is [src]."""
    notes = dict(islands=int(n_islands), small_pieces_merged=0, refused="")
    if n_islands < 2:
        return [src], notes
    if n_islands > MAX_LOOSE_PIECES:
        notes["refused"] = "pieces"
        return [src], notes
    util.activate(src)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    for ob in [ob for ob in bpy.context.scene.objects if ob.type == "MESH" and not len(ob.data.polygons)]:
        util.remove_object(ob)                         # loose vertices and edges become objects without faces
    pieces = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
    boxes = [_bbox(ob) for ob in pieces]
    size = np.array([float(np.linalg.norm(hi - lo)) for lo, hi in boxes])
    main = np.flatnonzero(size >= MAIN_PART_MIN_SIZE * size.max())
    if len(main) > MAX_ASSETS:
        notes["refused"] = "parts"
        main = main[:1]
    groups = {int(i): [pieces[i]] for i in main}
    for i in range(len(pieces)):
        if i in groups:
            continue
        # distance between the piece's box and every large part's box
        lo, hi = boxes[i]
        gap = [float(np.linalg.norm(np.maximum(0.0, np.maximum(boxes[j][0] - hi, lo - boxes[j][1])))) for j in main]
        groups[int(main[int(np.argmin(gap))])].append(pieces[i])
        notes["small_pieces_merged"] += 1
    parts = []
    for members in groups.values():
        util.activate(*members[::-1])                  # the large part is last = active
        if len(members) > 1:
            bpy.ops.object.join()
        parts.append(members[0])
    if notes["refused"]:
        parts[0].name = base_name
        return parts[:1], notes
    # largest first, so the numbering is stable and the main piece is _01
    parts.sort(key=lambda ob: -float(np.linalg.norm(np.subtract(*_bbox(ob)))))
    for n, ob in enumerate(parts):
        ob.name = "%s_%02d" % (base_name, n + 1)
    return parts, notes
