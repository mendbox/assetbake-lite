# SPDX-License-Identifier: GPL-3.0-or-later
"""The whole job, in order: gather, preclean, (split into parts), then per asset: low mesh, deviation, UV, bake,
material, back to the scene's units; finally the result file.

`run(job, emit)` is called by worker.py. `emit(event, **fields)` sends one message to the add-on.
"""
import json
import time

import bpy
import numpy as np

from . import (ASSET_PROPERTY, autouv, bake, gather, lowmesh, material, measure, parts, preclean, report,
               tris, util, uvlab)

UV_GAP_PX = 8
ASSET_SUFFIX = "_asset"


class UserError(Exception):
    """A problem the user can act on; the text is shown in the panel as it is."""


class Progress:
    """Maps the stage of the current asset onto one 0..1 bar."""
    HEAD, TAIL = 0.08, 0.03
    STAGE = dict(low=(0.0, 0.35), measure=(0.35, 0.45), uv=(0.45, 0.65), bake=(0.65, 1.0))

    def __init__(self, emit):
        self.emit, self.assets, self.index, self.stage = emit, 1, 0, "low"

    def head(self, share, text):
        self.emit("progress", fraction=round(self.HEAD * share, 4), text=text)

    def at(self, stage, share, text):
        self.stage = stage
        a, b = self.STAGE[stage]
        local = a + (b - a) * share
        span = (1.0 - self.HEAD - self.TAIL) / self.assets
        label = text if self.assets == 1 else "Asset %d of %d: %s" % (self.index + 1, self.assets, text)
        self.emit("progress", fraction=round(self.HEAD + span * (self.index + local), 4), text=label)

    def text(self, text):
        self.at(self.stage, 0.5, text)


def _uv_numbers(me):
    """Islands, triangles without UV area, and the vertex count a game engine sees (split at UV seams)."""
    topo = uvlab.Topo(me)
    uv = uvlab._uv_array(me)
    islands, _ = uvlab.uv_islands(topo, uv)
    tris = uv[topo.ls[:, None] + np.arange(3)]
    d1, d2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    key = np.concatenate([topo.lv[:, None].astype(np.float64), np.round(uv, 6)], 1)
    return dict(islands=int(islands.max()) + 1 if len(islands) else 0,
                zero_area_tris=int((area < 1e-12).sum()),
                engine_verts=int(len(np.unique(key, axis=0))))


def make_asset(src, name, options, progress):
    """One asset from one working mesh (diagonal 100, at the origin). Returns (object, report entry).
    The asset is left in working units; the caller moves it back."""
    t0 = time.perf_counter()
    size = int(options["texture_size"])
    for ob in bpy.context.scene.objects:
        ob.hide_render = ob is not src            # a second object at the same place ruins the AO bake
    source = bake.prepare_source_materials(src)
    source_mats, has, emission_max, constants = source["mats"], source["has"], source["emission_max"], source["constants"]
    wanted = [m for m in bake.MAPS if options["maps"].get(m) and has[m]]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    src_bvh = measure.bvh_of(src, depsgraph)

    progress.at("low", 0.0, "Building the low mesh")
    mode = options.get("mode", "GAME")
    builder = lowmesh.LOW_MESH_BUILDERS[mode]
    # surface behind glass must stay: with alpha in the source nothing counts as hidden
    drop_hidden = bool(options.get("remove_hidden", True)) and not has["alpha"]
    extra = dict(binding=options.get("binding")) if mode == "QUADS" else {}
    asset, low, low_stats = builder(src, int(options["budget"]), name, src_bvh=src_bvh, drop_hidden=drop_hidden,
                                    progress=progress.text, **extra)
    # Quad-dominant: `asset` keeps its quads and gets the UVs; the shape is measured and the maps are baked on
    # its fixed triangulation (tris.py), which is also what the GLB export writes. Game: asset and low are one.
    quads_kept = low is None

    progress.at("measure", 0.0, "Measuring the shape")
    if quads_kept:
        low = tris.triangulated_copy(asset, name + "_measure")
    dev = measure.deviation(src, src_bvh, low, ghost=low_stats.pop("ghost"))
    del src_bvh
    if quads_kept:
        util.remove_object(low)

    progress.at("uv", 0.0, "Unwrapping UVs")
    unwrapped = asset if quads_kept else low
    if not low_stats["under_budget"]:
        # a decimated mesh is shaded smooth and the normal map brings the hard edges back; geometry that was kept
        # as it is keeps the source's own flat faces and sharp edges
        util.clear_custom_normals(unwrapped)
        util.shade_smooth(unwrapped.data)
        if "sharp_edge" in unwrapped.data.attributes:
            unwrapped.data.attributes.remove(unwrapped.data.attributes["sharp_edge"])
    uv_stats = autouv.auto_uv(unwrapped, tex=size, gap_px=UV_GAP_PX)
    if quads_kept:
        low = tris.triangulated_copy(asset, name + "_bake")
        asset.hide_render = True                  # a second object at the same place ruins the AO bake
    uv_stats.update(_uv_numbers(low.data))

    progress.at("bake", 0.0, "Baking")
    low.hide_render = False
    cage, retry_cage = bake.cage_distances(dev)
    baker = bake.Baker(src, low, size, source_mats, emission_max, progress=progress.text)
    baked, texels = baker.run(wanted, cage, retry_cage)
    baker.finish()
    if quads_kept:
        low_stats["tris"] = util.tri_count(low.data)
        util.remove_object(low)
    wanted = [m for m in wanted if m in baked]            # an alpha map that changes nothing is dropped by the bake
    progress.at("bake", 0.95, "Packing the textures")
    images = bake.build_textures(baked, name, options["work_dir"], constants)
    del baked
    mat = material.build(name, images, constants, emission_strength=emission_max,
                         has_alpha="alpha" in wanted, has_occlusion="ao" in wanted)
    asset.data.materials.clear()
    asset.data.materials.append(mat)

    seconds = time.perf_counter() - t0
    lines, messages = report.asset_report(name, low_stats, dev, uv_stats, texels, wanted, size, seconds, mode=mode)
    entry = dict(name=name, lines=lines, messages=messages, seconds=round(seconds, 1), low=low_stats, deviation=dev,
                 uv={k: v for k, v in uv_stats.items() if k != "tm"}, texels=texels, maps=["normal"] + wanted,
                 texture_size=size, emission_strength=emission_max, replaced_materials=source["replaced"],
                 missing_images=source["missing"], vertex_colours=source["vertex_colours"], unlit=source["unlit"],
                 alpha_kept_hidden=bool(options.get("remove_hidden", True)) and has["alpha"])
    return asset, entry


def asset_name(base, index=None):
    """"<base>_asset", "<base>_asset_02" for one asset per part; a base that already ends in "asset" (the
    fallback name "Asset") is not doubled."""
    name = base if base.lower().endswith("asset") else base + ASSET_SUFFIX
    return name if index is None else "%s_%02d" % (name, index + 1)


def _to_scene(ob, scale, centre, origin):
    """Working units -> scene units, with the object's origin at `origin` (world)."""
    co = util.vert_coords(ob.data) / scale + centre
    util.set_vert_coords(ob.data, co - origin)
    ob.location = origin
    return co.min(0), co.max(0)


def lite_clamp(options):
    """Lite limits, applied in the worker whatever the job file says."""
    out = dict(options)
    out["mode"] = "GAME"
    out["budget"] = max(200, min(int(out.get("budget", 5000)), 5000))
    out["texture_size"] = 1024
    out["scope"] = "SELECTION"
    wanted = dict(options.get("maps") or {})
    out["maps"] = {m: (m == "basecolor" or (m == "ao" and bool(wanted.get("ao", True)))) for m in bake.MAPS}
    return out


def run(job, emit):
    t0 = time.perf_counter()
    options = dict(lite_clamp(job["options"]), work_dir=job["work_dir"])
    progress = Progress(emit)
    progress.head(0.0, "Reading the selection")
    objects, notes = gather.load_sources(job)
    if not objects:
        raise UserError("Nothing to work on: the selection has no mesh objects.")
    base = job["base_name"]
    src = gather.join(objects, base)
    total_tris = util.tri_count(src.data)
    if total_tris == 0:
        raise UserError("The selected meshes have no faces.")
    if total_tris > report.REFUSE_INPUT_TRIS:
        raise UserError("The selection has %s triangles. The limit is %s: decimate the source first." % (
            report.number(total_tris), report.number(report.REFUSE_INPUT_TRIS)))
    scale, centre = gather.normalise(src)
    progress.head(0.4, "Cleaning the source (%s triangles)" % report.number(total_tris))
    pre, _ = preclean.preclean(src)

    part_notes = {}
    sources = [src]
    if options.get("scope") == "PARTS":
        progress.head(0.8, "Splitting into parts")
        sources, part_notes = parts.split(src, pre["islands"], base)
    progress.assets = len(sources)
    several = len(sources) > 1

    results, entries = [], []
    for index, part in enumerate(sources):
        progress.index = index
        part_scale, part_centre = scale, centre
        if several:
            # tolerances are relative to the asset: every part becomes its own diagonal-100 mesh
            s2, c2 = gather.normalise(part)
            part_scale, part_centre = scale * s2, centre + c2 / scale
        name = asset_name(base, index if several else None)
        asset, entry = make_asset(part, name, options, progress)
        util.remove_object(part)
        origin = np.array(job["origin"], np.float64)
        if several:
            lo, hi = _to_scene(asset, part_scale, part_centre, np.zeros(3))
            origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            util.set_vert_coords(asset.data, util.vert_coords(asset.data) - origin)
            asset.location = origin
        else:
            _to_scene(asset, part_scale, part_centre, origin)
        asset.name = asset.data.name = name
        mark = dict(triangles=entry["low"]["tris"], texture_size=entry["texture_size"], maps=entry["maps"],
                    emission_strength=entry["emission_strength"])
        quad = entry["low"].get("quad")
        if quad and quad.get("used"):
            # "fixed_split": the GLB export splits every quad from corner 0 to corner 2, as the bake did
            mark.update(mode="QUADS", fixed_split=True, faces=quad["faces"], quads=quad["quads"],
                        topology=tris.topology_hash(asset.data))
        asset[ASSET_PROPERTY] = json.dumps(mark)
        results.append(asset)
        entries.append(entry)

    emit("progress", fraction=0.98, text="Saving the result")
    for ob in results:
        ob.hide_render = False                     # every bake hid all other objects from the renderer
    bpy.data.libraries.write(job["result_blend"], set(results), fake_user=False)
    def union(key):
        return sorted({x for e in entries for x in e[key]})
    messages = report.job_messages(notes, part_notes, total_tris, union("replaced_materials"), union("missing_images"),
                                   any(e["alpha_kept_hidden"] for e in entries),
                                   alpha_baked=any("alpha" in e["maps"] for e in entries),
                                   vertex_colours=any(e["vertex_colours"] for e in entries), unlit=union("unlit"))
    return dict(objects=[ob.name for ob in results], assets=entries, messages=messages,
                source=dict(objects=notes["objects"], triangles=total_tris, preclean=pre, notes=notes,
                            parts=part_notes),
                seconds=round(time.perf_counter() - t0, 1))
