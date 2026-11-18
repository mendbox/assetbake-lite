# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings and operators. The pipeline itself runs in the worker (pipeline/run.py); this file prepares the job,
watches it and brings the result into the scene."""
import json
import math
import os
import re
import uuid

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, PropertyGroup
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector

from . import const, vendor
from .job import Job, log_path
from .pipeline import ASSET_PROPERTY, gltf_export
from .pipeline.material import OCCLUSION_GROUP

MAP_SETTINGS = (
    ("basecolor", "use_basecolor", "Base Colour", "Colour of the surface"),
    ("ao", "use_ao", "AO", "Ambient occlusion, packed into the red channel of the ORM texture"),
)


# ---------------------------------------------------------------------------------------------- settings

class AssetBakeLiteSettings(PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        items=[("GAME", "Game (triangles)", "All triangles, reduced to the budget. Works on every model and keeps "
                                            "the shape best")],
        default="GAME")
    budget_preset: EnumProperty(
        name="Budget", description="Triangles in the finished asset",
        items=[(str(n), "{:,} triangles".format(n), "") for n in const.BUDGETS],
        default=str(const.LITE_MAX_BUDGET))
    texture_size: EnumProperty(
        name="Texture", description="Width and height of the baked textures, pixels",
        items=[(str(n), "%dk" % (n // 1024), "%d x %d pixels" % (n, n)) for n in const.TEXTURE_SIZES],
        default="1024")
    use_basecolor: BoolProperty(name="Base Colour", description=MAP_SETTINGS[0][3], default=True)
    use_ao: BoolProperty(name="AO", description=MAP_SETTINGS[1][3], default=True)
    remove_hidden: BoolProperty(
        name="Remove hidden surface", default=True,
        description="Surface that is shut in between parts and can never be seen is left out, so its triangles "
                    "and texture space go to the visible surface. The mesh is open there")
    export_textures: BoolProperty(
        name="Also save textures", default=False,
        description="Write the textures as PNG files next to the GLB file")
    report: StringProperty()

    def budget(self):
        return min(int(self.budget_preset), const.LITE_MAX_BUDGET)

    def maps(self):
        return {key: key in const.LITE_MAPS and bool(getattr(self, prop)) for key, prop, _, _ in MAP_SETTINGS} | {
            m: False for m in ("roughness", "metallic", "emission", "alpha")}


def settings(context):
    return getattr(context.scene, const.SETTINGS)


# ---------------------------------------------------------------------------------------------- selection

FILE_SUFFIXES = (".fbx", ".obj", ".glb", ".gltf", ".stl", ".ply", ".usd", ".usdz", ".dae")
NAME_MAX = 40
FALLBACK_NAME = "Asset"
# words that importers, download sites and AI generators put into node, mesh and material names; a name made
# only of these, numbers and hashes says nothing about the model
GENERIC_WORDS = frozenset((
    "sketchfab", "model", "rootnode", "root", "gltf", "scenerootnode", "scene", "auxscene", "world", "armature",
    "empty", "object", "mesh", "geometry", "node", "group", "collection", "default", "defaultmaterial", "material",
    "mat", "texture", "pbr", "baked", "base", "tripo", "meshy", "supavoxel", "badge", "cleaner", "materialmerger",
    "gles", "id", "lod", "submesh", "polysurface", "untitled"))
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32}", re.I)
_FILE_EXT = re.compile(r"(?:%s)(?=$|[._\s])" % "|".join(re.escape(x) for x in FILE_SUFFIXES + (".blend", ".max", ".ma", ".mb", ".3ds")))
_SPLIT = re.compile(r"[\s_.\-]+")


def _bone_shapes():
    return {pb.custom_shape for ob in bpy.data.objects if ob.type == "ARMATURE"
            for pb in ob.pose.bones if pb.custom_shape is not None}


def sources(context):
    """The selected mesh objects that make up the source; helper meshes for bone display and assets made by this
    add-on are left out."""
    shapes = None
    out = []
    for ob in context.selected_objects:
        if ob.type != "MESH" or ASSET_PROPERTY in ob:
            continue
        if shapes is None:
            shapes = _bone_shapes()
        if ob not in shapes:
            out.append(ob)
    return out


def only_assets_selected(context):
    picked = [ob for ob in context.selected_objects if ob.type == "MESH"]
    return bool(picked) and all(ASSET_PROPERTY in ob for ob in picked)


def triangle_count(objects):
    return sum(len(ob.data.loops) - 2 * len(ob.data.polygons) for ob in objects)


# ---------------------------------------------------------------------------------------------- Quad-dominant

_unavailable = dict(text=None)


def quad_unavailable(refresh=False):
    """"" when Quad-dominant can run on this system, else the sentence for the panel. The answer is looked up
    once and kept (the panel asks on every redraw, and the lookup reads the environment and the disk); a click on
    Make and registration look it up again."""
    if refresh or _unavailable["text"] is None:
        _unavailable["text"] = vendor.unavailable_text()
    return _unavailable["text"]


def quad_outlook(tris, budget, scope):
    """What the panel can say about Quad-dominant without measuring: (level, text, measure). `measure` is True
    when the quad share can be estimated (quad_precheck); otherwise the text tells what will happen, and no
    estimate is run (nothing is measured for a source that cannot take quads anyway)."""
    limit = const.QUAD_MAX_SOURCE_TRIS
    if scope == "PARTS":
        # every part has its own triangle count against the budget and the limit, and gets the whole budget
        return ("INFO", "One asset per part: every part is checked on its own and gets the full budget, so no "
                        "quad share is estimated.", False)
    if tris > limit:
        return ("WARNING", "Over {0:,} triangles: Quad-dominant is not available, a Game asset will be made. To get "
                           "quads, decimate the source in Blender first (Modifiers > Decimate, below {0:,} "
                           "triangles), apply it and run Quad-dominant.".format(limit), False)
    if tris <= 2 * budget:                          # the budget counts faces, a quad is two triangles
        return ("INFO", "The source is under the budget ({:,} faces are about {:,} triangles): nothing to reduce, "
                        "it is kept as it is.".format(budget, 2 * budget), False)
    return ("INFO", "", True)


def upsell_line(context, objs):
    """Lite only: one line about the full version, specific to the model. The thin-surface estimate is the scene-ray
    one (pure Python, no compiled module), asked at the budget the Quad-dominant mode is tuned for."""
    tail = " plus 2k/4k PBR (roughness, metallic, emission) - " + const.FULL_URL
    generic = "Full AssetBake: Quad-dominant output (quads on the thick parts)," + tail
    # the specific line only when the full version would really run Quad-dominant here; no synchronous scene-ray
    # estimate on a selection it would not run on (huge, or already under the budget)
    if not quad_outlook(triangle_count(objs), const.QUAD_ADVISED_BUDGET, "SELECTION")[2]:
        return generic
    try:
        level, text = quad_expectation(quad_estimate_now(context, objs, const.QUAD_ADVISED_BUDGET), const.QUAD_ADVISED_BUDGET)
    except Exception:                              # noqa: an upsell line must never stop a run
        level, text = "INFO", ""
    if level == "INFO" and text.startswith("Expect about"):
        return "Full AssetBake: Quad-dominant would give " + text[len("Expect "):].split(",")[0] + " here," + tail
    return generic


def quad_precheck(objects, budget, depsgraph, scene=None, samples=const.QUAD_ESTIMATE_SAMPLES, skip_hidden=True):
    """A fast look at the wall thickness of the selection: the share of the surface (0..1) that is thinner than
    the worker's thin limit at this face budget, or None when there is nothing to measure.

    Random faces, picked by area with a fixed seed, each send one ray through the wall to its back side (front
    sides of other parts are passed through, as in the worker). A face with nothing behind it at any distance
    belongs to an open sheet (a card, a leaf, a glass pane), which the worker keeps as triangles as a whole.
    The rays run through the scene, so a wall whose two sides are two objects is seen; objects outside the
    selection are passed through. The worker's closing and growing of thin regions and its rules for small pieces
    are left out: this is an estimate, quad_expectation() turns it into what to tell the user."""
    scene = scene or bpy.context.scene
    picks = []
    for ob in objects:
        # the evaluated mesh: what the rays through the scene meet (modifiers applied, a rigged mesh in its pose)
        shown = ob.evaluated_get(depsgraph)
        me = shown.data
        n = len(me.polygons)
        if not n:
            continue
        area = np.empty(n, np.float64)
        me.polygons.foreach_get("area", area)
        scale = abs(shown.matrix_world.to_3x3().determinant()) ** (1.0 / 3.0)
        if not scale > 0.0 or not area.sum() > 0.0:
            continue
        picks.append((shown, area, scale))
    total = sum(float(area.sum()) * scale * scale for _, area, scale in picks)
    if not total > 0.0:
        return None
    edge = math.sqrt(total / max(1, budget))                  # world units: the quad edge at this budget
    limit, reach, eps = const.QUAD_THIN_EDGES * edge, 4.0 * edge, 1e-4 * edge
    chosen_set = {ob.original for ob in objects}
    rng = np.random.default_rng(5)
    thin = tested = 0
    taken = []                                                 # (origin on the face, direction through the wall)
    for ob, area, scale in picks:
        share = float(area.sum()) * scale * scale / total
        count = max(1, int(round(samples * share)))
        chosen = rng.choice(len(area), size=count, p=area / area.sum())
        polygons = ob.data.polygons
        world = ob.matrix_world
        normal_matrix = world.to_3x3().inverted_safe().transposed()
        for i in chosen.tolist():
            poly = polygons[i]
            direction = -(normal_matrix @ poly.normal)
            if direction.length_squared < 1e-20:
                continue
            direction.normalize()
            origin = world @ poly.center + direction * eps
            gone, wall = 0.0, None
            tested += 1
            taken.append((origin.copy(), direction.copy()))
            for step_no in range(6):
                hit, location, hit_normal, _index, hit_ob, _matrix = scene.ray_cast(
                    depsgraph, origin, direction, distance=(reach - gone) if step_no < 5 and wall is None else 1e30)
                if not hit:
                    break
                step = (location - origin).length
                if hit_ob.original in chosen_set and hit_normal.dot(direction) > 0.0:
                    wall = gone + step                           # a back side: the other face of this wall
                    break
                gone += step + eps
                origin = location + direction * eps
                if gone >= reach:
                    wall = reach                                 # something is behind, further than a thin wall
                    break
            if wall is None:
                # nothing within reach: thick, unless nothing of the selection is behind this face at all
                far_origin, behind = origin, False
                for _ in range(8):
                    hit, location, hit_normal, _index, hit_ob, _matrix = scene.ray_cast(depsgraph, far_origin, direction)
                    if not hit:
                        break
                    if hit_ob.original in chosen_set:
                        behind = True
                        break
                    far_origin = location + direction * eps
                if not behind:
                    thin += 1
            elif wall < limit:
                thin += 1
    if not tested:
        return None
    if thin <= const.QUAD_THIN_FALLBACK * tested or not skip_hidden:
        return thin / tested
    # Mostly thin. A crust (TRELLIS.2: a second skin right under the seen one, more layers inside) reads like that
    # too; the worker then leaves out the layers that nothing outside can see and reads the thickness again. The
    # same here: shut-in samples are not counted, and a back side that is shut in is not the other face of a wall.

    def blocked(origin, direction):
        for _ in range(8):
            hit, location, _normal, _index, hit_ob, _matrix = scene.ray_cast(depsgraph, origin, direction)
            if not hit:
                return False
            if hit_ob.original in chosen_set:
                return True
            origin = location + direction * eps
        return False

    def shut_in(point, normal):
        """Fewer than two of 13 rays into the hemisphere around `normal` leave the selection (the worker's rule)."""
        helper = Vector((1.0, 0.0, 0.0)) if abs(normal.x) < 0.9 else Vector((0.0, 1.0, 0.0))
        t = normal.cross(helper).normalized()
        b = normal.cross(t)
        start = point + normal * (3.0 * eps)
        leave = 0 if blocked(start, normal) else 1
        for tilt, phase in ((45.0, 0.0), (75.0, 30.0)):
            s, c = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
            for k in range(6):
                a = math.radians(phase + 60.0 * k)
                if not blocked(start, t * (s * math.cos(a)) + b * (s * math.sin(a)) + normal * c):
                    leave += 1
                    if leave > 1:
                        return False
        return True

    seen = [(origin, direction) for origin, direction in taken
            if not (shut_in(origin, -direction) and shut_in(origin, direction))]
    hidden = 1.0 - len(seen) / tested
    if hidden < const.QUAD_HIDDEN_MIN or not seen:
        return thin / tested
    shrink = math.sqrt(1.0 - hidden)                            # the worker's quad edge on the surface that is left
    limit, reach = limit * shrink, reach * shrink
    thin_seen = 0
    for origin, direction in seen:
        gone, wall, behind = 0.0, None, False
        for _ in range(12):
            hit, location, hit_normal, _index, hit_ob, _matrix = scene.ray_cast(depsgraph, origin, direction)
            if not hit:
                break
            gone += (location - origin).length + eps
            origin = location + direction * eps
            if hit_ob.original not in chosen_set:
                continue
            behind = True
            if hit_normal.dot(direction) > 0.0 and not (shut_in(location, hit_normal) and shut_in(location, -hit_normal)):
                wall = gone                                      # a back side that can be seen: the other face of this wall
                break
            if gone >= reach:
                break
        if (wall is not None and wall < limit) or not behind:
            thin_seen += 1
    return thin_seen / len(seen)


def quad_expectation(thin, budget):
    """What to tell the user before a Quad-dominant run: (level, text). `thin` is quad_precheck()'s share.

    Measured on 17 corpus models at 5,000 / 10,000 / 20,000 faces (README, "The estimate before the run"): the
    worker's final thin share lies between the pre-check's share and QUAD_THIN_GAIN x that + QUAD_THIN_BASE (thin
    regions are closed, grown by one quad edge, and small pieces join them). Thin surface costs 1.5 triangles
    where thick surface costs one quad, which gives the quad share of the faces."""
    if thin is None:
        return "INFO", ""
    if thin > const.QUAD_THIN_FALLBACK:
        return "ERROR", "Mostly thin parts: will likely fall back to Game"

    def quads(share):
        return (1.0 - share) / ((1.0 - share) + 1.5 * share)
    most = min(1.0, const.QUAD_THIN_GAIN * thin + const.QUAD_THIN_BASE)
    lo = int(5 * math.floor(100 * quads(most) / 5 + 1e-9))
    hi = int(5 * math.ceil(100 * quads(thin) / 5 - 1e-9))
    text = "Expect about %d-%d %% quads" % (lo, max(hi, lo + 5))
    if thin > const.QUAD_THIN_RISK:
        return "WARNING", text + ", or a fallback to Game: many thin parts"
    return "INFO", text


_estimate = dict(key=None, thin=None, wanted=None)


def _skip_hidden():
    """The "Remove hidden surface" tick: the worker reads a crust without its inner layers only with it."""
    try:
        return bool(settings(bpy.context).remove_hidden)
    except Exception:                              # noqa: no scene, settings not registered
        return True


def _estimate_key(objects, budget):
    return (budget, _skip_hidden()) + tuple((ob.session_uid, len(ob.data.polygons), tuple(round(x, 5) for x in ob.matrix_world.to_scale()))
                             for ob in objects)


def quad_estimate_now(context, objects, budget):
    """Run the pre-check and remember it for the panel. Returns the thin share (or None)."""
    key = _estimate_key(objects, budget)
    try:
        thin = quad_precheck(objects, budget, context.evaluated_depsgraph_get(), context.scene,
                             skip_hidden=_skip_hidden())
    except Exception:                              # noqa: an estimate must never break the panel
        thin = None
    _estimate.update(key=key, thin=thin, wanted=None)
    return thin


def _estimate_timer():
    """The pre-check, run from a timer and not from the panel's draw()."""
    key, _estimate["wanted"] = _estimate["wanted"], None
    try:
        context = bpy.context
        if key is None or is_running() or context.mode != "OBJECT":
            return None
        objects = sources(context)
        budget = settings(context).budget()
        if _estimate_key(objects, budget) != key:
            return None                            # the selection changed meanwhile: the next draw asks again
        quad_estimate_now(context, objects, budget)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:                              # noqa: an estimate must never break anything
        _estimate.update(key=key, thin=None)
    return None


def estimate_allowed(context):
    """The pre-check runs only in Object Mode and while no job is running."""
    return context.mode == "OBJECT" and not is_running()


def quad_estimate_cached(objects, budget, allowed=True):
    """For the panel: (known, thin share). When the selection or the budget changed, the pre-check is started
    from a timer a moment later and (False, None) is returned until it is there. With `allowed` False (a job
    runs, or not Object Mode) nothing is started, and the panel must show nothing instead of "Estimating..."."""
    key = _estimate_key(objects, budget)
    if _estimate["key"] == key:
        return True, _estimate["thin"]
    if not allowed:
        _estimate["wanted"] = None
        return False, None
    if _estimate["wanted"] != key:
        _estimate["wanted"] = key
        if not bpy.app.background and not bpy.app.timers.is_registered(_estimate_timer):
            bpy.app.timers.register(_estimate_timer, first_interval=0.3)
    return False, None


def _common_root(objects):
    """The highest object that is an ancestor of every source (the root of an imported file), or None."""
    def chain(ob):
        out = []
        while ob.parent is not None:
            ob = ob.parent
            out.append(ob)
        return out
    shared = None
    for ob in objects:
        mine = chain(ob)
        shared = mine if shared is None else [x for x in shared if x in mine]
        if not shared:
            return None
    return shared[-1] if shared else None


def _is_generic(name):
    """True for a name that says nothing about the model: wrappers that importers and download sites put around
    it ("Sketchfab_model", "RootNode", "GLTF_SceneRootNode"), numbered defaults ("Object_10", "Mesh_0",
    "geometry_0", "Material.001"), generator ids ("tripo_node_<uuid>_0") and hashes."""
    tokens = [t for t in _SPLIT.split(_UUID.sub(" ", name).lower()) if t]
    for token in tokens:
        word = token.rstrip("0123456789")
        whole = len(tokens) == 1                       # the only word there is: "F16", "T800", "cafe2024" are names
        if not word or word in GENERIC_WORDS or (len(word) == 1 and word != token and not whole):   # "12", "v128"
            continue
        if (len(token) >= (12 if whole else 8) and all(c in "0123456789abcdef" for c in token)
                and any(c.isdigit() for c in token)):
            continue                                                                       # part of a hash
        return False
    return True


def _file_stem(name):
    """"Wild_Pig_0130023416_texture.fbx" -> "Wild_Pig": a file name that an exporter kept as a node name, without
    its extension, export time stamps and the word "texture". None when the name is not a file name."""
    low = name.lower()
    hit = _FILE_EXT.search(low)
    if not hit or hit.start() == 0:
        return None
    parts = [t for t in re.split(r"[\s_]+", name[:hit.start()]) if t]
    parts = [t for t in parts if not (t.isdigit() and len(t) >= 6) and t.lower() != "texture"]
    return "_".join(parts)


def _short(name):
    """At most NAME_MAX UTF-8 bytes: Blender 4.x datablock names hold 63 bytes, and "_asset_02_basecolor" follows."""
    while len(name.encode("utf-8")) > NAME_MAX:
        name = name[:-1]
    return name.rstrip(" _.-") or FALLBACK_NAME


def _is_ancestor(candidate, ob):
    ob = ob.parent
    while ob is not None:
        if ob is candidate:
            return True
        ob = ob.parent
    return False


def _main_material(objects):
    """Name of the material that covers most triangles of the objects, or ""."""
    weight = {}
    for ob in objects:
        mat = ob.material_slots[0].material if ob.material_slots else None
        if mat is not None:
            weight[mat.name] = weight.get(mat.name, 0) + triangle_count([ob])
    return max(weight, key=weight.get) if weight else ""


def base_name(objects, lead):
    """Name for the asset, at most NAME_MAX UTF-8 bytes. Walks from the selection upwards and takes the first name
    that says something: the object's own name (one object), then the ancestors that all objects share, nearest
    first. A file name kept as a node name ("Wild_Pig_0130023416_texture.fbx") gives its stem ("Wild_Pig").
    When every level is a wrapper or a numbered default: the lead object's own name (several objects), the
    material that covers most of the selection, the lead's mesh name, and at last FALLBACK_NAME."""
    names = [lead.name] if len(objects) == 1 else []
    ob = lead.parent
    while ob is not None:
        if all(other is lead or _is_ancestor(ob, other) for other in objects):
            names.append(ob.name)
        ob = ob.parent
    if len(objects) > 1:
        names.append(lead.name)
    names += [_main_material(objects), lead.data.name]
    for name in names:
        stem = _file_stem(name)
        if stem is not None:
            name = stem
        if name and not _is_generic(name):
            return _short(name)
    return FALLBACK_NAME


def _unsaved_images(objects):
    names = set()
    for ob in objects:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                im = getattr(node, "image", None) if node.type == "TEX_IMAGE" else None
                if im is not None and im.is_dirty and im.packed_file is None:
                    names.add(im.name)
    return sorted(names)


def _by_uid(collection, uid, name):
    """A datablock by its session id (survives a rename), else by the name it had."""
    for item in collection:
        if item.session_uid == uid:
            return item
    return collection.get(name)


# ---------------------------------------------------------------------------------------------- the running job

SOURCE_PROPERTY = "assetbake_source"    # on every asset: JSON, the objects it was made from (Select Source)
_session = uuid.uuid4().hex            # session ids of datablocks mean something only until another file is loaded

_job = None
_stats = None                # full statistics of the last finished job (not saved with the file; tests read it)
_drawn = (None, None)        # progress the panel showed last
WAITING = "Done - return to Object Mode to get the result"


def current_job():
    return _job


def last_stats():
    return _stats


def is_running():
    """A job is under way: the worker runs, or its result waits for Object Mode."""
    return _job is not None and _job.state in ("running", "ready")


def _redraw():
    """Redraw the sidebars (not the viewports), and only when the progress changed."""
    global _drawn
    job = _job
    shown = (round(job.fraction, 3), job.text, job.state) if job is not None else None
    if shown == _drawn:
        return
    _drawn = shown
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "UI":
                        region.tag_redraw()


def _store_report(job, report):
    scene = _by_uid(bpy.data.scenes, job.context.get("scene_uid"), job.context.get("scene", ""))
    if scene is not None:
        getattr(scene, const.SETTINGS).report = json.dumps(report)


def _object_mode(scene):
    return all(ob.mode == "OBJECT" for ob in scene.objects)


def _append_result(job, scene):
    """Bring the assets into the scene next to the source, hide the source and the assets made from the same
    source before. One undo step. Returns (names of the assets, names of the earlier assets that were hidden)."""
    ctx = job.context
    layer = scene.view_layers.get(ctx["view_layer"]) or scene.view_layers[0]
    groups = set(bpy.data.node_groups)
    images = set(bpy.data.images)
    shared = bpy.data.node_groups.get(OCCLUSION_GROUP)
    with bpy.data.libraries.load(job.result_blend, link=False) as (data_from, data_to):
        data_to.objects = [n for n in job.report["objects"] if n in data_from.objects]
    assets = [ob for ob in data_to.objects if ob is not None]
    if not assets:
        raise RuntimeError("the result file holds no asset")
    if shared is not None:
        # the file already has the glTF settings group: the appended copy would be "... .001", a name the glTF
        # exporter does not recognise, and the occlusion would be lost on export
        for group in set(bpy.data.node_groups) - groups:
            if group.name.startswith(OCCLUSION_GROUP):
                group.user_remap(shared)
                bpy.data.node_groups.remove(group)
    for image in set(bpy.data.images) - images:
        # the append rebases the path into the job folder, which is deleted: "Unpack" must not write there
        image.filepath_raw = "//textures/" + re.sub(r'[:*?"<>|/\\]', "_", image.name).rstrip(" .") + ".png"
        for pf in image.packed_files:
            pf.filepath = image.filepath_raw
    collection = _by_uid(bpy.data.collections, ctx["collection_uid"], ctx["collection"]) if ctx["collection"] else None
    if collection is None or collection not in scene.collection.children_recursive:
        collection = scene.collection
    for ob in layer.objects:
        ob.select_set(False, view_layer=layer)
    for ob in assets:
        collection.objects.link(ob)
    layer.update()
    record, used = [], []
    for uid, name, hide_render in ctx["sources"]:
        src = _by_uid(bpy.data.objects, uid, name)
        if src is None:
            continue                               # deleted while the job ran
        record.append(dict(uid=uid, name=src.name, hide_render=bool(hide_render)))
        used.append(src)
        src.hide_render = True
        try:
            src.hide_set(True, view_layer=layer)
        except RuntimeError:
            pass                                   # not in this view layer any more
    # assets made from the same source before sit exactly where the new one is: the new one would not be seen
    hidden = []
    for ob in _earlier_versions(scene, used, assets):
        try:
            if ob.visible_get(view_layer=layer):
                ob.hide_set(True, view_layer=layer)        # the eye icon only; render visibility stays as it is
                hidden.append(ob.name)
        except RuntimeError:
            pass
    for ob in assets:
        # every asset of the job (also one per part) remembers the whole source: Select Source brings it back
        ob[SOURCE_PROPERTY] = json.dumps(dict(session=_session, objects=record, asset=ob.session_uid,
                                              matrix=[round(v, 6) for row in ob.matrix_world for v in row]))
        try:
            ob.select_set(True, view_layer=layer)
        except RuntimeError:
            pass
    if assets[0].name in layer.objects:
        layer.objects.active = assets[0]
    return [ob.name for ob in assets], hidden


def _earlier_versions(scene, used, new):
    """Assets in the scene that this session made from exactly the source objects `used` (not the assets in
    `new`) and that still sit where they were made. A copy the user duplicated (the record names another asset),
    an asset moved aside, or one from a file opened later (session ids mean nothing then) is left alone."""
    want = {ob.session_uid for ob in used}
    if not want:
        return []
    by_uid = None
    out = []
    for ob in scene.objects:
        if ob.type != "MESH" or ASSET_PROPERTY not in ob or any(ob is n for n in new):
            continue
        try:
            data = json.loads(ob.get(SOURCE_PROPERTY) or "")
            if data.get("session") != _session or data.get("asset") != ob.session_uid:
                continue
            uids = {int(r["uid"]) for r in data["objects"] if r.get("uid") is not None}
            matrix = data["matrix"]
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if uids != want or len(uids) != len(data["objects"]):
            continue
        now = [v for row in ob.matrix_world for v in row]
        if len(matrix) != 16 or max(abs(a - b) for a, b in zip(now, matrix)) > 1e-4:
            continue
        if by_uid is None:
            by_uid = {o.session_uid for o in bpy.data.objects}
        if uids <= by_uid:                       # every source of the record still exists
            out.append(ob)
    return out


def _undo_push():
    """The scene was changed from a timer, outside any operator: record it as one undo step."""
    wm = bpy.context.window_manager
    if bpy.app.background or not wm.windows:
        return
    try:
        with bpy.context.temp_override(window=wm.windows[0]):
            bpy.ops.ed.undo_push(message=const.ACTION)
    except RuntimeError:
        pass


def _fail(job, text, user_error=False):
    """Failure: message in the panel, and the worker's log kept where the user can find and send it."""
    report = dict(ok=False, error=text if user_error else "The job failed: {}.".format(text))
    if not user_error:
        report["log"] = job.keep_log(text)
    _store_report(job, report)
    job.state = "failed"
    job.cleanup()


def _finish(job):
    """Called on every tick once the worker has ended. Returns True while the result still waits."""
    global _stats
    if job.state == "failed":
        _fail(job, job.error, job.user_error)
        return False
    scene = _by_uid(bpy.data.scenes, job.context["scene_uid"], job.context["scene"])
    if scene is None:
        _fail(job, "the scene the job was started in was deleted")
        return False
    if not _object_mode(scene):
        # appending while an object is in Edit, Sculpt or Paint mode leaves that object stuck in its mode and
        # puts the undo step into the wrong undo stack: wait
        job.state, job.text, job.fraction = "ready", WAITING, 1.0
        return True
    try:
        names, hidden = _append_result(job, scene)
    except Exception as ex:                        # noqa: whatever went wrong, the user gets a message, not a trace
        _fail(job, "could not bring the result into the scene ({})".format(ex))
        return False
    report = job.report
    _stats = report
    earlier = [["INFO", "Earlier versions of this asset were hidden: %s. Hidden in the viewport only: they still "
                        "show in a render. Click their eye icon in the Outliner to show them again." % _name_list(hidden)]] if hidden else []
    _store_report(job, dict(
        ok=True, objects=names, seconds=round(job.seconds, 1),
        assets=[dict(name=name, lines=e["lines"], messages=e["messages"]) for e, name in zip(report["assets"], names)],
        messages=job.context["messages"] + report["messages"] + earlier))
    job.state = "done"
    job.cleanup()
    _undo_push()
    return False


def tick():
    """One step of the watcher. Returns True while a job is under way (running or waiting for Object Mode)."""
    job = _job
    if job is None or job.state not in ("running", "ready"):
        return False
    if job.state == "running" and job.poll():
        return True
    return _finish(job)


def _timer():
    busy = tick()
    _redraw()
    if not busy:
        return None
    return 0.5 if _job.state == "ready" else 0.1


@persistent
def _on_load_pre(*_):
    global _session
    _session = uuid.uuid4().hex
    if is_running():
        _job.cancel()


# ---------------------------------------------------------------------------------------------- operators

class ASSETBAKELITE_OT_make(Operator):
    bl_idname = const.OPERATOR + ".make"
    bl_label = const.ACTION
    bl_description = ("Build a low-poly, UV-unwrapped, textured asset from the selected objects. The work runs in "
                      "the background; the selected objects are not changed")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and not is_running()

    def execute(self, context):
        global _job, _drawn
        p = settings(context)
        objs = sources(context)
        if not objs:
            self.report({"ERROR"}, "Select one or more mesh objects first")
            return {"CANCELLED"}
        tris = triangle_count(objs)
        if tris == 0:
            self.report({"ERROR"}, "The selected meshes have no faces")
            return {"CANCELLED"}
        if tris > const.REFUSE_TRIS:
            text = "The selection has {:,} triangles. The limit is {:,}: decimate the source first.".format(
                tris, const.REFUSE_TRIS)
            p.report = json.dumps(dict(ok=False, error=text))
            self.report({"ERROR"}, text)
            return {"CANCELLED"}
        mode, binding = p.mode, None
        if mode == "QUADS":
            if p.budget() < const.QUAD_MIN_BUDGET:
                text = "Quad-dominant needs a budget of at least {:,} faces.".format(const.QUAD_MIN_BUDGET)
                p.report = json.dumps(dict(ok=False, error=text))
                self.report({"ERROR"}, text)
                return {"CANCELLED"}
            binding = vendor.find_module_path()[0]
        root = _common_root(objs) if len(objs) > 1 else None
        active = context.view_layer.objects.active
        lead = active if active in objs else objs[0]
        placed = root if root is not None else lead
        messages = []
        messages.append(["INFO", upsell_line(context, objs)])
        if mode == "QUADS" and binding is None:
            # the panel does not offer the mode then; this is for a file saved with it on another system
            mode = "GAME"
            messages.append(["WARNING", quad_unavailable(refresh=True) + " A Game asset was made."])
        unsaved = _unsaved_images(objs)
        if unsaved:
            text = ("Texture with unsaved changes: %s. Save or pack it, otherwise the bake uses the old "
                    "picture." % ", ".join(unsaved[:4]))
            messages.append(["WARNING", text])
            self.report({"WARNING"}, text)
        job = Job()
        collection = lead.users_collection[0] if lead.users_collection else None
        job.context = dict(scene=context.scene.name, scene_uid=context.scene.session_uid,
                           view_layer=context.view_layer.name,
                           sources=[(ob.session_uid, ob.name, ob.hide_render) for ob in objs],
                           collection=collection.name if collection else "",
                           collection_uid=collection.session_uid if collection else 0, messages=messages)
        try:
            job.make_dir()
            bpy.data.libraries.write(job.source_blend, set(objs), path_remap="ABSOLUTE")
            job.start(dict(
                objects=[dict(name=ob.name, matrix=[x for row in ob.matrix_world for x in row]) for ob in objs],
                base_name=base_name(objs, lead),
                origin=list(placed.matrix_world.translation),
                options=dict(mode=mode, binding=binding, budget=p.budget(), texture_size=const.LITE_TEXTURE_SIZE,
                             maps=p.maps(), scope="SELECTION", remove_hidden=p.remove_hidden)))
        except Exception as ex:                    # noqa: disk full, no permission, the process cannot be started
            job.cleanup()
            text = "Could not start the background job: {}".format(ex)
            p.report = json.dumps(dict(ok=False, error=text))
            self.report({"ERROR"}, text)
            return {"CANCELLED"}
        _job = job
        _drawn = (None, None)
        p.report = ""
        if not bpy.app.background and not bpy.app.timers.is_registered(_timer):
            bpy.app.timers.register(_timer, first_interval=0.1)
        return {"FINISHED"}


class ASSETBAKELITE_OT_cancel(Operator):
    bl_idname = const.OPERATOR + ".cancel"
    bl_label = "Cancel"
    bl_description = "Stop the background job. Nothing in the scene changes"

    @classmethod
    def poll(cls, context):
        return is_running()

    def execute(self, context):
        job = _job
        job.cancel()
        _store_report(job, dict(ok=False, cancelled=True, error="Cancelled. Nothing was changed."))
        return {"FINISHED"}


class ASSETBAKELITE_OT_copy_log_path(Operator):
    bl_idname = const.OPERATOR + ".copy_log_path"
    bl_label = "Copy log path"
    bl_description = "Copy the location of the log file of the failed job, to attach the file to a support request"

    def execute(self, context):
        context.window_manager.clipboard = log_path()
        self.report({"INFO"}, "Copied: " + log_path())
        return {"FINISHED"}


def _last_assets(context):
    """The assets of the last run that are still in the file."""
    try:
        names = json.loads(settings(context).report or "{}").get("objects", [])
    except (ValueError, AttributeError):
        names = []
    return [ob for ob in (bpy.data.objects.get(n) for n in names) if ob is not None and ASSET_PROPERTY in ob]


def source_records(assets):
    """What the assets remember about their sources: (records without doubles, assets that remember nothing).
    A record is dict(uid, name, hide_render); uid is None when it cannot be trusted any more (the file was
    reopened: session ids start again and may belong to another object)."""
    records, seen, unknown = [], set(), []
    for asset in assets:
        try:
            data = json.loads(asset.get(SOURCE_PROPERTY) or "")
            live = data.get("session") == _session
            items = [(int(r["uid"]) if live else None, str(r["name"]), bool(r.get("hide_render", False)))
                     for r in data["objects"]]
        except (ValueError, TypeError, KeyError, AttributeError):
            unknown.append(asset)                  # made by a version before 0.1.3, or the property was edited
            continue
        for uid, name, hide_render in items:
            if (uid, name) not in seen:
                seen.add((uid, name))
                records.append(dict(uid=uid, name=name, hide_render=hide_render))
    return records, unknown


def _find_source(record):
    """The source object of a record, or None. In the session the asset was made in, the session id decides: a
    renamed source is found, and another object that got the name of a deleted source is not taken for it. After
    the file was reopened only the name is left."""
    if record["uid"] is not None:
        for ob in bpy.data.objects:
            if ob.session_uid == record["uid"]:
                return ob
        return None
    ob = bpy.data.objects.get(record["name"])
    return ob if ob is not None and ASSET_PROPERTY not in ob else None


def select_source_targets(context, last=False):
    """Assets whose source Select Source works on: the selected assets, or (last=True, the button in the Result
    box, or nothing selected) the assets of the last run."""
    if not last:
        picked = [ob for ob in context.selected_objects if ob.type == "MESH" and ASSET_PROPERTY in ob]
        if picked:
            return picked
    return _last_assets(context)


class ASSETBAKELITE_OT_select_source(Operator):
    bl_idname = const.OPERATOR + ".select_source"
    bl_label = "Select Source"
    bl_description = ("Show and select the original objects this asset was made from, to make another version with "
                      "other settings. The asset stays where it is")
    bl_options = {"REGISTER", "UNDO"}

    last: BoolProperty(options={"HIDDEN", "SKIP_SAVE"}, default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and not is_running()

    def execute(self, context):
        assets = select_source_targets(context, self.last)
        if not assets:
            self.report({"ERROR"}, "Select an asset made with %s first" % const.ACTION)
            return {"CANCELLED"}
        records, unknown = source_records(assets)
        if not records:
            self.report({"ERROR"}, "This asset does not remember its source (it was made with an older version). "
                                   "Unhide the original objects in the Outliner")
            return {"CANCELLED"}
        layer = context.view_layer
        found, missing = [], []
        for record in records:
            ob = _find_source(record)
            if ob is None or context.scene.objects.get(ob.name) is not ob:
                missing.append(record["name"])
            elif all(ob is not other for other, _ in found):
                found.append((ob, record))
        if not found:
            self.report({"ERROR"}, "The source is not in this scene any more (deleted, or renamed after the file "
                                   "was reopened): %s" % _name_list(missing))
            return {"CANCELLED"}
        chosen, blocked = [], []
        for ob, record in found:
            ok = False
            try:
                ob.hide_set(False, view_layer=layer)
                if not ob.visible_get(view_layer=layer):
                    ob.hide_viewport = False               # the monitor icon of the Outliner
                ob.hide_select = False
                ob.select_set(True, view_layer=layer)
                ok = ob.select_get(view_layer=layer)
            except RuntimeError:
                pass                                       # not in this view layer: its collection is excluded
            if ok:
                ob.hide_render = record["hide_render"]     # as it was before the job
                chosen.append(ob)
            else:
                blocked.append(ob.name)
        if not chosen:
            self.report({"ERROR"}, "The source is in a collection that is hidden or switched off. Show the "
                                   "collection in the Outliner, then click again: %s" % _name_list(blocked))
            return {"CANCELLED"}
        for ob in layer.objects:
            if all(ob is not c for c in chosen):
                ob.select_set(False, view_layer=layer)
        layer.objects.active = chosen[0]
        problems = []
        if missing:
            problems.append("not found (deleted, or renamed after the file was reopened): %s" % _name_list(missing))
        if blocked:
            problems.append("in a hidden collection, show it in the Outliner: %s" % _name_list(blocked))
        if unknown:
            problems.append("no source remembered for: %s" % _name_list([ob.name for ob in unknown]))
        text = "Selected %d of %d source object%s" % (len(chosen), len(records), "" if len(records) == 1 else "s")
        if problems:
            self.report({"WARNING"}, text + "; " + "; ".join(problems))
        else:
            self.report({"INFO"}, text)
        return {"FINISHED"}


def _name_list(names, most=4):
    return ", ".join(names[:most]) + (" and %d more" % (len(names) - most) if len(names) > most else "")


def export_targets(context):
    """Assets to export: the selected ones, or the ones of the last run when none is selected."""
    picked = [ob for ob in context.selected_objects if ob.type == "MESH" and ASSET_PROPERTY in ob]
    if picked:
        return picked
    return _last_assets(context)


def export_paths(filepath, targets):
    """One file per asset. A single asset gets exactly the name that was typed; several get the typed name plus
    their own: <typed>_<asset>.glb."""
    folder, stem = os.path.dirname(filepath), os.path.splitext(os.path.basename(filepath))[0]
    if len(targets) == 1:
        return [os.path.join(folder, stem + ".glb")]
    return [os.path.join(folder, "%s_%s.glb" % (stem, bpy.path.clean_name(ob.name))) for ob in targets]


class ASSETBAKELITE_OT_export(Operator, ExportHelper):
    bl_idname = const.OPERATOR + ".export_glb"
    bl_label = "Export GLB"
    bl_description = ("Save the asset as a GLB file for a game engine: baked triangles, scale applied, base colour and "
                      "normal map (OpenGL) and ambient occlusion")
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(export_targets(context))

    def invoke(self, context, event):
        targets = export_targets(context)
        if targets and not self.filepath:
            self.filepath = (targets[0].name if len(targets) == 1 else "assets") + ".glb"
        return super().invoke(context, event)

    def execute(self, context):
        p = settings(context)
        targets = export_targets(context)
        if not targets:
            self.report({"ERROR"}, "Select an asset made with %s" % const.ACTION)
            return {"CANCELLED"}
        if not gltf_export.available():
            self.report({"ERROR"}, "Blender's glTF 2.0 add-on is switched off: enable it in Preferences > Add-ons")
            return {"CANCELLED"}
        done = []
        try:
            for ob, path in zip(targets, export_paths(self.filepath, targets)):
                done.append(gltf_export.export_asset(context, ob, path, directx=False,
                                                     save_textures=p.export_textures))
        except Exception as ex:                    # noqa: the exporter raises many types
            self.report({"ERROR"}, "Export failed: {}".format(ex))
            return {"CANCELLED"}
        warned = [item["warning"] for item in done if item.get("warning")]
        for text in warned:
            self.report({"WARNING"}, text)
        self.report({"INFO"}, "Exported %d file(s) to %s" % (len(done), os.path.dirname(self.filepath)))
        return {"FINISHED"}


classes = (AssetBakeLiteSettings, ASSETBAKELITE_OT_make, ASSETBAKELITE_OT_cancel, ASSETBAKELITE_OT_copy_log_path,
           ASSETBAKELITE_OT_select_source, ASSETBAKELITE_OT_export)


def register_props():
    _unavailable["text"] = None
    setattr(bpy.types.Scene, const.SETTINGS, PointerProperty(type=AssetBakeLiteSettings))
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)


def unregister_props():
    if is_running():
        _job.cancel()               # never leave a worker behind when the add-on is switched off
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    if bpy.app.timers.is_registered(_estimate_timer):
        bpy.app.timers.unregister(_estimate_timer)
    _estimate.update(key=None, thin=None, wanted=None)
    delattr(bpy.types.Scene, const.SETTINGS)
