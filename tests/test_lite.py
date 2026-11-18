"""Headless test of the Lite build, through its real operators and its real background worker.

  tests/bl.sh 52 <profile> --python tests/test_lite.py -- <out_dir>      (see run_lite_tests.sh)

Prints one CHECK line per check and a final CASE line.
"""
import importlib
import json
import os
import struct
import sys
import time

import addon_utils
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
BOAR = os.path.join(ROOT, "..", "research", "ai-remesh-2026-11", "models_real", "meshy_boar.glb")
LITE_ID = "assetbake_lite"
results = []


def check(name, ok, note=""):
    results.append(bool(ok))
    print("CHECK %02d %-62s %s %s" % (len(results), name, "OK" if ok else "FAIL", note), flush=True)


def load_addon():
    installed = "bl_ext.user_default." + LITE_ID
    if addon_utils.check(installed)[1]:
        return importlib.import_module(installed), "installed extension"
    sys.path.insert(0, ROOT)
    module = importlib.import_module(LITE_ID)
    module.register()
    return module, "source folder"


addon, how = load_addon()
print("ADDON from", how, flush=True)
ops = importlib.import_module(addon.__name__ + ".ops")
const = importlib.import_module(addon.__name__ + ".const")
runmod = importlib.import_module(addon.__name__ + ".pipeline.run")
os.environ[const.TMP_ENV] = os.path.join(OUT, "tmp")
OP = getattr(bpy.ops, const.OPERATOR)


def settings():
    return getattr(bpy.context.scene, const.SETTINGS)


def run(op, **kw):
    try:
        return op(**kw)
    except RuntimeError as ex:
        return str(ex)


def wait_job(timeout=600):
    t0 = time.perf_counter()
    while ops.tick():
        if time.perf_counter() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Scene, const.SETTINGS):
        addon.register()
    return settings()


def select(objs):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def sphere(loc=(0, 0, 0), seg=128, rings=64):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings, location=loc)
    return bpy.context.active_object


def image_kinds(ob):
    mat = ob.data.materials[0]
    return {n.name for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"}


def parse_glb(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 12)[0]
    J = json.loads(data[20:20 + n])
    prim = J["meshes"][0]["primitives"][0]
    mat = J["materials"][prim["material"]]
    pbr = mat.get("pbrMetallicRoughness", {})
    return dict(images=len(J.get("images", [])), base="baseColorTexture" in pbr, normal="normalTexture" in mat,
                orm="metallicRoughnessTexture" in pbr, occlusion="occlusionTexture" in mat,
                emissive="emissiveTexture" in mat, tris=J["accessors"][prim["indices"]]["count"] // 3)


# ---------------------------------------------------------------------- A: settings and panels
p = fresh()
names = set(p.bl_rna.properties.keys())
check("budgets are 1,000 / 2,000 / 5,000 only", [i.identifier for i in p.bl_rna.properties["budget_preset"].enum_items] == ["1000", "2000", "5000"])
check("default budget 5,000, texture 1k", p.budget() == 5000 and p.texture_size == "1024")
check("texture size has no 2k / 4k", [i.identifier for i in p.bl_rna.properties["texture_size"].enum_items] == ["1024"])
check("no custom budget, no scope, no DirectX option", not names & {"budget_custom", "scope", "normal_convention"})
check("no roughness / metallic / emission / alpha switches (AO is allowed)", not names & {"use_roughness", "use_metallic", "use_emission", "use_alpha"})
check("maps(): base colour + AO", p.maps() == {"basecolor": True, "roughness": False, "metallic": False, "emission": False, "alpha": False, "ao": True}, str(p.maps()))
check("panels: main, result, full-version box", all(hasattr(bpy.types, n) for n in ("ASSETBAKELITE_PT_panel", "ASSETBAKELITE_PT_zresult", "ASSETBAKELITE_PT_zzfull")))
check("full-version URL constant is set (placeholder until named)", const.FULL_URL.startswith("http"), const.FULL_URL)
check("pro operator / settings names not used", not hasattr(bpy.types.Scene, "assetbake") and const.OPERATOR == "assetbake_lite")
clamp = runmod.lite_clamp(dict(mode="QUADS", budget=50000, texture_size=4096, scope="PARTS", maps={"ao": True, "roughness": True, "basecolor": True}))
check("worker clamp: mode, budget, texture, scope, maps", clamp["mode"] == "GAME" and clamp["budget"] == 5000
      and clamp["texture_size"] == 1024 and clamp["scope"] == "SELECTION" and clamp["maps"] == {
          "basecolor": True, "roughness": False, "metallic": False, "emission": False, "alpha": False, "ao": True}, str(clamp))
res = run(OP.make)
check("nothing selected: refused", isinstance(res, str) and "Select" in res, str(res)[:50])

# ---------------------------------------------------------------------- B: a 16k-triangle sphere, budget 5,000
p = fresh()
ball = sphere()
tris_in = len(ball.data.polygons) * 2
select([ball])
res = run(OP.make)
check("job starts", res == {"FINISHED"} and ops.is_running(), str(res))
check("job ends in time", wait_job())
job = ops.current_job()
rep = json.loads(p.report) if p.report else {}
check("job done", job.state == "done" and rep.get("ok") is True, "%s %s" % (job.state, job.error))
if rep.get("ok"):
    assets = [bpy.data.objects[n] for n in rep["objects"]]
    a = assets[0]
    check("one asset", len(assets) == 1)
    told = " | ".join(m[1] for m in rep.get("messages", []))
    check("result box: one model-specific upsell line with the URL", told.count("Full AssetBake:") == 1 and const.FULL_URL in told, told[:200])
    print("UPSELL", told[:300], flush=True)
    check("triangles within 5%% of 5,000 (source %d)" % tris_in, abs(len(a.data.polygons) - 5000) <= 250, str(len(a.data.polygons)))
    check("marked as a Lite asset", const.OPERATOR in a.keys() and "assetbake" not in a.keys())
    kinds = image_kinds(a)
    check("textures: normal + base colour only", kinds == {"asset_normal", "asset_basecolor", "asset_orm"}, str(sorted(kinds)))
    ims = [n.image for n in a.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE"]
    check("textures are 1024 x 1024 and packed", all(tuple(im.size) == (1024, 1024) and im.packed_file is not None for im in ims))
    e = ops.last_stats()["assets"][0]
    check("report lists the maps: normal + basecolor + ao", sorted(e["maps"]) == ["ao", "basecolor", "normal"], str(e["maps"]))
    check("source hidden, not changed", ball.hide_get() and len(ball.data.polygons) * 2 == tris_in)
    # export
    path = os.path.join(OUT, "sphere.glb")
    select([a])
    res = run(OP.export_glb, filepath=path)
    check("export GLB", res == {"FINISHED"} and os.path.isfile(path), str(res))
    g = parse_glb(path)
    check("GLB: base colour + normal + ORM (AO only), no emission", g["base"] and g["normal"] and g["orm"]
          and not g["emissive"] and g["images"] == 3, str(g))
    check("GLB: triangles = asset", g["tris"] == len(a.data.polygons), str(g["tris"]))
    check("no DirectX property on the operator", not hasattr(p, "normal_convention"))

# ---------------------------------------------------------------------- C: tampered settings, the worker still caps
p = fresh()
ball = sphere()
select([ball])
orig_budget, orig_maps = type(p).budget, type(p).maps
type(p).budget = lambda self: 50000
type(p).maps = lambda self: {m: True for m in ("basecolor", "roughness", "metallic", "emission", "alpha", "ao")}
res = run(OP.make)
wait_job()
type(p).budget, type(p).maps = orig_budget, orig_maps
rep = json.loads(p.report) if p.report else {}
check("tampered settings: job done", rep.get("ok") is True, str(rep.get("error")))
if rep.get("ok"):
    a = bpy.data.objects[rep["objects"][0]]
    check("tampered: still 5,000 triangles, 1k, ORM only for AO", abs(len(a.data.polygons) - 5000) <= 250
          and image_kinds(a) == {"asset_normal", "asset_basecolor", "asset_orm"}
          and all(tuple(n.image.size) == (1024, 1024) for n in a.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE"),
          "%d %s" % (len(a.data.polygons), sorted(image_kinds(a))))

# ---------------------------------------------------------------------- D: two objects -> one asset
p = fresh()
b1, b2 = sphere((0, 0, 0), 64, 32), sphere((5, 0, 0), 64, 32)
select([b1, b2])
res = run(OP.make)
wait_job()
rep = json.loads(p.report) if p.report else {}
check("two separate objects: exactly one asset", rep.get("ok") is True and len(rep.get("objects", [])) == 1, str(rep.get("error", rep.get("objects"))))

# ---------------------------------------------------------------------- E: a real AI mesh (Meshy boar)
if os.path.exists(BOAR):
    p = fresh()
    bpy.ops.import_scene.gltf(filepath=BOAR)
    srcs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    select(srcs)
    n_in = ops.triangle_count(srcs)
    t0 = time.perf_counter()
    res = run(OP.make)
    ok_t = wait_job()
    rep = json.loads(p.report) if p.report else {}
    check("boar (%s tris): done in time" % "{:,}".format(n_in), res == {"FINISHED"} and ok_t and rep.get("ok") is True,
          "%.0f s %s" % (time.perf_counter() - t0, rep.get("error", "")))
    if rep.get("ok"):
        a = bpy.data.objects[rep["objects"][0]]
        e = ops.last_stats()["assets"][0]
        check("boar: about 5,000 triangles", abs(len(a.data.polygons) - 5000) <= 250, str(len(a.data.polygons)))
        check("boar: deviation p99 under 0.5 pct, max under 2 pct", e["deviation"]["p99"] <= 0.5 and e["deviation"]["max"] <= 2.0,
              "%.3f / %.3f" % (e["deviation"]["p99"], e["deviation"]["max"]))
        check("boar: bake not empty", not e["texels"].get("empty") and e["texels"]["fill"] > 0.3, "fill %.2f" % e["texels"]["fill"])
        check("boar: normal + base colour + ORM textures", image_kinds(a) == {"asset_normal", "asset_basecolor", "asset_orm"}, str(sorted(image_kinds(a))))
        path = os.path.join(OUT, "boar.glb")
        select([a])
        res = run(OP.export_glb, filepath=path)
        g = parse_glb(path) if os.path.isfile(path) else {}
        check("boar: GLB exported, base + normal + ORM", res == {"FINISHED"} and g.get("base") and g.get("normal") and g.get("orm"), str(g))
else:
    check("boar corpus file present", False, BOAR)

addon.unregister()
check("unregister removes the settings", not hasattr(bpy.types.Scene, const.SETTINGS))
print("CASE lite %s pass %d of %d" % (bpy.app.version_string, sum(results), len(results)), flush=True)
sys.stdout.flush()
os._exit(0 if all(results) else 1)
