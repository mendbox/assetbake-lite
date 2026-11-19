# SPDX-License-Identifier: GPL-3.0-or-later
"""Names in one place. The short product name is ADDON_NAME; the long store name (SEO) lives in blender_manifest.toml."""
import os
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))


def _manifest():
    try:
        with open(os.path.join(HERE, "blender_manifest.toml"), "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_M = _manifest()
ADDON_ID = _M.get("id", "assetbake_lite")
STORE_NAME = _M.get("name", "AssetBake Lite - AI Mesh to Low Poly Game Asset")   # the long keyword name of the store listing (manifest); never shown in the UI
ADDON_NAME = "AssetBake Lite"            # the short product name: sidebar tab, panel titles, messages
VERSION = _M.get("version", "?")

TAB = ADDON_NAME                    # sidebar tab
ACTION = "Make Game Asset"          # the one button; also the name of the undo step
SETTINGS = "assetbake_lite"               # bpy.types.Scene.<SETTINGS>
OPERATOR = "assetbake_lite"               # operators are bpy.ops.<OPERATOR>.*
TMP_ENV = "AI_ASSET_TMP"            # environment variable: folder for job folders (tests)

BUDGETS = (1000, 2000, 5000)
TEXTURE_SIZES = (1024,)
WARN_TRIS = 5000000
REFUSE_TRIS = 8000000

from .pipeline import QUAD_MAX_SOURCE_TRIS          # noqa: E402,F401  (the worker's limit: one number for both)

# Quad-dominant mode. The budget counts faces there.
QUAD_MIN_BUDGET = 5000              # below this nearly every model falls back to Game (Finding 11)
QUAD_ADVISED_BUDGET = 10000         # the mode was tuned and gated at this budget; below it the quad share drops
# the estimate shown before the run (ops.quad_precheck / ops.quad_expectation)
QUAD_ESTIMATE_SAMPLES = 1500
QUAD_THIN_EDGES = 1.25              # a wall thinner than this many quad edges is thin (the worker's thin_k)
QUAD_THIN_GAIN = 1.4                # the worker's final thin share lies between the pre-check's share and
QUAD_THIN_BASE = 0.12               # GAIN x that share + BASE (17 corpus models, three budgets)
QUAD_THIN_FALLBACK = 0.45           # pre-check share above this: "will likely fall back to Game"
QUAD_THIN_RISK = 0.27               # above this the fallback is mentioned next to the expected quad share
QUAD_HIDDEN_MIN = 0.2               # a mostly-thin selection with this much shut-in surface is read again without it (the worker's MIN_HIDDEN)

# Lite limits (also enforced in the worker, pipeline/run.py)
LITE_MAX_BUDGET = 5000
LITE_TEXTURE_SIZE = 1024
LITE_MAPS = ("basecolor", "ao")      # plus the normal map, which is always baked
FULL_URL = "https://mendboxstudio.com/addons/assetbake"    # set in tools/build_lite.py
FULL_FEATURES = "Full version adds: Quad-dominant mode, budgets up to 50,000 triangles, 2k and 4k textures, roughness, metallic, emission and alpha maps, one asset per part, DirectX normal maps."
