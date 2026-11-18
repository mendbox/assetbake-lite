# SPDX-License-Identifier: GPL-3.0-or-later
"""The asset pipeline. Runs inside the background worker (worker.py), never in the user's Blender session,
except gltf_export.py, which the Export button calls directly.

Modules import each other with relative imports only and never import the parent package, so the worker can load
this folder on its own.

All geometry work happens in "working units": the source is scaled so that its bounding-box diagonal is 100 and
moved to the origin. A distance in working units is therefore a percentage of the model size, and every tolerance
below is relative to the model.
"""

# Every line the worker writes for the add-on starts with this; everything else on its stdout is Blender's own text.
PROTOCOL_PREFIX = "@@ASSETBAKE "

# Quad-dominant mode takes sources up to this many triangles (the research corpus ends at 1.76 million; 50 s and
# 7 GB there). The panel (const.py) and the worker (quads.py) read it from here, so they cannot disagree.
QUAD_MAX_SOURCE_TRIS = 2000000

# Custom property that marks a result object (JSON with its key numbers).
ASSET_PROPERTY = "assetbake_lite"
