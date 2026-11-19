#!/bin/zsh
# Build and validate the extension zip: dist/<id>-<version>.zip
# usage: ./build.sh            (uses the newest Blender; any 4.2+ builds the same zip)
HERE=${0:A:h}
ID=$(sed -n 's/^id = "\(.*\)"/\1/p' "$HERE/assetbake_lite/blender_manifest.toml")
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$HERE/assetbake_lite/blender_manifest.toml")
ZIP="$HERE/dist/$ID-$VER.zip"
mkdir -p "$HERE/dist"; rm -f "$ZIP"
find "$HERE/assetbake_lite" -name __pycache__ -type d -prune -exec rm -rf {} +
"$HERE/tests/bl.sh" 52 build --command extension build --source-dir "$HERE/assetbake_lite" --output-dir "$HERE/dist" || exit 1
"$HERE/tests/bl.sh" 52 build --command extension validate "$ZIP" || exit 1
# Lite carries no compiled remesher and no vendor folder
unzip -l "$ZIP" | grep -q " vendor/" && { echo "vendor/ must NOT be in the Lite zip"; exit 1 }
unzip -l "$ZIP" | grep -qE "\.(so|pyd|dylib|dll)$" && { echo "a compiled module is in the Lite zip"; exit 1 }
unzip -l "$ZIP" | grep -q " LICENSE$" || { echo "MISSING in the zip: LICENSE"; exit 1 }
echo "BUILT $ZIP ($(stat -f %z "$ZIP") bytes, $(unzip -l "$ZIP" | tail -1 | awk '{print $2}') files, no vendor folder)"
