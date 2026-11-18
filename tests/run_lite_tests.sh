#!/bin/zsh
# Headless test of the BUILT Lite extension on Blender 5.2 only (tests/bl.sh runs it under the memory guard,
# in a throwaway user folder). usage: tests/run_lite_tests.sh [out dir]
D=${0:A:h}; R=${D:h}
O=${1:-$D/out/lite}; [[ $O != /* ]] && O=$PWD/$O
mkdir -p $O
"$R/build.sh" > $O/build.log 2>&1 || { echo "BUILD FAILED (see $O/build.log)"; exit 1 }
ZIP=$(ls -t $R/dist/*.zip | head -1)
PROFILE="lite 52"
rm -rf "$D/profiles/$PROFILE"
"$D/bl.sh" 52 "$PROFILE" --command extension install-file -r user_default -e "$ZIP" > $O/install.log 2>&1
echo "ZIP $ZIP install exit $?"
"$D/bl.sh" 52 "$PROFILE" --python "$D/test_lite.py" -- $O > $O/lite.log 2>&1
grep -E "^ADDON|^CASE|Traceback" $O/lite.log
grep "^CHECK.* FAIL" $O/lite.log
