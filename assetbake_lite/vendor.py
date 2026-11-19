# SPDX-License-Identifier: GPL-3.0-or-later
"""Locating the compiled quad remesher (module im_remesh, our own binding of Instant Meshes) that ships inside
the extension.

The packaging test T7 showed what works on Blender 4.2, 4.5 and 5.x: no `wheels` in the manifest. The unpacked
module lies in vendor/<platform>/<python tag>/im_remesh/; the add-on never imports it in the user's session, it
only looks whether the folder for this system is there and hands the path to the worker, which puts it on sys.path
(Quad-dominant mode). Without a folder for this system the mode is switched off with a message and Game mode works
as before.

  platform    macos_arm64, macos_x64, linux_x64, windows_x64 (Windows on ARM: no build)
  python tag  cp311 for Python 3.11 (Blender 4.2 / 4.5), abi3 for Python 3.12 and newer (Blender 5.x)

The module file inside is named by its builder (_core.abi3.so, _core.cpython-311-darwin.so, _core.pyd,
_core.cp311-win_amd64.pyd): Python's importer finds it, nothing here spells the name.

Licences of what is inside the binary: vendor/THIRD_PARTY.md and vendor/licenses/.
"""
import os
import platform
import sys

from .const import HERE

MODULE = "im_remesh"
MISSING_ENV = "AI_ASSET_NO_BINDING"      # test hook (set only by tests/): behave as if no build for this system exists
SYSTEM_NAMES = dict(macos_arm64="macOS (Apple Silicon)", macos_x64="macOS (Intel)", linux_x64="Linux (x86-64)",
                    linux_arm64="Linux (ARM)", windows_x64="Windows (x86-64)", windows_arm64="Windows (ARM)")


def python_tag():
    if sys.version_info[:2] == (3, 11):
        return "cp311"
    return "abi3" if sys.version_info[:2] >= (3, 12) else None


def platform_tag():
    system = {"darwin": "macos", "linux": "linux", "win32": "windows"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    machine = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)
    return "%s_%s" % (system, machine)


def find_module_path(name=MODULE):
    """(folder to put on sys.path in the worker, how it was found) or (None, why it is missing)."""
    if os.environ.get(MISSING_ENV):
        return None, "missing"
    tag, system = python_tag(), platform_tag()
    if tag is None:
        return None, "python"
    folder = os.path.join(HERE, "vendor", system, tag)
    if os.path.isfile(os.path.join(folder, name, "__init__.py")):
        return folder, "bundled:%s/%s" % (system, tag)
    return None, "missing"


def unavailable_text():
    """Why Quad-dominant cannot run here, in words for the panel; "" when it can."""
    folder, how = find_module_path()
    if folder is not None:
        return ""
    if how == "python":
        return "Quad-dominant needs Python 3.11 or newer inside Blender. Game mode works."
    system = platform_tag()
    return ("Quad-dominant is not available on %s yet: the quad remesher has no build for this system. "
            "Game mode works." % SYSTEM_NAMES.get(system, system))
