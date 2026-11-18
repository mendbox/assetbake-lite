# SPDX-License-Identifier: GPL-3.0-or-later
"""Background worker: `blender -b --factory-startup --python worker.py -- <job.json>`.

Does the whole job in its own Blender process, so the user's session stays responsive and its scene is not
touched. Talks to the add-on with one JSON object per line on stdout; every such line starts with
pipeline.PROTOCOL_PREFIX. Events: hello, progress (fraction, text), done (report), error (msg, user).

A watchdog thread ends the worker when the Blender that started it is gone, so no orphan keeps baking.
"""
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.perf_counter()
PREFIX = ""                # set in main() from the pipeline package
MARKER = None              # a file that says why the worker gave up (for tests and bug reports)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
except (AttributeError, ValueError):
    pass


def load_pipeline():
    """Import the pipeline folder on its own, under a private name (the parent package is not needed here)."""
    name = "ai_asset_pipeline"
    folder = os.path.join(HERE, "pipeline")
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "__init__.py"),
                                                  submodule_search_locations=[folder])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, importlib.import_module(name + ".run")


def die(reason):
    """The parent is gone: leave at once and take our own children along. Never returns."""
    try:
        with open(MARKER, "w", encoding="utf-8") as fh:
            fh.write("{}; worker {} exits\n".format(reason, os.getpid()))
    except (OSError, TypeError):
        pass
    if os.name != "nt":
        import signal
        try:
            if os.getpgid(0) == os.getpid():               # we lead our own group (the add-on starts us so)
                os.killpg(os.getpid(), signal.SIGKILL)
        except OSError:
            pass
    os._exit(3)


def emit(event, **fields):
    fields["ev"] = event
    try:
        sys.stdout.write(PREFIX + json.dumps(fields, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):                           # the reading end closed: the parent is gone
        die("stdout pipe closed")


def _windows_parent_alive(pid, _handle=[]):
    """Windows branch (written from the documentation, not run on Windows yet): wait on the parent's handle."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    if not _handle:
        # a HANDLE is pointer-sized: the default int return type would cut it on 64-bit Windows
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        _handle.append(kernel.OpenProcess(0x00100000, False, pid))         # SYNCHRONIZE
    if not _handle[0]:
        return True                                                         # cannot tell: do not kill a good job
    return kernel.WaitForSingleObject(_handle[0], 0) == 0x00000102         # WAIT_TIMEOUT = still running


def parent_alive(pid):
    if os.name == "nt":
        return _windows_parent_alive(pid)
    if os.getppid() != pid:                                 # re-parented to launchd / init
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def watchdog(pid):
    while True:
        time.sleep(0.5)
        if not parent_alive(pid):
            die("watchdog: parent {} gone".format(pid))


def main():
    global PREFIX, MARKER
    job_file = sys.argv[sys.argv.index("--") + 1]
    with open(job_file, encoding="utf-8") as fh:
        job = json.load(fh)
    MARKER = os.path.join(os.path.dirname(job_file), "watchdog_fired.txt")
    threading.Thread(target=watchdog, daemon=True, args=(job["parent_pid"],)).start()
    package, run = load_pipeline()
    PREFIX = package.PROTOCOL_PREFIX

    import bpy
    emit("hello", pid=os.getpid(), blender=bpy.app.version_string, python=list(sys.version_info[:3]),
         job_file=job_file)
    # test hook (environment variable, set only by tests/): stretch the job so that Cancel and the watchdog
    # have something to interrupt
    end = time.perf_counter() + float(os.environ.get("AI_ASSET_DEBUG_SLEEP") or 0.0)
    while time.perf_counter() < end:
        time.sleep(0.2)
        emit("progress", fraction=0.0, text="waiting (test)")
    try:
        result = run.run(job, emit)
    except run.UserError as ex:
        emit("error", msg=str(ex), user=True)
        sys.stdout.flush()
        os._exit(2)
    emit("done", report=result, seconds=round(time.perf_counter() - T0, 1))


try:
    main()
except SystemExit:
    raise
except BaseException as ex:                                 # noqa: a Python error must reach the add-on as an event
    traceback.print_exc()
    emit("error", msg="{}: {}".format(type(ex).__name__, ex), user=False)
    sys.stdout.flush()
    os._exit(1)            # a Python error in --python does not change Blender's exit code by itself
