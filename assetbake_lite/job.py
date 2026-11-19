# SPDX-License-Identifier: GPL-3.0-or-later
"""The background job, seen from the user's Blender session.

One worker process per job (worker.py in a second Blender, `-b --factory-startup`). A reader thread only moves
stdout lines into a queue; everything else happens on the main thread in poll(), which a bpy.app.timers callback
calls ten times a second (tests call it directly: background mode has no event loop).

Cancel kills the worker's whole process group. The worker also ends itself when this Blender dies.
The job folder (job.json, source.blend, result.blend, logs) is removed when the job ends, whatever the outcome.
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import bpy

from . import const
from .pipeline import PROTOCOL_PREFIX

WORKER = os.path.join(const.HERE, "worker.py")


def temp_base():
    """Folder for job folders: Blender's own session folder, which Blender removes when it quits."""
    return os.environ.get(const.TMP_ENV) or bpy.app.tempdir or tempfile.gettempdir()


def log_path():
    """Where the log of the last failed job is kept for the user to send (outside Blender's session folder,
    which is deleted when Blender quits)."""
    return os.path.join(tempfile.gettempdir(), const.ADDON_ID + "_last_error.log")


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)     # the worker is a session leader: pgid == pid
        except ProcessLookupError:
            pass


class Job:
    """States: new -> running -> done | failed | cancelled.
    ops.py adds "ready" between running and done: the result is finished but waits for Object Mode."""

    def __init__(self):
        self.state = "new"
        self.fraction = 0.0
        self.text = "Starting"
        self.error = ""
        self.user_error = False
        self.report = None
        self.events = []
        self.proc = None
        self.dir = None
        self.t_start = self.t_end = None
        self.context = {}                   # what finish() needs: names of the scene, sources, collection
        self._q = queue.Queue()
        self._log = None

    # ------------------------------------------------------------------ files
    def make_dir(self):
        base = temp_base()
        os.makedirs(base, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="assetbake_", dir=base)
        self.source_blend = os.path.join(self.dir, "source.blend")
        self.result_blend = os.path.join(self.dir, "result.blend")
        self.job_file = os.path.join(self.dir, "job.json")
        return self.dir

    def keep_log(self, headline):
        """Copy what the worker wrote to stderr to log_path(). Returns the path, or "" when that failed."""
        try:
            with open(log_path(), "w", encoding="utf-8") as fh:
                fh.write("{} {}\nBlender {}\n{}\n\n".format(const.ADDON_NAME, const.VERSION, bpy.app.version_string,
                                                           headline))
                fh.write(self.stderr_tail(200000))
            return log_path()
        except OSError:
            return ""

    def cleanup(self):
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.dir and os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)

    def stderr_tail(self, size=1500):
        try:
            with open(os.path.join(self.dir, "worker_stderr.log"), encoding="utf-8", errors="replace") as fh:
                return fh.read()[-size:]
        except (OSError, TypeError):
            return ""

    # ------------------------------------------------------------------ process
    def start(self, job):
        job = dict(job, parent_pid=os.getpid(), source_blend=self.source_blend, result_blend=self.result_blend,
                   work_dir=self.dir)
        with open(self.job_file, "w", encoding="utf-8") as fh:
            json.dump(job, fh, ensure_ascii=False)
        cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python", WORKER, "--", self.job_file]
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True          # own process group, so Cancel can kill the whole tree
        self._log = open(os.path.join(self.dir, "worker_stderr.log"), "wb")
        self.t_start = time.perf_counter()
        # raises OSError when the process cannot be started; the caller cleans up and tells the user
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self._log, **kw)
        threading.Thread(target=self._reader, daemon=True).start()
        self.state = "running"

    def _reader(self):
        # bytes are decoded here, so a bad byte can never raise in the timer
        for raw in self.proc.stdout:
            self._q.put(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        self._q.put(None)

    def poll(self):
        """Never blocks. Returns True while the worker is still running. When it returns False for the first
        time the state is done, failed or cancelled; the caller then appends the result and calls cleanup()."""
        eof = False
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                eof = True
            elif PROTOCOL_PREFIX in line:
                # find, not startswith: C-level output without a newline can sit in front of our line
                try:
                    ev = json.loads(line[line.index(PROTOCOL_PREFIX) + len(PROTOCOL_PREFIX):])
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                self.events.append(ev.get("ev"))
                if ev.get("ev") == "progress":
                    self.fraction = float(ev.get("fraction", self.fraction))
                    self.text = ev.get("text", self.text)
                elif ev.get("ev") == "done":
                    self.report = ev.get("report")
                elif ev.get("ev") == "error":
                    self.error = ev.get("msg", "")
                    self.user_error = bool(ev.get("user"))
        if self.state != "running":
            return False
        if not eof:
            return True                 # also when the process has ended: the pipe is drained on the next tick
        self.proc.wait()
        self.t_end = time.perf_counter()
        # "done" was received and the result file is there: that is success, whatever the exit code says
        # (a background Blender can crash in its own shutdown after the work is finished)
        if self.report is not None and os.path.isfile(self.result_blend):
            self.state = "done"
            self.fraction = 1.0
        else:
            self.state = "failed"
            self.error = self.error or "the worker stopped with exit code {}".format(self.proc.returncode)
        return False

    def cancel(self):
        if self.state not in ("running", "ready"):
            return
        if self.proc is not None:
            kill_tree(self.proc)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.t_end = time.perf_counter()
        self.state = "cancelled"
        self.cleanup()

    @property
    def seconds(self):
        end = self.t_end if self.t_end is not None else time.perf_counter()
        return end - self.t_start if self.t_start is not None else 0.0
