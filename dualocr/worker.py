"""Background job queue.

Each job is a folder in data/jobs/<id>/ with a job.json describing it and its
live progress. A single worker process runs the jobs one after another, so the
app window can be closed while documents are still being processed. The worker
quits by itself after a couple of idle minutes (freeing the GPU).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from . import osutil
from .config import DATA, JOBS, Settings

PIDFILE = DATA / "worker.pid"
LOGFILE = DATA / "worker.log"
IDLE_EXIT_SECONDS = 120


# ------------------------------------------------------------ job files

def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: the window is reading the file right now
            if attempt == 19:
                raise
            time.sleep(0.05)


def read_job(job_id: str) -> Optional[dict]:
    p = JOBS / job_id / "job.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_jobs() -> List[dict]:
    out = []
    for d in JOBS.iterdir() if JOBS.exists() else []:
        j = read_job(d.name)
        if j:
            out.append(j)
    out.sort(key=lambda j: j.get("created", 0))
    return out


def submit(inputs: List[str], settings: Settings, out_dir: str = "", out_path: str = "") -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = JOBS / job_id
    d.mkdir(parents=True)
    _write(d / "job.json", {
        "id": job_id, "inputs": [str(Path(p).resolve()) for p in inputs], "settings": asdict(settings),
        "out_dir": out_dir, "out_path": out_path, "created": time.time(), "status": "queued",
        "progress": {"stage": "queued", "page": 0, "pages": 0, "fraction": 0.0, "detail": "Waiting to start"},
    })
    ensure_worker()
    return job_id


def cancel(job_id: str) -> None:
    (JOBS / job_id / "cancel").touch()
    j = read_job(job_id)
    if j and j["status"] == "queued":
        j["status"] = "cancelled"
        j["progress"]["detail"] = "Cancelled"
        _write(JOBS / job_id / "job.json", j)


def remove(job_id: str) -> None:
    import shutil
    j = read_job(job_id)
    if j and j["status"] in ("queued", "running"):
        return
    shutil.rmtree(JOBS / job_id, ignore_errors=True)


# ------------------------------------------------------------ worker lifecycle

def worker_alive() -> bool:
    try:
        return osutil.pid_alive(int(PIDFILE.read_text()))
    except Exception:
        return False


def ensure_worker() -> None:
    if worker_alive():
        return
    from .config import FROZEN
    log = open(LOGFILE, "a", encoding="utf-8")
    if FROZEN:   # the app's own executable doubles as the worker
        cmd, cwd = [sys.executable, "--worker"], str(Path.home())
    else:
        cmd, cwd = [sys.executable, "-m", "dualocr.worker"], str(Path(__file__).resolve().parent.parent)
    subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     **osutil.detached_popen_kwargs())
    for _ in range(50):
        if worker_alive():
            return
        time.sleep(0.1)


def notify(title: str, text: str) -> None:
    osutil.notify(title, text)


def _run_job(job_id: str) -> None:
    from . import pipeline

    d = JOBS / job_id
    j = read_job(job_id)
    j["status"] = "running"
    j["started"] = time.time()
    _write(d / "job.json", j)
    settings = Settings(**{k: v for k, v in j["settings"].items() if k in Settings.__dataclass_fields__})
    last = [0.0]

    def report(p: "pipeline.Progress"):
        now = time.time()
        if now - last[0] < 0.4 and p.fraction < 1.0:
            return
        last[0] = now
        el = now - j["started"]
        eta = el / p.fraction * (1 - p.fraction) if p.fraction > 0.03 else None
        j["progress"] = {"stage": p.stage, "page": p.page, "pages": p.pages, "fraction": round(p.fraction, 4),
                         "detail": p.detail, "elapsed": round(el), "eta": round(eta) if eta else None}
        _write(d / "job.json", j)

    def cancelled():
        return (d / "cancel").exists()

    try:
        res = pipeline.process(j["inputs"], settings, out_dir=j.get("out_dir") or None, report=report,
                               cancelled=cancelled, out_path=j.get("out_path") or None)
        j.update(status="done", result=res, finished=time.time())
        j["progress"].update(fraction=1.0, detail="Done", stage="done")
        _write(d / "job.json", j)
        st = res.get("stats", {})
        notify("DualOCR finished", f"{Path(j['inputs'][0]).name}: {res['pages']} pages, "
                                   f"{st.get('uncertain', 0)} words flagged for review")
        if settings.open_when_done:
            osutil.open_path(res["outputs"]["docx"])
    except pipeline.Cancelled:
        j.update(status="cancelled", finished=time.time())
        j["progress"]["detail"] = "Cancelled"
        _write(d / "job.json", j)
    except Exception as e:
        logging.exception("job failed")
        j.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                 finished=time.time())
        j["progress"]["detail"] = f"Failed: {e}"
        _write(d / "job.json", j)
        notify("DualOCR failed", f"{Path(j['inputs'][0]).name}: {e}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if worker_alive():
        return
    PIDFILE.write_text(str(os.getpid()))
    # jobs left 'running' by a crashed worker go back to the queue
    for j in list_jobs():
        if j["status"] == "running":
            j["status"] = "queued"
            _write(JOBS / j["id"] / "job.json", j)
    idle_since = time.time()
    try:
        while True:
            queued = [j for j in list_jobs() if j["status"] == "queued" and not (JOBS / j["id"] / "cancel").exists()]
            if queued:
                _run_job(queued[0]["id"])
                idle_since = time.time()
                continue
            if time.time() - idle_since > IDLE_EXIT_SECONDS:
                break
            time.sleep(1.0)
    finally:
        try:
            from .engines import surya
            surya.shutdown()
        except Exception:
            pass
        try:
            if PIDFILE.read_text() == str(os.getpid()):
                PIDFILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
