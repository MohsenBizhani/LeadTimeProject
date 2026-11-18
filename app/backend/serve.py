"""Windowless server launcher used by run_platform.bat.

Started with pythonw.exe so no console window appears. Because pythonw has no
console, standard shell redirection does not reach it; this script therefore
opens logs/backend.log itself and points stdout/stderr at it, then serves the
FastAPI app. The process stays detached after the launcher window is closed.
"""
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

LOG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "backend.log")

_log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
sys.stdout = _log
sys.stderr = _log

print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] starting uvicorn on http://127.0.0.1:8001")

import uvicorn  # noqa: E402

uvicorn.run("main:app", host="127.0.0.1", port=8001, log_level="info")
