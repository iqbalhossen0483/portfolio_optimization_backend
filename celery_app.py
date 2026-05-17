"""
Root-level Celery entry point.

Explicitly adds the project root to sys.path so that worker subprocesses
spawned by Celery can always import `app.*` regardless of CWD or how the
worker pool forks new interpreters (required on Windows prefork/solo pools).

Run from the project root:
    celery -A celery_app worker --loglevel=info --pool=solo
"""
import sys
import os

# Anchor project root into sys.path before any app.* imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.workers.tasks import celery_app  # noqa: E402

__all__ = ["celery_app"]
 