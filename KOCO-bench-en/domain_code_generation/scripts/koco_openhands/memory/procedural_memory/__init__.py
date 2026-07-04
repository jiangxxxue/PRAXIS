import sys
from pathlib import Path

KOCO_OPENHANDS_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = KOCO_OPENHANDS_DIR.parent
for path in (SCRIPTS_DIR, KOCO_OPENHANDS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .config import KOCO_OPENHANDS_DIR  # noqa: E402,F401

# runner.py, cli.py, agent/sdk.py live under koco_openhands/. Add to sys.path
# once here so all procedural_memory modules can import them directly.
if str(KOCO_OPENHANDS_DIR) not in sys.path:
    sys.path.insert(0, str(KOCO_OPENHANDS_DIR))
