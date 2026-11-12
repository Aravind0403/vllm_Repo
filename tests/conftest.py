import os
import sys
from pathlib import Path

# Tests never talk to GitHub; the server tests analyze the committed cache.
os.environ.setdefault("PR_RADAR_OFFLINE", "1")

# Make `from helpers import ...` work regardless of pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
