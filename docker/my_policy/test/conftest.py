import sys
from pathlib import Path

# Make the entry-point scripts in ``docker/my_policy/scripts`` importable in
# tests (e.g. ``from train_gazebo_residual_act import resolve_init_checkpoint``)
# without forcing them into the installed ``my_policy`` package.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
