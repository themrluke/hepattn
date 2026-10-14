"""Make `import hepattn` resolve to the worktree in HEPATTN_SRC instead of the editable install
(whose finder ignores PYTHONPATH), so running jobs keep using the main checkout."""
import importlib.machinery
import os
import sys

_SRC = os.environ.get("HEPATTN_SRC")


class _WorktreeFinder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == "hepattn" or name.startswith("hepattn."):
            return importlib.machinery.PathFinder.find_spec(name, path if path is not None else [_SRC])
        return None


if _SRC:
    sys.meta_path.insert(0, _WorktreeFinder)
