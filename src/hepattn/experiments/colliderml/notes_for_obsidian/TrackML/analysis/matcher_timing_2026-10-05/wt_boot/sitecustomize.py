# Point this interpreter (and its spawned children, via PYTHONPATH) at the matcher worktree's hepattn
# instead of the main checkout's editable install. lap1015 etc. still come from the environment.
import importlib.machinery
import sys

_WT = ["/shared/projects/hepattn-matcher/src"]


class _WorktreeFinder:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "hepattn":
            return importlib.machinery.PathFinder.find_spec(name, _WT)
        if name.startswith("hepattn."):
            return importlib.machinery.PathFinder.find_spec(name, path)
        return None


sys.meta_path.insert(0, _WorktreeFinder)
