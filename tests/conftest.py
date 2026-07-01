import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def import_from_project(project_name, module_name):
    project_path = str(REPO_ROOT / project_name)
    for loaded_name in [
        module_name,
        "util",
        "game",
        "learningAgents",
        "featureExtractors",
        "mdp",
    ]:
        sys.modules.pop(loaded_name, None)
    sys.path.insert(0, project_path)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(project_path)
