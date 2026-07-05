"""Workspace construction for observed-memory generation.

Provides workspace builders that copy only code/ (no knowledge_corpus),
with GT functions stubbed.  Reuses stubbing logic from runner.py.
"""

import os
import shutil
from runner import _stub_gt_functions, _stub_one_function


def build_explore_workspace(code_root: str, gt_locations: dict, tmp_dir: str) -> dict:
    """Copy code/ only (exclude test_code/), stub all GT functions.

    No knowledge_corpus is included.

    Returns:
        {"workspace": abs_path, "code": abs_path}
    """
    ws_dir = os.path.join(tmp_dir, "workspace")
    os.makedirs(ws_dir, exist_ok=True)

    code_dst = os.path.join(ws_dir, "code")

    def _ignore(_dir, contents):
        return {c for c in contents if c in ("test_code", "__pycache__", ".pytest_cache")}

    shutil.copytree(code_root, code_dst, symlinks=True, ignore=_ignore)

    _stub_gt_functions(code_dst, gt_locations)

    return {"workspace": ws_dir, "code": code_dst}


def build_single_stub_workspace(
    code_root: str,
    gt_locations: dict,
    target_file: str,
    target_start: int,
    target_end: int,
    tmp_dir: str,
) -> dict:
    """Copy code/ only, stub only the target function.

    Args:
        target_file: Relative path within code/ (e.g. "recipe/prime/prime_core_algos.py").
        target_start, target_end: 1-indexed line range of the target function.

    Returns:
        {"workspace": abs_path, "code": abs_path}
    """
    ws_dir = os.path.join(tmp_dir, "workspace")
    os.makedirs(ws_dir, exist_ok=True)

    code_dst = os.path.join(ws_dir, "code")

    def _ignore(_dir, contents):
        return {c for c in contents if c in ("test_code", "__pycache__", ".pytest_cache")}

    shutil.copytree(code_root, code_dst, symlinks=True, ignore=_ignore)

    # Stub only the target function
    file_path = os.path.join(code_dst, target_file)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = _stub_one_function(lines, target_start, target_end)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return {"workspace": ws_dir, "code": code_dst}
