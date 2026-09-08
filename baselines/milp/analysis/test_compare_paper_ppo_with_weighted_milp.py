import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_paper_ppo_with_weighted_milp.py")
SPEC = importlib.util.spec_from_file_location("paper_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    old = np.array([[100, 100], [50, 0]], dtype=np.int64)
    new = np.array([[75, 50], [25, 0]], dtype=np.int64)
    moved = old - new
    fitted = MODULE.source_metrics(old, new, moved)
    assert abs(fitted["source_structure_loss_normalized"] - 0.0625) < 1e-12
    supplied = MODULE.source_metrics(old, new, moved, np.array([0.25, 0.5]))
    assert abs(supplied["source_structure_loss_normalized"] - 0.0625) < 1e-12
    assert fitted["sources_fully_depleted"] == 0
    assert fitted["sources_remaining_below_10pct"] == 0
    print("paper comparison metric test: PASS")


if __name__ == "__main__":
    main()
