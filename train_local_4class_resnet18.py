from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    import train_local_shiptype_cv as base_trainer

    forwarded_args = sys.argv[1:]
    sys.argv = [
        sys.argv[0],
        "--class-scheme",
        "4",
        "--arch",
        "resnet18",
        "--split-mode",
        "patch",
        *forwarded_args,
    ]
    base_trainer.main()


if __name__ == "__main__":
    main()
