
"""Dataset loader for the Information Theory project."""

from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np


DATA_DIR = Path(__file__).resolve().parent

NAMES_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/makemore/master/names.txt"
)


def load(name):
    """Load the original project datasets."""

    if name == "names":
        path = DATA_DIR / "names.txt"

        # Download the original dataset if it is missing.
        if not path.exists():
            print("Downloading original names dataset...")
            with urlopen(NAMES_URL, timeout=60) as response:
                data = response.read()

            path.write_bytes(data)

        # Read the actual names.
        names = [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(names) != 32033:
            raise ValueError(
                f"Expected 32033 names, found {len(names)}."
            )

        print(f"Loaded {len(names)} original names.")

        return SimpleNamespace(names=names)

    elif name == "breast-cancer":
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer()

    elif name == "sklearn-digits":
        from sklearn.datasets import load_digits
        return load_digits()

    else:
        raise ValueError(f"Unknown dataset: {name}")
