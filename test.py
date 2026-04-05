from pathlib import Path
import sys

# Ensure local src package is importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from human_resource.main import run


if __name__ == "__main__":
    run()
