import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_config() -> dict:
    return json.loads((ROOT / "configs" / "experiments" / "p1c1_myoassist.json").read_text())
