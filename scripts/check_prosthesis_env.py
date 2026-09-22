from pathlib import Path
from myoassist_utils.env_spec import EnvSpec

workspace = Path(__file__).resolve().parents[2]
spec_path = workspace / "myoassist" / "docs" / "examples" / "env_prosthesis_rough.json"

print("Loading:", spec_path)

spec = EnvSpec.load(spec_path)
spec.validate()

print("Environment valid")
print("MSK:", spec.msk)
print("Device:", spec.device)
print("Terrain:", spec.terrain)
