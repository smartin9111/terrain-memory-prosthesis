import sys
from pathlib import Path

print("Python:", sys.version)
print("Executable:", sys.executable)

workspace = Path(__file__).resolve().parents[2]
myoassist_dir = workspace / "myoassist"

print("Workspace:", workspace)
print("MyoAssist path:", myoassist_dir)
print("MyoAssist exists:", myoassist_dir.exists())

if not myoassist_dir.exists():
    raise FileNotFoundError(f"MyoAssist not found at {myoassist_dir}")

print("MyoAssist directory OK")
