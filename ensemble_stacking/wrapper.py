import subprocess
import sys

with open("clean_output.txt", "w", encoding="utf-8") as f:
    result = subprocess.run(
        [sys.executable, "centralized_benchmark_stacking.py"],
        capture_output=True,
        text=True,
        encoding='utf-8' # Force utf-8 reading of subprocess
    )
    f.write(result.stdout)
    f.write(result.stderr)
    print("Done writing Clean Output.")
