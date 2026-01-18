import subprocess
import sys

with open("socket_output_log.txt", "w", encoding="utf-8") as f:
    result = subprocess.run(
        [sys.executable, "socket_implementation/launcher.py"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace' # Handle encoding errors gracefully
    )
    if result.stdout:
        f.write(result.stdout)
    else:
        f.write("[No Standard Output]\n")
        
    if result.stderr:
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
    else:
        f.write("\n[No Standard Error]\n")
        
    print("Done writing Socket Output Log.")
