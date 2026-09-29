"""Download the two Mordor APT29 host-telemetry zips from the OTRF Security-Datasets repository and unzip them into data/."""
import io, sys, zipfile, urllib.request
from pathlib import Path
BASE = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/compound/apt29/"
FILES = {"day1/apt29_evals_day1_manual.zip": "apt29_evals_day1_manual_2020-05-01225525.json",
         "day2/apt29_evals_day2_manual.zip": "apt29_evals_day2_manual_2020-05-02035409.json"}
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data"); out.mkdir(parents=True, exist_ok=True)
for rel, expect in FILES.items():
    if (out / expect).exists(): print("exists", expect); continue
    print("downloading", rel); data = urllib.request.urlopen(BASE + rel, timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist(); print("  members:", names); z.extractall(out)
print("done ->", out)
