"""
Builds data/LANNA/ from the official LANNA archive, in the layout this code expects:

    data/LANNA/TD/H<k>/01SAMOHL/*.wav, *.lbl      (typically developing children)
    data/LANNA/SLI/P<k>/01SAMOHL/*.wav, *.lbl     (children with SLI)

The LANNA corpus (Grill and Tučková, 2016; CC BY 4.0) is downloaded from Figshare:
    https://doi.org/10.6084/m9.figshare.2360626   ->   Lanna_speech_database_children.zip
Only the sustained vowels (folder 01SAMOHL) of the 97 children used in the paper are copied (43 TD and 54 SLI, the
same children as Sharma and Singh, 2022). Folder names follow data_mapping.csv: this code numbers the children
H1..H44 and P1..P54, the archive uses Healthy/H26.. and Patients/P..; the files are byte-identical.

Usage:
    python prepare_data.py /path/to/Lanna_speech_database_children.zip      # or the extracted folder
"""
import csv
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "LANNA"


def main(src):
    src = Path(src)
    mapping = list(csv.DictReader(open(HERE / "data_mapping.csv")))
    wanted = {r["folder_in_official_zip"]: r["folder_in_this_code"] for r in mapping}
    n = 0
    if src.suffix == ".zip":
        Z = zipfile.ZipFile(src)
        names = [x for x in Z.namelist() if "__MACOSX" not in x and "/01SAMOHL/" in x and not x.endswith("/")]
        read = lambda x: Z.read(x)
    else:
        names = [str(p) for p in src.rglob("01SAMOHL/*") if p.is_file() and "__MACOSX" not in str(p)]
        read = lambda x: open(x, "rb").read()
    for x in names:
        parts = x.replace("\\", "/").split("/")
        key = f"{parts[-4]}/{parts[-3]}"                  # e.g. Healthy/H26
        if key not in wanted:
            continue
        dst = OUT / wanted[key] / "01SAMOHL" / parts[-1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(read(x)); n += 1
    found = {p.parent.parent.relative_to(OUT).as_posix() for p in OUT.rglob("01SAMOHL/*")}
    missing = sorted(set(wanted.values()) - found)
    print(f"{n} files copied to {OUT}; {len(found)} of {len(wanted)} children" + (f"; MISSING: {missing}" if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
