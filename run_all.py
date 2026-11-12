"""Prepare le dataset depuis ~/Downloads/archive.zip puis lance le pipeline.

Usage :
    python run_all.py                      # test rapide : 2 epoques, batch 8
    python run_all.py --epochs 30 ...      # tous les arguments sont transmis a main.py
"""
import os
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("==================================================")
print("[1/2] Verification et Preparation du Dataset...")
print("==================================================")

base_dir = Path(__file__).parent.resolve()
os.chdir(base_dir)
target_dir = base_dir / "data" / "PlantVillage"
downloads_zip = Path.home() / "Downloads" / "archive.zip"

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
images_in_target = [p for p in target_dir.rglob("*") if p.suffix.lower() in IMG_EXT] if target_dir.exists() else []

if len(images_in_target) == 0:
    if not downloads_zip.exists():
        print(f"[!] Fichier introuvable : {downloads_zip}")
        print("[!] Placez 'archive.zip' dans votre dossier Telechargements, ou remplissez data/PlantVillage a la main.")
        sys.exit(1)

    print(f"[+] Archive trouvee : {downloads_zip}")
    print("[+] Extraction en cours (cela peut prendre 1 a 2 minutes)...")

    temp_extract = base_dir / "temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    with zipfile.ZipFile(downloads_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)

    # data/PlantVillage est vide a ce stade (verifie plus haut) : on peut le recreer sans risque
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    for root, dirs, files in os.walk(temp_extract):
        images = [f for f in files if f.lower().endswith(IMG_EXT)]
        if images:
            class_name = Path(root).name
            if class_name.lower() in ['train', 'valid', 'test', 'data', 'temp_extract', 'archive']:
                continue

            dest_class = target_dir / class_name
            dest_class.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(Path(root) / img, dest_class / f"{img_count}_{img}")
                img_count += 1

    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"[OK] {img_count} images organisees dans {target_dir}")
else:
    print(f"[OK] Dataset deja pret dans {target_dir} ({len(images_in_target)} images).")

# Anciens dossiers derives : regeneres par main.py
for folder in ["data_clean", "splits"]:
    p = base_dir / folder
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

print("\n==================================================")
print("[2/2] Lancement du Pipeline de Deep Learning...")
print("==================================================")

extra_args = sys.argv[1:] or ["--epochs", "2", "--batch_size", "8", "--workers", "0"]
cmd = [sys.executable, "main.py", "--step", "all", *extra_args]
sys.exit(subprocess.call(cmd))
