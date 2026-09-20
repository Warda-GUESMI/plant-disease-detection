import os
import sys
import shutil
import zipfile
from pathlib import Path

# --- FIX 1 : Force l'encodage UTF-8 pour éviter le crash des caractères Windows ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("==================================================")
print("[1/3] Verification et Preparation du Dataset...")
print("==================================================")

base_dir = Path(__file__).parent.resolve()
os.chdir(base_dir)
target_dir = base_dir / "data" / "PlantVillage"
downloads_zip = Path.home() / "Downloads" / "archive.zip"

# Si data/PlantVillage est vide ou n'existe pas, on extrait archive.zip
images_in_target = list(target_dir.rglob("*.jpg")) + list(target_dir.rglob("*.png")) if target_dir.exists() else []

if len(images_in_target) == 0:
    if not downloads_zip.exists():
        print(f"[!] Fichier introuvable : {downloads_zip}")
        print("[!] Veuillez verifier que 'archive.zip' est bien dans votre dossier Telechargements.")
        sys.exit(1)

    print(f"[+] Archive trouvee : {downloads_zip}")
    print("[+] Extraction en cours (cela peut prendre 1 a 2 minutes)...")

    temp_extract = base_dir / "temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    with zipfile.ZipFile(downloads_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    for root, dirs, files in os.walk(temp_extract):
        jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if jpgs:
            class_name = Path(root).name
            if class_name.lower() in ['train', 'valid', 'test', 'data', 'temp_extract', 'archive']:
                continue
            
            dest_class = target_dir / class_name
            dest_class.mkdir(parents=True, exist_ok=True)
            for img in jpgs:
                src_file = Path(root) / img
                dst_file = dest_class / f"{img_count}_{img}"
                shutil.copy2(src_file, dst_file)
                img_count += 1

    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    print(f"[OK] {img_count} images organisees dans {target_dir}")
else:
    print(f"[OK] Dataset deja pret dans {target_dir} ({len(images_in_target)} images).")

# --- Nettoyage des anciens dossiers temporaires ---
for folder in ["data_clean", "splits"]:
    p = base_dir / folder
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

print("\n==================================================")
print("[2/3] Correction automatique de l'encodage du code...")
print("==================================================")

# Inserer la ligne UTF-8 en haut de main.py pour eviter UnicodeEncodeError
main_file = base_dir / "main.py"
if main_file.exists():
    content = main_file.read_text(encoding='utf-8')
    utf8_header = "import sys\nif sys.platform == 'win32':\n    try:\n        sys.stdout.reconfigure(encoding='utf-8')\n    except Exception:\n        pass\n"
    if "reconfigure(encoding='utf-8')" not in content:
        main_file.write_text(utf8_header + content, encoding='utf-8')
        print("[OK] Encodage UTF-8 ajoute dans main.py")

print("\n==================================================")
print("[3/3] Lancement du Pipeline de Deep Learning...")
print("==================================================")

# Execution du pipeline principal
cmd = f'"{sys.executable}" main.py --step all --epochs 2 --batch_size 8 --workers 0'
os.system(cmd)
