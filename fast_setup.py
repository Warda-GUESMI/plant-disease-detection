import sys
import shutil
from pathlib import Path

# Fix encodage UTF-8 pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("⚡ Déplacement ultra-rapide des images...")

base = Path(__file__).parent.resolve()
temp_data = base / "temp_zip" / "data"
target = base / "data" / "PlantVillage"

# Nettoyer l'ancien dossier s'il existe
if target.exists():
    shutil.rmtree(target, ignore_errors=True)
target.mkdir(parents=True, exist_ok=True)

# Répertoires sources
sources = [temp_data / "train", temp_data / "valid", temp_data / "test"]

classes_count = set()
total_files = 0

for src_dir in sources:
    if not src_dir.exists():
        continue
    for class_dir in src_dir.iterdir():
        if class_dir.is_dir() and ("___" in class_dir.name or "healthy" in class_dir.name.lower()):
            dest_dir = target / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in class_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy2(img, dest_dir / f"{src_dir.name}_{img.name}")
                    total_files += 1
            classes_count.add(class_dir.name)

# Supprimer temp_zip
if (base / "temp_zip").exists():
    shutil.rmtree(base / "temp_zip", ignore_errors=True)

print("=" * 50)
print(f"✅ SUCCÈS ! {len(classes_count)} classes de plantes prêtes dans data/PlantVillage !")
print(f"📸 Total d'images : {total_files}")
print("=" * 50)
