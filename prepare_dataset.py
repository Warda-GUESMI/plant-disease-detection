import os
import shutil
from pathlib import Path

# Chemins
project_dir = Path(__file__).parent
target_dir = project_dir / "data" / "PlantVillage"

# Emplacements possibles du téléchargement Kaggle
user_home = Path.home()
possible_sources = [
    user_home / "Downloads" / "archive" / "data",
    user_home / "Downloads" / "archive",
    user_home / "Downloads" / "data",
    user_home / "Downloads",
    Path("C:/Users/Warda Guesmi/Downloads/archive/data"),
    Path("C:/Users/Warda Guesmi/Downloads/archive"),
]

print("[*] Recherche des images dans vos Telechargements...")

source_found = None
for src in possible_sources:
    if src.exists() and src.is_dir():
        try:
            # Pour éviter de bloquer sur le dossier Downloads complet
            if src == user_home / "Downloads":
                # Vérifier si des sous-dossiers spécifiques existent
                for sub in ["color", "segmented", "PlantVillage", "data", "archive"]:
                    candidate = src / sub
                    if candidate.exists() and candidate.is_dir():
                        source_found = candidate
                        break
                if source_found:
                    break
                continue
            
            subdirs = [d.name for d in src.iterdir() if d.is_dir()]
            if any(s in ['train', 'valid', 'test'] for s in subdirs) or any('___' in s for s in subdirs):
                source_found = src
                break
        except Exception:
            continue

if not source_found:
    print("[!] Impossible de trouver le dossier telecharge automatiquement.")
    print("Veuillez verifier que vous avez bien telecharge et dezippe le dataset PlantVillage dans vos Telechargements.")
    exit(1)

print(f"[+] Source trouvee : {source_found}")

# Nettoyer et récréer data/PlantVillage
if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

# Copier les fichiers
count_imgs = 0
count_classes = set()

for root, dirs, files in os.walk(source_found):
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if images:
        class_name = Path(root).name
        # Si le dossier parent est train/valid/test, garder le nom de la classe
        if class_name in ['train', 'valid', 'test', 'data', 'archive']:
            continue
        
        dest_class_dir = target_dir / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        count_classes.add(class_name)

        for img in images:
            src_file = Path(root) / img
            # Donner un nom unique pour éviter d'écraser
            dst_file = dest_class_dir / f"{count_imgs}_{img}"
            shutil.copy2(src_file, dst_file)
            count_imgs += 1

print("\n" + "=" * 50)
print(f"[OK] SUCCES ! {len(count_classes)} classes copiees dans data/PlantVillage")
print(f"[OK] Total d'images copiees : {count_imgs}")
print("=" * 50)