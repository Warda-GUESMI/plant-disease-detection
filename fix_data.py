import os
import shutil
import zipfile
from pathlib import Path

# Chemins
base_dir = Path(__file__).parent
target_dir = base_dir / "data" / "PlantVillage"
downloads_dir = Path.home() / "Downloads"
desktop_dir = Path.home() / "OneDrive" / "Bureau"

print("=" * 50)
print("Recherche des images sur votre ordinateur...")
print("=" * 50)

# 1. Nettoyer le dossier cible
if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

# 2. Chercher les fichiers ZIP dans Telechargements ou Bureau
search_dirs = []
if downloads_dir.exists():
    search_dirs.append(downloads_dir)
if desktop_dir.exists():
    search_dirs.append(desktop_dir)

# Chercher aussi dans le Bureau sans OneDrive
alt_desktop = Path.home() / "Desktop"
if alt_desktop.exists() and alt_desktop not in search_dirs:
    search_dirs.append(alt_desktop)

print(f"Dossiers scannes : {[str(d) for d in search_dirs]}")

zip_files = []
for d in search_dirs:
    zip_files.extend(list(d.glob("*.zip")))

print(f"Fichiers ZIP trouves : {len(zip_files)}")
for z in zip_files:
    print(f"  - {z.name} ({z.stat().st_size / 1024 / 1024:.1f} MB)")

archive_zip = None
for z in zip_files:
    if "archive" in z.name.lower() or "plant" in z.name.lower() or "dataset" in z.name.lower():
        archive_zip = z
        break

if archive_zip:
    print(f"\nArchive trouvee : {archive_zip}")
    extract_temp = base_dir / "temp_extract"
    if extract_temp.exists():
        shutil.rmtree(extract_temp)
    print("Extraction en cours (patientez quelques secondes)...")
    with zipfile.ZipFile(archive_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_temp)

    # Afficher la structure extraite
    print("\nStructure extraite :")
    for root, dirs, files in os.walk(extract_temp):
        level = len(Path(root).relative_to(extract_temp).parts)
        if level <= 3:
            indent = "  " * level
            print(f"{indent}{Path(root).name}/ ({len(files)} fichiers, {len(dirs)} sous-dossiers)")

    # Copier les classes
    found_count = 0
    for root, dirs, files in os.walk(extract_temp):
        # Si on trouve un dossier qui contient des images
        jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(jpgs) > 0:
            folder_name = Path(root).name
            # Ignorer les dossiers 'train', 'valid', 'test' eux-memes
            if folder_name.lower() in ['train', 'valid', 'test', 'data', 'temp_extract', 'plantvillage']:
                continue

            dest = target_dir / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in jpgs:
                src_file = Path(root) / img
                dst_file = dest / f"{found_count}_{img}"
                shutil.copy2(src_file, dst_file)
            found_count += 1

    # Supprimer le dossier temporaire
    if extract_temp.exists():
        shutil.rmtree(extract_temp)
    print(f"\nDossiers de classes copies : {found_count}")

else:
    print("\nAucune archive ZIP trouvee automatiquement.")
    print("Verifiez que vous avez telecharge le dataset depuis Kaggle.")

# Verification finale
classes = [d for d in target_dir.iterdir() if d.is_dir() and len(list(d.glob('*'))) > 0]

print("\n" + "=" * 50)
if len(classes) > 0:
    total_imgs = sum(len(list(d.glob('*'))) for d in classes)
    print(f"SUCCES ! {len(classes)} classes d'images ont ete placees dans data/PlantVillage !")
    print(f"Total d'images trouvees : {total_imgs}")
    print("=" * 50)
    print(f"\nVous pouvez maintenant lancer :")
    print(f"  python main.py --step all --epochs 2 --batch_size 8 --workers 0")
else:
    print("ECHEC : Aucune image trouvee automatiquement.")
    print("Telechargez le dataset depuis Kaggle et placez les dossiers dans :")
    print(f"  {target_dir.absolute()}")
print("=" * 50)
