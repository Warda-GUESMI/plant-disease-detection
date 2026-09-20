import os
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(self, raw_dir: str, clean_dir: str):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.stats = {
            'total_raw': 0,
            'invalid_removed': 0,
            'duplicates_removed': 0,
            'total_clean': 0,
            'classes': 0,
        }

    @staticmethod
    def is_valid_image(filepath: Path, min_size: int = 50) -> bool:
        try:
            img = cv2.imread(str(filepath))
            if img is None:
                return False
            h, w = img.shape[:2]
            if h < min_size or w < min_size:
                return False
            mean_val = img.mean()
            if mean_val < 5 or mean_val > 250:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def compute_phash(filepath: Path, hash_size: int = 16) -> str:
        try:
            img = Image.open(filepath).convert('RGB')
            return str(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            return ""

    def remove_duplicates_in_class(self, class_dir: Path, threshold: int = 5):
        hashes = {}
        duplicates = []

        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            h = self.compute_phash(f)
            if not h:
                continue

            is_dup = False
            for existing_hash in hashes:
                dist = imagehash.hex_to_hash(h) - imagehash.hex_to_hash(existing_hash)
                if dist <= threshold:
                    is_dup = True
                    duplicates.append(f)
                    break

            if not is_dup:
                hashes[h] = f

        for dup in duplicates:
            dup.unlink()

        return len(duplicates)

    def clean(self, remove_duplicates: bool = True, min_size: int = 50):
        logger.info("\n" + "=" * 60)
        logger.info("🧹 STEP 1 : DATA CLEANING")
        logger.info("=" * 60)

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Dataset introuvable : {self.raw_dir}")

        class_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        logger.info(f"📂 Classes trouvées : {len(class_dirs)}")

        if self.clean_dir.exists():
            shutil.rmtree(self.clean_dir, ignore_errors=True)

        logger.info("\n🔍 Phase A : Suppression des images invalides...")
        for class_dir in tqdm(class_dirs, desc="Validation"):
            dest_dir = self.clean_dir / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                self.stats['total_raw'] += 1
                if self.is_valid_image(img_file, min_size):
                    shutil.copy2(img_file, dest_dir / img_file.name)
                else:
                    self.stats['invalid_removed'] += 1

        if remove_duplicates and self.clean_dir.exists():
            logger.info("\n🔍 Phase B : Détection des doublons (pHash)...")
            clean_classes = [d for d in self.clean_dir.iterdir() if d.is_dir()]
            for class_dir in tqdm(clean_classes, desc="Déduplication"):
                n_dups = self.remove_duplicates_in_class(class_dir)
                self.stats['duplicates_removed'] += n_dups

        # Nettoyage classes vides
        if not self.clean_dir.exists():
            self.clean_dir.mkdir(parents=True, exist_ok=True)

        empty_classes = []
        for class_dir in self.clean_dir.iterdir():
            if class_dir.is_dir() and len(list(class_dir.glob('*'))) == 0:
                empty_classes.append(class_dir.name)
                shutil.rmtree(class_dir)

        final_classes = [d for d in self.clean_dir.iterdir() if d.is_dir()]
        self.stats['classes'] = len(final_classes)
        self.stats['total_clean'] = sum(len(list(d.glob('*'))) for d in final_classes)

        self._print_report(final_classes)
        return self.stats

    def _print_report(self, class_dirs):
        logger.info("\n" + "=" * 60)
        logger.info("📊 RAPPORT DE NETTOYAGE")
        logger.info("=" * 60)
        logger.info(f"   Images brutes          : {self.stats['total_raw']}")
        logger.info(f"   Images invalides       : -{self.stats['invalid_removed']}")
        logger.info(f"   Doublons               : -{self.stats['duplicates_removed']}")
        logger.info(f"   Images propres         : {self.stats['total_clean']}")
        logger.info(f"   Classes finales        : {self.stats['classes']}")
        logger.info("=" * 60)

        os.makedirs('results', exist_ok=True)
        class_counts = {d.name: len(list(d.glob('*'))) for d in class_dirs}
        df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['count']).sort_values('count', ascending=False)
        df.to_csv('results/cleaning_report.csv')
