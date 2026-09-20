import os
import shutil
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, clean_dir: str, output_dir: str, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        self.clean_dir = Path(clean_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    def split(self, copy_files: bool = True):
        logger.info("\n" + "=" * 60)
        logger.info("✂️  STEP 2 : SPLIT STRATIFIÉ 70 / 15 / 15")
        logger.info("=" * 60)

        for split_name in ['train', 'val', 'test']:
            (self.output_dir / split_name).mkdir(parents=True, exist_ok=True)

        class_dirs = sorted([d for d in self.clean_dir.iterdir() if d.is_dir()])
        total_train, total_val, total_test = 0, 0, 0

        for class_dir in tqdm(class_dirs, desc="Split par classe"):
            class_name = class_dir.name
            images = sorted([str(f) for f in class_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])

            if len(images) < 3:
                continue

            if len(images) < 5:
                n_total = len(images)
                train_imgs = images[:max(1, n_total - 2)]
                rem = images[max(1, n_total - 2):]
                val_imgs = rem[:1]
                test_imgs = rem[1:] if len(rem) > 1 else rem[:1]
            else:
                val_test_ratio = self.val_ratio + self.test_ratio
                train_imgs, val_test_imgs = train_test_split(images, test_size=val_test_ratio, random_state=self.seed)
                relative_test = self.test_ratio / val_test_ratio
                if len(val_test_imgs) < 2:
                    val_imgs = val_test_imgs
                    test_imgs = val_test_imgs
                else:
                    val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=relative_test, random_state=self.seed)

            for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                dest_class = self.output_dir / split_name / class_name
                dest_class.mkdir(parents=True, exist_ok=True)
                for img_path in split_imgs:
                    dst = dest_class / Path(img_path).name
                    if copy_files:
                        shutil.copy2(img_path, dst)
                    elif not dst.exists():
                        dst.symlink_to(Path(img_path).resolve())

            total_train += len(train_imgs)
            total_val += len(val_imgs)
            total_test += len(test_imgs)

        total = total_train + total_val + total_test
        logger.info(f"   Train : {total_train:6d} ({total_train/total*100:5.1f}%)")
        logger.info(f"   Val   : {total_val:6d} ({total_val/total*100:5.1f}%)")
        logger.info(f"   Test  : {total_test:6d} ({total_test/total*100:5.1f}%)")
        return {'train': total_train, 'val': total_val, 'test': total_test}
