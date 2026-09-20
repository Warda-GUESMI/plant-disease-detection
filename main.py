import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import os
import json
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

def run_pipeline(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[DEVICE] Device utilise : {device}")

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    clean_dir = 'data_clean/PlantVillage_clean'
    splits_dir = 'splits'

    # STEP 1 : Clean
    if args.step in ['all', 'clean']:
        from src.step1_cleaning import DataCleaner
        cleaner = DataCleaner(raw_dir=args.data_dir, clean_dir=clean_dir)
        cleaner.clean(remove_duplicates=True)

    # STEP 2 : Split
    if args.step in ['all', 'split']:
        from src.step2_split import DataSplitter
        splitter = DataSplitter(clean_dir=clean_dir, output_dir=splits_dir)
        splitter.split(copy_files=True)

    # STEP 3-4-5 : Train
    if args.step in ['all', 'train']:
        from src.step3_transforms import create_dataloaders
        from src.step4_model import build_model
        from src.step5_training import FineTuner

        train_loader, val_loader, test_loader, class_names, num_classes = create_dataloaders(
            splits_dir=splits_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.workers
        )
        with open('models/class_names.json', 'w') as f:
            json.dump(class_names, f, indent=2)

        model = build_model(num_classes=num_classes, pretrained=True, dropout=args.dropout, device=device)
        tuner = FineTuner(model=model, train_loader=train_loader, val_loader=val_loader, device=device, lr_head=args.lr)
        tuner.fit(total_epochs=args.epochs, phase1_epochs=args.phase1, patience=args.patience)
        tuner.plot_history()

    # STEP 6 : Eval
    if args.step in ['all', 'eval']:
        from src.step3_transforms import create_dataloaders
        from src.step4_model import build_model
        from src.step6_evaluation import ModelEvaluator

        _, _, test_loader, class_names, num_classes = create_dataloaders(splits_dir=splits_dir, img_size=args.img_size, batch_size=args.batch_size)
        model = build_model(num_classes=num_classes, pretrained=False, device=device)
        ckpt = torch.load('models/best_model.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        evaluator = ModelEvaluator(model, test_loader, class_names, device)
        evaluator.full_evaluation()

    # STEP 7 : Predict
    if args.step == 'predict':
        from src.step7_prediction import PlantDiseasePredictor
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        predictor = PlantDiseasePredictor('models/best_model.pth', class_names, device=device)
        if args.image:
            res = predictor.predict(args.image)
            print(f"\n[RESULT] {res['plant']} | {res['disease']} | Confiance: {res['confidence']*100:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='all', choices=['all', 'clean', 'split', 'train', 'eval', 'predict'])
    parser.add_argument('--data_dir', type=str, default='data/PlantVillage')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--phase1', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args)
