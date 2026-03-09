#!/usr/bin/env python3
"""
==========================================================================
  Biometric Few-Shot Verification — Google Colab Training Script
  Trains Siamese & Prototypical Networks on Signatures, Faces, Fingerprints
  
  INSTRUCTIONS:
    1. Upload data_raw.zip to your Google Drive root
    2. Open this script in Google Colab (or paste cells into a notebook)
    3. Set Runtime → Change runtime type → GPU (T4)
    4. Run all cells
    5. Download results.zip when done
==========================================================================
"""

# ── Cell 1: Setup & Dependencies ─────────────────────────────────────────
import subprocess, sys

def install_deps():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'torch', 'torchvision', 'opencv-python-headless', 'albumentations',
        'scikit-learn', 'matplotlib', 'seaborn', 'pyyaml', 'tqdm',
        'tensorboard'])

install_deps()

# ── Cell 2: Mount Google Drive & Extract Data ─────────────────────────────
import os, zipfile, shutil

def setup_data():
    """Extract data_raw.zip and auto-discover dataset paths."""
    zip_path = 'data_raw.zip'
    
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found in current directory!")
        print("Make sure you copied it: !cp /content/drive/MyDrive/data_raw.zip .")
        return {}
    
    os.makedirs('data/raw', exist_ok=True)
    print("Extracting datasets...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('data/raw')
    
    # Print directory tree (depth <= 4)
    print("\nExtracted directory structure:")
    for root, dirs, files in os.walk('data/raw'):
        depth = root.replace('data/raw', '').count(os.sep)
        if depth < 4:
            indent = ' ' * 2 * depth
            print(f'{indent}{os.path.basename(root)}/ ({len(files)} files, {len(dirs)} dirs)')
    
    # Auto-discover dataset paths by searching for marker folders/files
    discovered = {}
    
    for root, dirs, files in os.walk('data/raw'):
        basename = os.path.basename(root)
        # CEDAR: look for full_org folder
        if basename == 'full_org':
            cedar_root = os.path.dirname(root)
            discovered['cedar'] = cedar_root
            print(f"\n[AUTO-DETECT] CEDAR signatures at: {cedar_root}")
        # ATT: look for folder named 's1' inside a directory with many s* folders
        if basename == 's1' and os.path.isdir(root):
            att_root = os.path.dirname(root)
            # Verify it has multiple s* directories
            s_dirs = [d for d in os.listdir(att_root) if d.startswith('s') and os.path.isdir(os.path.join(att_root, d))]
            if len(s_dirs) >= 10:
                discovered['att'] = att_root
                print(f"[AUTO-DETECT] ATT Faces at: {att_root}")
        # SOCOFing: look for Real folder
        if basename == 'Real' and os.path.isdir(root):
            parent = os.path.dirname(root)
            # Verify it also has Altered
            if os.path.isdir(os.path.join(parent, 'Altered')):
                discovered['socofing'] = parent
                print(f"[AUTO-DETECT] SOCOFing at: {parent}")
        # LFW: look for lfw-funneled or lfw as a directory with person-name subdirs
        if basename in ('lfw-funneled', 'lfw', 'lfw_funneled'):
            if os.path.isdir(root) and len(dirs) > 100:
                discovered['lfw'] = root
                print(f"[AUTO-DETECT] LFW at: {root}")
    
    if not discovered:
        print("\n[WARNING] Could not auto-detect any datasets! Check the zip structure.")
    
    return discovered

discovered_paths = setup_data()

# ── Cell 3: Check GPU ─────────────────────────────────────────────────────
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] No GPU detected — training will be slow on CPU!")
print(f"[Device] {device}")

# ── Cell 4: Import All Project Modules ────────────────────────────────────
import argparse, glob, yaml, warnings
warnings.filterwarnings('ignore', category=UserWarning)

from data.dataset_factory import get_dataset
from training.trainer import Trainer

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGS — loaded from YAML (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_DIR = 'configs'
CONFIGS = []
for yaml_path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['_config_path'] = yaml_path
    config['name'] = os.path.splitext(os.path.basename(yaml_path))[0]
    CONFIGS.append(config)

print(f"\n[Configs] Loaded {len(CONFIGS)} experiment configs from {CONFIG_DIR}/")
for c in CONFIGS:
    print(f"  - {c['name']} ({c['model']['type']}/{c['dataset']['modality']})")


def get_dataset_for_config(config):
    """Create dataset with Colab auto-discovered path overrides."""
    # Override root_dir with auto-discovered paths if available
    name = config['dataset']['name']
    if name == 'cedar' and 'cedar' in discovered_paths:
        config['dataset']['root_dir'] = discovered_paths['cedar']
    elif name == 'att' and 'att' in discovered_paths:
        config['dataset']['root_dir'] = discovered_paths['att']
    elif name == 'socofing' and 'socofing' in discovered_paths:
        config['dataset']['root_dir'] = discovered_paths['socofing']
    elif name == 'lfw' and 'lfw' in discovered_paths:
        config['dataset']['root_dir'] = discovered_paths['lfw']

    print(f"  Using root_dir: {config['dataset']['root_dir']}")
    return get_dataset(config, training=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Run All Training
# ═══════════════════════════════════════════════════════════════════════════

def run_all_training(configs):
    print("=" * 70)
    print("  BIOMETRIC FEW-SHOT VERIFICATION — TRAINING SUITE")
    print(f"  Device: {device}")
    print(f"  Experiments: {len(configs)}")
    print("=" * 70)

    results_summary = []

    for i, config in enumerate(configs):
        print(f"\n\n{'#' * 70}")
        print(f"  EXPERIMENT {i+1}/{len(configs)}: {config['name']}")
        print(f"{'#' * 70}\n")

        try:
            dataset = get_dataset_for_config(config)
            trainer = Trainer(config['_config_path'], config_override=config)
            trainer.train(dataset)
            results_summary.append({
                'name': config['name'],
                'status': 'SUCCESS',
                'best_val_loss': trainer.best_val_loss,
                'epochs': trainer.epoch,
            })
        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'name': config['name'],
                'status': f'FAILED: {e}',
            })

    # Print Summary
    print(f"\n\n{'=' * 70}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for r in results_summary:
        status = r['status']
        name = r['name']
        if status == 'SUCCESS':
            print(f"  [OK] {name:40s} | best_val_loss={r['best_val_loss']:.4f} | epochs={r['epochs']}")
        else:
            print(f"  [FAIL] {name:40s} | {status}")

    print("\nZipping best checkpoints and training logs for download...")
    import zipfile
    with zipfile.ZipFile('results_trained.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for config in configs:
            results_dir = config.get('results_dir', 'results')
            best_path = os.path.join(results_dir, 'checkpoints', 'best.pth')
            if os.path.exists(best_path):
                zf.write(best_path)
            # Include CSV training logs
            csv_path = os.path.join(results_dir, 'logs', 'training_log.csv')
            if os.path.exists(csv_path):
                zf.write(csv_path)
            # Include training curve plots
            curves_path = os.path.join(results_dir, 'figures', 'training_curves.png')
            if os.path.exists(curves_path):
                zf.write(curves_path)
    print("Results saved to results_trained.zip")
    print("Download it from the Colab file browser (left sidebar) or run:")
    print("  from google.colab import files; files.download('results_trained.zip')")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train biometric verification models')

    # ── Model / modality filter ──────────────────────────────────────
    parser.add_argument('--modality', type=str, default=None,
                        choices=['signature', 'face', 'fingerprint'],
                        help='Train only this modality (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['siamese', 'prototypical'],
                        help='Train only this model type (default: all)')

    # ── Training hyper-parameter overrides ───────────────────────────
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override early-stopping patience')
    parser.add_argument('--loss', type=str, default=None,
                        choices=['bce', 'contrastive'],
                        help='Override loss function (siamese only)')

    args = parser.parse_args()

    # ── Filter configs ───────────────────────────────────────────────
    selected = CONFIGS
    if args.modality:
        selected = [c for c in selected if c['dataset']['modality'] == args.modality]
    if args.model:
        selected = [c for c in selected if c['model']['type'] == args.model]

    if not selected:
        print(f"No configs matched --modality={args.modality} --model={args.model}")
        sys.exit(1)

    # ── Apply hyper-parameter overrides to each config ───────────────
    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.patience is not None:
        overrides['patience'] = args.patience
    if args.loss is not None:
        overrides['loss'] = args.loss

    if overrides:
        print(f"\n[Overrides] Applying CLI overrides: {overrides}")
        for c in selected:
            for key, val in overrides.items():
                c['training'][key] = val

    print(f"\n[Filter] Training {len(selected)}/{len(CONFIGS)} configs:")
    for c in selected:
        t = c['training']
        print(f"  → {c['name']}  (epochs={t['epochs']}, lr={t['lr']}, "
              f"batch_size={t['batch_size']}, patience={t['patience']})")

    run_all_training(selected)
