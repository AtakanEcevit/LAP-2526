"""
Dataset downloader for all supported biometric datasets.
Downloads and extracts datasets that are freely available.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path


# Dataset download URLs
DATASETS = {
    'att_faces': {
        'url': 'https://github.com/doganmeh/att-database-of-faces/archive/refs/heads/master.zip',
        'extract_to': 'data/raw/faces/att_faces',
        'description': 'AT&T/ORL Face Database (40 subjects x 10 images)',
    },
    'lfw': {
        'url': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'extract_to': 'data/raw/faces/lfw',
        'description': 'Labeled Faces in the Wild (13,000+ images)',
    },
    'socofing': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/ruizgara/socofing',
        'extract_to': 'data/raw/fingerprints/SOCOFing',
        'description': 'SOCOFing Fingerprint Dataset (6,000 images)',
        'note': 'May require Kaggle API credentials. See README for manual download.',
    },
    'cedar': {
        'url': None,  # Requires manual download from cedar.buffalo.edu
        'extract_to': 'data/raw/signatures/CEDAR',
        'description': 'CEDAR Signature Database (55 writers × 48 images)',
        'note': 'Manual download required from: https://cedar.buffalo.edu/NIJ/data/',
    },
}


def download_file(url, save_path, description=""):
    """Download a file with progress bar."""
    print(f"\n[Download] {description}")
    print(f"  URL:  {url}")
    print(f"  Save: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 / total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Progress: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)"
            )
        else:
            mb_done = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=progress_hook)
        print(f"\n  [OK] Download complete")
        return True
    except Exception as e:
        print(f"\n  [FAIL] Download failed: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract zip or tar.gz archive."""
    print(f"  Extracting to: {extract_to}")
    os.makedirs(extract_to, exist_ok=True)

    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(extract_to)
        print(f"  [OK] Extraction complete")
        return True
    except Exception as e:
        print(f"  [FAIL] Extraction failed: {e}")
        return False


def download_dataset(name, base_dir="."):
    """Download and extract a single dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    extract_to = os.path.join(base_dir, info['extract_to'])

    # Check if already downloaded
    if os.path.exists(extract_to) and any(os.scandir(extract_to)):
        print(f"\n[{name}] Already exists at {extract_to}, skipping.")
        return True

    if info['url'] is None:
        print(f"\n[{name}] {info['description']}")
        print(f"  [INFO] {info.get('note', 'Manual download required.')}")
        print(f"  Please download and extract to: {extract_to}")
        return False

    # Download
    ext = '.zip' if info['url'].endswith('.zip') else '.tgz'
    archive_path = os.path.join(base_dir, 'data', 'downloads', f'{name}{ext}')

    if not download_file(info['url'], archive_path, info['description']):
        return False

    # Extract
    if not extract_archive(archive_path, extract_to):
        return False

    return True


def download_all(base_dir="."):
    """Download all available datasets."""
    print("=" * 60)
    print("  Biometric Dataset Downloader")
    print("=" * 60)

    results = {}
    for name in DATASETS:
        success = download_dataset(name, base_dir)
        results[name] = success

    print(f"\n" + "=" * 60)
    print(f"  Summary")
    print(f"=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[MANUAL] download needed"
        print(f"  {name:15s}: {status}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download biometric datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=list(DATASETS.keys()) + ['all'],
                        help='Dataset to download (default: all)')
    parser.add_argument('--dir', type=str, default='.',
                        help='Base directory (default: current)')
    args = parser.parse_args()

    if args.dataset == 'all':
        download_all(args.dir)
    else:
        download_dataset(args.dataset, args.dir)
