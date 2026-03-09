"""
Tests for trainer enhancements: CSV logging, TensorBoard, training curves,
mixed-precision (AMP), and gradient accumulation.

These tests run WITHOUT trained checkpoints and use synthetic data (like
test_dataloader_training.py uses DummyDataset).
"""

import os
import sys
import csv
import tempfile
import shutil

import pytest
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer


# ── Helpers ────────────────────────────────────────────────────────────


def _make_config(tmpdir, model_type='siamese', extra_training=None):
    """Create a minimal YAML config in tmpdir and return the path."""
    config = {
        'model': {
            'type': model_type,
            'backbone': 'light',
            'embedding_dim': 32,
            'pretrained': False,
            'in_channels': 1,
        },
        'dataset': {
            'modality': 'signature',
            'name': 'cedar',
            'root_dir': 'data/raw/signatures/CEDAR',
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'lr': 0.001,
            'weight_decay': 0.00001,
            'loss': 'bce',
            'scheduler': 'step',
            'lr_step': 10,
            'lr_gamma': 0.5,
            'patience': 50,
            'iterations_per_epoch': 2,
            'num_workers': 0,
        },
        'results_dir': os.path.join(tmpdir, 'results'),
    }
    if model_type == 'prototypical':
        config['model']['distance'] = 'cosine'
        config['training']['n_way'] = 3
        config['training']['k_shot'] = 2
        config['training']['q_query'] = 2
        del config['training']['batch_size']
        del config['training']['loss']

    if extra_training:
        config['training'].update(extra_training)

    path = os.path.join(tmpdir, 'config.yaml')
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


def _make_dummy_dataset(num_subjects=15, images_per_subject=10):
    """Create a minimal dataset-like object with synthetic data for Trainer."""
    from data.base_loader import BiometricDataset
    import cv2

    tmpdir = tempfile.mkdtemp()

    class _DummyDataset(BiometricDataset):
        IMG_SIZE = (96, 96)

        def __init__(self, root_dir, **kwargs):
            self._num_subjects = num_subjects
            self._images_per_subject = images_per_subject
            self._tmpdir = root_dir
            super().__init__(root_dir, **kwargs)

        def _load_data(self):
            for subj_idx in range(self._num_subjects):
                subj_name = f"subj_{subj_idx:03d}"
                subj_dir = os.path.join(self._tmpdir, subj_name)
                os.makedirs(subj_dir, exist_ok=True)

                genuine_paths = []
                for img_idx in range(self._images_per_subject):
                    img = np.random.randint(
                        0, 256, (96, 96), dtype=np.uint8
                    )
                    path = os.path.join(subj_dir, f"gen_{img_idx}.png")
                    cv2.imwrite(path, img)
                    genuine_paths.append(path)

                self.data[subj_name] = {
                    'genuine': genuine_paths,
                    'forgery': [],
                }

        def _preprocess(self, image):
            img_arr = np.array(image)
            return cv2.resize(img_arr, self.IMG_SIZE)

    ds = _DummyDataset(tmpdir)
    return ds, tmpdir


# ── CSV Logging Tests ─────────────────────────────────────────────────


class TestCSVLogging:
    """Verify CSV training log is created and populated correctly."""

    def test_csv_log_created_on_init(self):
        """Trainer.__init__ should create training_log.csv with header."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            csv_path = os.path.join(tmpdir, 'results', 'logs',
                                    'training_log.csv')
            assert os.path.exists(csv_path), "CSV log file was not created"

            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
            assert 'epoch' in header
            assert 'train_loss' in header
            assert 'timestamp' in header
        finally:
            shutil.rmtree(tmpdir)

    def test_csv_log_columns(self):
        """CSV header should contain all expected columns."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)

            expected = [
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                'lr', 'is_best', 'elapsed_s', 'timestamp',
            ]
            csv_path = os.path.join(tmpdir, 'results', 'logs',
                                    'training_log.csv')
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == expected
        finally:
            shutil.rmtree(tmpdir)

    def test_csv_log_populated_after_train(self):
        """After 2-epoch training, CSV should have header + 2 data rows."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            trainer.train(ds)

            csv_path = os.path.join(tmpdir, 'results', 'logs',
                                    'training_log.csv')
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            # 1 header + 2 data rows
            assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)

    def test_csv_log_append_on_resume(self):
        """Resuming training should append rows, not overwrite."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            # Use 4 total epochs: first run trains 1-2, resume trains 3-4
            config_path = _make_config(
                tmpdir, extra_training={'epochs': 4, 'patience': 50}
            )
            trainer = Trainer(config_path)
            # Override epochs to 2 for first run
            trainer.config['training']['epochs'] = 2
            trainer.train(ds)

            # Get checkpoint path
            ckpt_path = os.path.join(
                tmpdir, 'results', 'checkpoints',
                'checkpoint_epoch_2.pth'
            )
            assert os.path.exists(ckpt_path)

            # Resume: create new trainer with 4 epochs, load from epoch 2
            trainer2 = Trainer(config_path)
            trainer2.config['training']['epochs'] = 4
            trainer2.load_checkpoint(ckpt_path)
            trainer2.train(ds)  # trains epochs 3-4

            csv_path = os.path.join(tmpdir, 'results', 'logs',
                                    'training_log.csv')
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            # 1 header + 2 from first run + 2 from second run = 5
            assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)


# ── TensorBoard Tests ─────────────────────────────────────────────────


class TestTensorBoard:
    """Verify TensorBoard integration."""

    def test_tensorboard_disabled_by_default(self):
        """Trainer with no tensorboard key should have _tb_writer == None."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            assert trainer._tb_writer is None
        finally:
            shutil.rmtree(tmpdir)

    def test_tensorboard_creates_logdir(self):
        """When tensorboard: true, the tensorboard/ subdir should exist."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(
                tmpdir, extra_training={'tensorboard': True}
            )
            trainer = Trainer(config_path)
            tb_dir = os.path.join(tmpdir, 'results', 'logs', 'tensorboard')
            assert os.path.isdir(tb_dir), \
                f"TensorBoard directory not created at {tb_dir}"
            # Writer should be set
            assert trainer._tb_writer is not None
            trainer._tb_writer.close()
        finally:
            shutil.rmtree(tmpdir)


# ── Training Curves Plot Test ─────────────────────────────────────────


class TestTrainingCurvesPlot:
    """Verify post-training plot is generated."""

    def test_training_curves_plot_generated(self):
        """After 2-epoch training, training_curves.png should exist."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            trainer.train(ds)

            plot_path = os.path.join(
                tmpdir, 'results', 'figures', 'training_curves.png'
            )
            assert os.path.exists(plot_path), \
                "Training curves plot was not generated"
            # Verify it's a valid file (not empty)
            assert os.path.getsize(plot_path) > 0
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)


# ── AMP Tests ─────────────────────────────────────────────────────────


class TestAMP:
    """Verify mixed-precision training behavior."""

    def test_amp_disabled_on_cpu(self):
        """Even with amp: true, use_amp should be False on CPU."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(
                tmpdir, extra_training={'amp': True}
            )
            trainer = Trainer(config_path)
            # On a CPU-only machine, AMP must be disabled
            if trainer.device.type != 'cuda':
                assert trainer.use_amp is False, \
                    "AMP should be disabled on non-CUDA device"
                assert trainer.scaler is None, \
                    "Scaler should be None on non-CUDA device"
        finally:
            shutil.rmtree(tmpdir)

    def test_amp_scaler_in_checkpoint(self):
        """When AMP is enabled, checkpoint should contain scaler state."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(
                tmpdir, extra_training={'amp': True}
            )
            trainer = Trainer(config_path)
            trainer.epoch = 1
            trainer.save_checkpoint(filename='test.pth')

            ckpt_path = os.path.join(
                tmpdir, 'results', 'checkpoints', 'test.pth'
            )
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)

            if trainer.use_amp:
                assert 'scaler_state_dict' in ckpt, \
                    "AMP scaler state not saved in checkpoint"
            else:
                # On CPU, scaler is None, so key shouldn't be there
                assert 'scaler_state_dict' not in ckpt
        finally:
            shutil.rmtree(tmpdir)

    def test_old_checkpoint_loads_without_scaler(self):
        """Loading pre-enhancement checkpoint (no scaler key) should work."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)

            # Simulate old checkpoint (no scaler_state_dict key)
            old_ckpt = {
                'epoch': 5,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_loss': 0.5,
                'config': trainer.config,
            }
            ckpt_path = os.path.join(
                tmpdir, 'results', 'checkpoints', 'old.pth'
            )
            torch.save(old_ckpt, ckpt_path)

            # Should not crash
            trainer.load_checkpoint(ckpt_path)
            assert trainer.epoch == 5
        finally:
            shutil.rmtree(tmpdir)


# ── Checkpoint Persistence Tests ──────────────────────────────────────


class TestCheckpointPersistence:
    """Verify that scheduler, patience, and epoch state survive resume."""

    def test_scheduler_state_saved_and_restored(self):
        """Scheduler state should be in checkpoint and restored on load."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)

            # Step the scheduler a few times to change its state
            for _ in range(3):
                trainer.scheduler.step()
            original_lr = trainer.optimizer.param_groups[0]['lr']

            trainer.epoch = 3
            trainer.save_checkpoint(filename='sched.pth')

            ckpt_path = os.path.join(
                tmpdir, 'results', 'checkpoints', 'sched.pth'
            )
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)
            assert 'scheduler_state_dict' in ckpt

            # Create fresh trainer and load — LR should match
            trainer2 = Trainer(config_path)
            trainer2.load_checkpoint(ckpt_path)
            restored_lr = trainer2.optimizer.param_groups[0]['lr']
            assert abs(original_lr - restored_lr) < 1e-10, \
                f"LR mismatch: {original_lr} vs {restored_lr}"
        finally:
            shutil.rmtree(tmpdir)

    def test_patience_counter_saved_and_restored(self):
        """Patience counter should be in checkpoint and restored on load."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            trainer.patience_counter = 7
            trainer.epoch = 5
            trainer.save_checkpoint(filename='patience.pth')

            ckpt_path = os.path.join(
                tmpdir, 'results', 'checkpoints', 'patience.pth'
            )
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)
            assert ckpt['patience_counter'] == 7

            trainer2 = Trainer(config_path)
            trainer2.load_checkpoint(ckpt_path)
            assert trainer2.patience_counter == 7
        finally:
            shutil.rmtree(tmpdir)

    def test_resume_continues_from_correct_epoch(self):
        """After loading checkpoint at epoch N, training should start at N+1."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            # Train for 2 epochs (config has epochs=4 for this test)
            config_path = _make_config(
                tmpdir, extra_training={'epochs': 4, 'patience': 50}
            )
            trainer = Trainer(config_path)
            trainer.train(ds)  # trains epochs 1-4

            # Now create a scenario: train 2 epochs, save, resume, train 2 more
            config_path2 = _make_config(
                tmpdir, extra_training={'epochs': 4, 'patience': 50}
            )
            trainer2 = Trainer(config_path2)
            # Manually set epoch=2 and save
            trainer2.epoch = 2
            trainer2.save_checkpoint(filename='mid.pth')

            # Resume from epoch 2 — should train epochs 3 and 4 only
            trainer3 = Trainer(config_path2)
            trainer3.load_checkpoint(os.path.join(
                tmpdir, 'results', 'checkpoints', 'mid.pth'
            ))
            assert trainer3.epoch == 2

            # After training, final epoch should be 4 (not 4 starting from 1)
            trainer3.train(ds)
            assert trainer3.epoch == 4
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)


# ── Gradient Accumulation Tests ───────────────────────────────────────


class TestGradientAccumulation:
    """Verify gradient accumulation behavior."""

    def test_accumulation_steps_default(self):
        """Default accumulation_steps should be 1."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(tmpdir)
            trainer = Trainer(config_path)
            assert trainer.accumulation_steps == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_accumulation_steps_configurable(self):
        """accumulation_steps should be settable via config."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _make_config(
                tmpdir, extra_training={'accumulation_steps': 4}
            )
            trainer = Trainer(config_path)
            assert trainer.accumulation_steps == 4
        finally:
            shutil.rmtree(tmpdir)

    def test_accumulation_training_runs(self):
        """Training with accumulation_steps > 1 should complete without error."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            config_path = _make_config(
                tmpdir, extra_training={'accumulation_steps': 2}
            )
            trainer = Trainer(config_path)
            trainer.train(ds)
            # Should have completed 2 epochs
            assert trainer.epoch == 2
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)


# ── Prototypical Integration Test ─────────────────────────────────────


class TestPrototypicalEnhancements:
    """Verify enhancements work for prototypical model type too."""

    def test_proto_training_with_logging(self):
        """Prototypical training with CSV logging should work."""
        tmpdir = tempfile.mkdtemp()
        ds, ds_tmpdir = _make_dummy_dataset()
        try:
            config_path = _make_config(tmpdir, model_type='prototypical')
            trainer = Trainer(config_path)
            trainer.train(ds)

            csv_path = os.path.join(tmpdir, 'results', 'logs',
                                    'training_log.csv')
            assert os.path.exists(csv_path)

            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) == 3  # header + 2 epochs
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(ds_tmpdir)

