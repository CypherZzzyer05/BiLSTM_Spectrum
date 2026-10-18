"""Training integrity, real optimizer updates, and deterministic CPU resumption."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from schema import PARAMETER_NAMES
from utils import atomic_npz
from model import InverseBiLSTM, BiLSTMConfig
from train import (TrainingConfig, load_training_data, regression_metrics,
                             train, train_epoch, validate_training_splits)


TINY_MODEL = BiLSTMConfig(patch_count=10, embedding_dim=8, hidden_size=4,
                         recurrent_layers=1, mlp_dim=16, geometry_dim=5,
                         angle_symmetry='none')


def tiny_data():
    labels = np.random.default_rng(12).uniform(.1, .9, (6, 5)).astype(np.float32)
    lower = np.array([3000, 850, 800, 150, 150], dtype=float)
    upper = np.array([3400, 1050, 1100, 350, 350], dtype=float)
    return {'spectra': np.random.default_rng(3).uniform(0, 1, (6, 1000)).astype(np.float32),
            'geometry_01': labels, 'geometry_nm': lower + labels * (upper - lower),
            'lower_nm': lower, 'upper_nm': upper, 'wavelength_nm': np.linspace(4000, 8000, 1000),
            'sample_ids': np.arange(6), 'status': np.full(6, 2),
            'parameter_names': np.array(PARAMETER_NAMES),
            'train_indices': np.array([0, 1]), 'val_indices': np.array([2, 3]), 'test_indices': np.array([4, 5]),
            'split_names': np.array(['train', 'train', 'val', 'val', 'test', 'test']),
            'metadata_json': np.array(json.dumps({'config_fingerprint': 'test-only',
                                    'config': {'quality_level': 'pipeline_validation'}}))}


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_split_overlap_and_omissions_are_rejected(self):
        data = tiny_data()
        validate_training_splits(data)
        data['test_indices'] = np.array([3, 4])
        with self.assertRaises(ValueError):
            validate_training_splits(data)
        data = tiny_data()
        data['test_indices'] = np.array([4])
        with self.assertRaisesRegex(ValueError, 'overlap or omit'):
            validate_training_splits(data)

    def test_validation_permission_does_not_allow_unconverged_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = tiny_data()
            atomic_npz(root / 'dataset.npz', **data)
            config = TrainingConfig(root / 'dataset.npz', root / 'runs', model=TINY_MODEL)
            with self.assertRaisesRegex(ValueError, 'pipeline-validation'):
                load_training_data(config)
            data['status'][0] = 1
            atomic_npz(root / 'dataset.npz', **data)
            with self.assertRaisesRegex(ValueError, 'status=2'):
                load_training_data(replace(config, allow_validation_data=True))

    def test_physical_error_metrics_use_parameter_ranges(self):
        score = regression_metrics(np.full((2, 5), .6), np.full((2, 5), .5),
                                   np.zeros(5), np.array([400, 200, 300, 200, 200]))
        self.assertAlmostEqual(score['normalized_mse'], .01)
        for name, expected in zip(PARAMETER_NAMES, [40, 20, 30, 20, 20]):
            self.assertAlmostEqual(score['mae_nm'][name], expected)

    def test_training_changes_parameters_with_real_backward_and_optimizer(self):
        model = InverseBiLSTM(TINY_MODEL)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        data = tiny_data()
        loader = DataLoader(TensorDataset(torch.from_numpy(data['spectra']),
                                         torch.from_numpy(data['geometry_01'])), batch_size=2)
        loss, steps = train_epoch(model, loader, optimizer, torch.device('cpu'),
                                 torch.amp.GradScaler('cuda', enabled=False), False, 1.0)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(steps, 3)
        self.assertTrue(any(not torch.equal(before[k], p) for k, p in model.named_parameters()))

    def test_interrupted_training_resumes_with_same_weights_and_history(self):
        with TemporaryDirectory() as directory, patch('plot_training.plot_training'):
            root = Path(directory)
            atomic_npz(root / 'dataset.npz', **tiny_data())
            config = TrainingConfig(root / 'dataset.npz', root / 'runs', allow_validation_data=True,
                                    device='cpu', epochs=2, batch_size=2, model=TINY_MODEL)
            full_dir, report = train(config)
            full_checkpoint = torch.load(full_dir / 'last.pt', map_location='cpu', weights_only=True)
            calls = 0
            def interrupted_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt('test interruption')
                return train_epoch(*args, **kwargs)
            with patch('train.train_epoch', side_effect=interrupted_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train(config)
            pointer = json.loads((config.output_dir / 'latest_run.json').read_text())
            resumed_dir, resumed_report = train(config, resume=pointer['run_dir'])
            checkpoint = torch.load(resumed_dir / 'last.pt', map_location='cpu', weights_only=True)
            for key, weights in full_checkpoint['model_state'].items():
                self.assertTrue(torch.equal(weights, checkpoint['model_state'][key]), key)
            self.assertEqual(report['optimizer_steps'], resumed_report['optimizer_steps'])
            self.assertEqual(report['metrics']['test'], resumed_report['metrics']['test'])
            self.assertEqual(resumed_report['epochs_completed'], 2)
            self.assertTrue(resumed_report['training_performed'])
            with self.assertRaisesRegex(ValueError, 'differs'):
                train(replace(config, learning_rate=.002), resume=resumed_dir)
            status = json.loads((resumed_dir / 'status.json').read_text())
            self.assertEqual(status['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
