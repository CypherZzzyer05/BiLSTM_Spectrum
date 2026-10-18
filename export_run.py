"""Validate a completed run on CPU and export portable weights and result files."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from dataset import SpectrumGeometryDataset
from model import BiLSTMConfig, InverseBiLSTM
from train import normalized_errors
from utils import atomic_json, checksum


def export(run, root, reference_metrics=None):
    report = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if saved['epoch'] != report['best_epoch']:
        raise ValueError('Checkpoint epoch differs from validation selection')
    for name, expected in report['implementation_sha256'].items():
        if checksum(root / name) != expected:
            raise ValueError('Source implementation changed after training: ' + name)
    data_path = root / 'examples/au100_metal8_100/dataset.npz'
    if checksum(data_path) != report['dataset_sha256']:
        raise ValueError('Example dataset differs from training data')
    model = InverseBiLSTM(BiLSTMConfig.from_dict(saved['metadata']['config']['model'])).eval()
    model.load_state_dict(saved['model_state'])
    maximum_difference = 0.0
    for split in ('train', 'val', 'test'):
        dataset = SpectrumGeometryDataset(data_path, split, True)
        with torch.inference_mode():
            prediction = model(dataset.spectra).numpy()
        csv = np.atleast_1d(np.genfromtxt(run / (split + '_predictions.csv'), delimiter=',', names=True))
        expected = np.column_stack([csv['pred_01_' + name] for name in dataset.parameter_names])
        difference = float(np.max(np.abs(prediction - expected)))
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-5:
            raise ValueError(f'CPU checkpoint predictions disagree: {split}, {difference}')
        if not np.array_equal(dataset.sample_ids, csv['sample_id']):
            raise ValueError('Prediction sample order differs from dataset split')
        errors = normalized_errors(prediction, dataset.labels.numpy(), 'square_periodic')
        if abs(float(np.mean(errors**2)) - report['metrics'][split]['normalized_mse']) > 1e-6:
            raise ValueError('Independently recomputed metric disagrees: ' + split)
        if split == 'train':
            mean = dataset.labels.mean(0).numpy()
        if split == 'test':
            targets, predictions = dataset.labels.numpy(), prediction

    destination = root / 'results/au100_seed42'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.json', 'history.json', 'run.json', 'training.log', 'training_preview.png',
                 'train_predictions.csv', 'val_predictions.csv', 'test_predictions.csv'):
        shutil.copyfile(run / name, destination / name)
    checkpoint_path = root / 'models/au100_seed42.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(saved, checkpoint_path)
    per_sample_gain = (normalized_errors(np.broadcast_to(mean, targets.shape), targets, 'square_periodic')**2
                       - normalized_errors(predictions, targets, 'square_periodic')**2).mean(1)
    rng = np.random.default_rng(20261018)
    bootstrap = per_sample_gain[rng.integers(0, len(targets), (10000, len(targets)))].mean(1)
    score = report['metrics']['test']['normalized_mse']
    baseline = report['train_mean_baseline']['test']['normalized_mse']
    summary = {
        'dataset_sha256': report['dataset_sha256'],
        'split_sizes': {k: len(v) for k, v in report['split_sample_ids'].items()},
        'seed': report['config']['seed'], 'model_parameters': report['model_parameters'],
        'epochs_completed': report['epochs_completed'], 'best_epoch': report['best_epoch'],
        'optimizer_steps': report['optimizer_steps'], 'training_loop_seconds': report['training_seconds'],
        'test_mse': score, 'train_mean_baseline_test_mse': baseline,
        'test_mse_improvement_over_mean_percent': 100*(1-score/baseline),
        'paired_bootstrap_mean_mse_gain_95pct_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
        'bootstrap_definition': 'Resample the 10 paired test rows 10000 times, seed 20261018; baseline MSE minus model MSE; diagnostic only, no model selection',
        'cpu_reload_max_absolute_prediction_difference': maximum_difference,
        'source_files_unchanged_after_training': True,
        'checkpoint_sha256': checksum(checkpoint_path),
        'evaluation_limit': 'Geometry errors only; no spectral reconstruction or FDTD validation; pipeline_validation data; one training seed and 10 test rows',
        'precision_audit': 'TF32 disabled for FP32 evaluation after detecting batch-dependent cuDNN TF32/CPU discrepancies; this run was repeated for numerical portability, not test-metric tuning',
    }
    if reference_metrics:
        old = json.loads(reference_metrics.read_text(encoding='utf-8'))
        if old['dataset_sha256'] != report['dataset_sha256'] or old['split_sample_ids'] != report['split_sample_ids']:
            raise ValueError('Historical reference uses a different dataset or split')
        summary['historical_resnet_metaformer'] = {
            'parameters': old['model_parameters'], 'test_mse': old['metrics']['test']['normalized_mse'],
            'metrics_sha256': checksum(reference_metrics), 'same_dataset_and_splits': True,
            'comparison_limit': 'Historical reference only; different capacity; plain Metaformer without ResNet was not retrained in this experiment',
        }
    atomic_json(destination / 'summary.json', summary)
    return summary


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--reference-metrics', type=Path)
    args = parser.parse_args()
    run = args.run or Path(json.loads((root / 'runs/au100_bilstm_100/latest_run.json').read_text())['run_dir'])
    print(json.dumps(export(run, root, args.reference_metrics), indent=2))


if __name__ == '__main__':
    main()
