"""Train the standalone inverse BiLSTM on a saved real FDTD dataset."""
# 训练入口：读取固定划分的数据 → 批量前向/反向传播 → 按验证误差选择模型
#          → 用最佳权重评估训练/验证/测试集 → 导出归一化与 nm 单位的预测。
# 网络结构见 model.py；本文件负责优化、早停、续训和结果保存。
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from schema import PARAMETER_NAMES, ROTATION_PARAMETER_NAMES, QUADRANT_PARAMETER_NAMES, PROJECT_ROOT
from utils import atomic_json, checksum, directory_lock, utc_now
from model import InverseBiLSTM, BiLSTMConfig, geometry_mse
from dataset import SpectrumGeometryDataset

LOG = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/train_au100_100.json'


@dataclass(frozen=True)
class TrainingConfig:
    dataset: Path
    output_dir: Path
    allow_validation_data: bool = False
    seed: int = 42
    device: str = 'auto'
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 35
    min_delta: float = 1e-5
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    mixed_precision: bool = True
    model: BiLSTMConfig = field(default_factory=BiLSTMConfig)

    def __post_init__(self):
        # 在启动训练前检查配置，避免无效批量、学习率或输入维度运行到中途才报错。
        for name in ('seed', 'epochs', 'batch_size', 'early_stopping_patience', 'lr_patience'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        for name in ('learning_rate', 'gradient_clip_norm', 'min_learning_rate'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('weight_decay', 'min_delta'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.lr_factor < 1 or self.min_learning_rate > self.learning_rate:
            raise ValueError('Invalid learning-rate schedule')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        if not isinstance(self.allow_validation_data, bool) or not isinstance(self.mixed_precision, bool):
            raise ValueError('Data permission and mixed_precision must be booleans')
        if self.model.spectrum_points != 1000 or self.model.geometry_dim not in (5, 6, 7):
            raise ValueError('FDTD dataset requires 1000 inputs and five, six or seven outputs')
        if self.model.geometry_dim == 6 and self.model.angle_symmetry != 'square_mirror':
            raise ValueError('The common-square rotation task requires square_mirror angle loss')
        if self.model.geometry_dim == 7 and self.model.angle_symmetry != 'square_periodic':
            raise ValueError('Quadrant rotations require square_periodic angle loss')

    def metadata(self):
        result = asdict(self)
        result['dataset'] = str(self.dataset)
        result['output_dir'] = str(self.output_dir)
        return result


def load_training_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding='utf-8'))
    # JSON 中的数据和输出路径相对于配置文件解析，便于从 PyCharm 或命令行启动。
    for key in ('dataset', 'output_dir'):
        values[key] = (path.parent / values[key]).resolve()
    values['model'] = BiLSTMConfig.from_dict(values.get('model', {}))
    return TrainingConfig(**values)


def matching_training_config(saved, current):
    """Compare standalone BiLSTM configurations before resumption."""
    saved, current = dict(saved), dict(current)
    try:
        saved['model'] = asdict(BiLSTMConfig.from_dict(saved['model']))
        current['model'] = asdict(BiLSTMConfig.from_dict(current['model']))
    except (TypeError, ValueError):
        return False
    return saved == current


def validate_training_splits(data):
    """Reject leakage or omitted rows before any parameter updates."""
    n = len(data['spectra'])
    pieces = []
    # 使用采样阶段已保存的划分，检查重复、越界、遗漏和训练/验证/测试交叉。
    for split in ('train', 'val', 'test'):
        indices = data[split + '_indices']
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
            raise ValueError(f'{split} needs nonempty integer indices')
        if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
            raise ValueError(f'Invalid or duplicate {split} indices')
        if not np.all(data['split_names'][indices] == split):
            raise ValueError(f'{split} indices disagree with stored split labels')
        pieces.append(indices)
    if not np.array_equal(np.sort(np.concatenate(pieces)), np.arange(n)):
        raise ValueError('Train/validation/test splits overlap or omit samples')
    if len(np.unique(data['sample_ids'])) != n:
        raise ValueError('Dataset sample IDs must be unique')
    # 即使允许使用工程验证数据，也要求每个训练样本实际达到自动关断收敛。
    if np.any(data['status'] != 2):
        raise ValueError('Training requires every sample to reach auto-shutoff (status=2)')


def load_training_data(config):
    with np.load(config.dataset, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    validate_training_splits(data)
    # 三个 Dataset 读取同一文件中的不同固定索引；训练期间不重新随机划分。
    datasets = {split: SpectrumGeometryDataset(config.dataset, split,
                    allow_validation_data=config.allow_validation_data)
                for split in ('train', 'val', 'test')}
    if len(datasets['train'].parameter_names) != config.model.geometry_dim:
        raise ValueError('Dataset parameter count differs from model output count')
    return datasets


def normalized_errors(prediction, target, angle_symmetry='none'):
    difference = prediction - target
    if angle_symmetry == 'square_mirror':
        if prediction.shape[-1] != 6:
            raise ValueError('Mirror-equivalent angle metrics require six outputs')
        difference = difference.copy()
        difference[..., 5] = np.abs(2 * prediction[..., 5] - 1) - np.abs(2 * target[..., 5] - 1)
    elif angle_symmetry == 'square_periodic':
        if prediction.shape[-1] != 7:
            raise ValueError('Quadrant periodic metrics require seven outputs')
        difference = difference.copy()
        difference[..., 5:] = (difference[..., 5:] + .5) % 1 - .5
    elif angle_symmetry != 'none':
        raise ValueError('Unknown angle symmetry')
    return difference


def regression_metrics(prediction, target, lower_nm, upper_nm, parameter_names=None, angle_symmetry='none'):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] not in (5, 6, 7) or not len(target):
        raise ValueError('Metrics require matching nonempty [N,5] or [N,6] arrays')
    names = tuple(parameter_names) if parameter_names is not None else (
        QUADRANT_PARAMETER_NAMES if prediction.shape[1] == 7 else (ROTATION_PARAMETER_NAMES if prediction.shape[1] == 6 else PARAMETER_NAMES))
    if names not in (PARAMETER_NAMES, ROTATION_PARAMETER_NAMES, QUADRANT_PARAMETER_NAMES) or len(names) != prediction.shape[1]:
        raise ValueError('Metric parameter order differs from the supported schema')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Predictions and labels must be finite')
    difference = normalized_errors(prediction, target, angle_symmetry)
    # 按各参数跨度还原误差；前五列为 nm，可选第六列为 deg。
    # 总 normalized_mse 衡量结构参数误差；镜像模式的角度项比较 |theta|/45。
    width = np.asarray(upper_nm) - np.asarray(lower_nm)
    if width.shape != (len(names),) or not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError('Metric bounds must match every parameter')
    physical = difference * width
    if angle_symmetry == 'square_mirror':
        # 等效角归一化跨度为 45 度，原有符号角的保存跨度为 90 度。
        physical[:, 5] *= .5
    result = {
        'samples': len(target), 'normalized_mse': float(np.mean(difference ** 2)),
        'normalized_rmse': float(np.sqrt(np.mean(difference ** 2))),
        'normalized_mae': float(np.mean(np.abs(difference))),
        'mae_nm': dict(zip(PARAMETER_NAMES, np.mean(np.abs(physical[:, :5]), axis=0).tolist())),
        'rmse_nm': dict(zip(PARAMETER_NAMES, np.sqrt(np.mean(physical[:, :5] ** 2, axis=0)).tolist())),
    }
    if len(names) == 6:
        raw_degrees = (prediction[:, 5] - target[:, 5]) * width[5]
        result.update(mae_deg={'rotation_deg': float(np.mean(np.abs(physical[:, 5])))},
                      rmse_deg={'rotation_deg': float(np.sqrt(np.mean(physical[:, 5] ** 2)))},
                      raw_signed_mae_deg=float(np.mean(np.abs(raw_degrees))),
                      angle_symmetry=angle_symmetry)
    if len(names) == 7:
        result.update(mae_deg=dict(zip(names[5:], np.mean(np.abs(physical[:, 5:]), axis=0).tolist())),
                      rmse_deg=dict(zip(names[5:], np.sqrt(np.mean(physical[:, 5:]**2, axis=0)).tolist())),
                      angle_symmetry=angle_symmetry)
    return result


def autocast_context(device, enabled):
    # AMP 仅在外层确认使用 CUDA 时启用；CPU 或关闭混合精度时使用常规精度。
    return torch.autocast(device_type='cuda', dtype=torch.float16) if enabled else nullcontext()


def train_epoch(model, loader, optimizer, device, scaler, use_amp, clip_norm):
    # train() 启用 Dropout；每个 batch 为完整光谱 [B,1000] 及五、六或七列标签。
    model.train()
    total, count, steps = 0.0, 0, 0
    for spectrum, target in loader:
        spectrum, target = spectrum.to(device), target.to(device)
        # 各 batch 独立计算梯度，防止上一批的梯度累积到本批。
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            prediction = model(spectrum)
            # 前向可使用半精度，但把预测转换成 float32 后计算归一化尺寸 MSE。
            loss = geometry_mse(prediction.float(), target, model.config.angle_symmetry)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        # 误差沿回归头、双向 LSTM 和光谱嵌入反向传播，计算各参数梯度。
        # AMP 时先放大损失，降低小梯度在半精度下发生下溢的风险。
        scaler.scale(loss).backward()
        # 裁剪前先还原梯度尺度，使 clip_norm 针对实际梯度范数生效。
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if torch.isfinite(norm):
            # AdamW 根据梯度更新可训练参数；注意力矩阵本身由下一次输入重新计算。
            scaler.step(optimizer)
            steps += 1
        elif not use_amp:
            raise ValueError('Nonfinite gradients')
        scaler.update()
        # 按样本数加权累加，避免不足一个完整 batch 的末批与大 batch 权重相同。
        total += float(loss.detach()) * len(target)
        count += len(target)
    if not steps:
        raise ValueError('No finite optimizer update in this epoch')
    return total / count, steps


@torch.inference_mode()
def evaluate(model, loader, device):
    """Evaluation uses full precision and eval mode with dropout disabled."""
    model.eval()
    # inference_mode 禁用梯度记录，eval 关闭 Dropout；评估不做优化更新。
    predictions, labels = [], []
    for spectrum, target in loader:
        prediction = model(spectrum.to(device))
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite evaluation prediction')
        predictions.append(prediction.cpu().numpy())
        labels.append(target.numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(labels)
    return float(np.mean(normalized_errors(prediction, target, model.config.angle_symmetry) ** 2)), prediction, target


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    # 先写临时文件再替换，减少中断时留下半个检查点的风险。
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_snapshot(generator):
    # 保存模型随机过程和 DataLoader 打乱顺序的状态，供完整断点续训恢复。
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    # 只加载权重不足以恢复原训练轨迹；还需恢复随机数、优化器及调度器状态。
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    generator.set_state(state['loader'].cpu())


def export_predictions(path, dataset, prediction, target):
    lower, width = dataset.lower_values, dataset.upper_values - dataset.lower_values
    # 逐参数用 lower + g_01 * (upper - lower) 还原物理值，各列单位由数据集标明。
    truth_nm, prediction_nm = lower + target * width, lower + prediction * width
    # 同时输出样本 ID、归一化值和物理值；长度列用 nm，角度列用 deg。
    columns = [dataset.sample_ids[:, None], target, prediction, truth_nm, prediction_nm]
    names, units = dataset.parameter_names, dataset.parameter_units
    header = (['sample_id'] + [f'true_01_{p}' for p in names]
              + [f'pred_01_{p}' for p in names] + [f'true_{u}_{p}' for p, u in zip(names, units)]
              + [f'pred_{u}_{p}' for p, u in zip(names, units)])
    temporary = path.with_suffix('.csv.tmp')
    np.savetxt(temporary, np.column_stack(columns), delimiter=',', header=','.join(header), comments='',
               fmt=['%d'] + ['%.10g'] * (4 * len(names)))
    os.replace(temporary, path)


def train(config, resume=None):
    datasets = load_training_data(config)
    # 数据文件的 SHA256 与配置写入运行元信息，续训时检查是否换过数据。
    dataset_hash = checksum(config.dataset)
    device = torch.device('cuda' if (config.device == 'auto' and torch.cuda.is_available()) else
                          'cpu' if config.device == 'auto' else config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(resume).resolve() if resume else config.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not resume:
        run_dir.mkdir(exist_ok=False)
    elif not (run_dir / 'last.pt').is_file():
        raise ValueError('Resume requires a run directory containing last.pt')
    with directory_lock(config.output_dir), directory_lock(run_dir):
        # 目录锁防止两次训练同时覆盖同一个运行目录或 latest_run.json。
        if resume:
            saved_metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
            if not matching_training_config(saved_metadata['config'], config.metadata()) or saved_metadata['dataset_sha256'] != dataset_hash:
                raise ValueError('Resume configuration or dataset differs from saved run')
        file_log = logging.FileHandler(run_dir / 'training.log', encoding='utf-8')
        file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOG.addHandler(file_log)
        try:
            return _train_locked(config, datasets, device, dataset_hash, run_dir, resume)
        except BaseException as error:
            atomic_json(run_dir / 'status.json', {'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        'time': utc_now(), 'error': str(error), 'resume_from': str(run_dir / 'last.pt')})
            LOG.exception('Training did not complete; saved checkpoints can be resumed')
            atomic_json(config.output_dir / 'latest_run.json', {
                'run_dir': str(run_dir), 'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'dataset': str(config.dataset)})
            raise
        finally:
            LOG.removeHandler(file_log)
            file_log.close()


def _train_locked(config, datasets, device, dataset_hash, run_dir, resume):
    # 固定随机种子和 cuDNN 选项，使本机相同配置的训练更便于重复核对。
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # FP32 evaluation should agree with the portable CPU checkpoint. cuDNN's
    # default TF32 policy otherwise changes some small-batch LSTM predictions.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    use_amp = device.type == 'cuda' and config.mixed_precision
    generator = torch.Generator().manual_seed(config.seed)
    # 训练集每轮打乱，验证集保持固定顺序；num_workers=0 适配当前 Windows 入口。
    train_loader = DataLoader(datasets['train'], batch_size=config.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    # Test data is evaluated only after training and validation-based selection finish.
    # 验证集用于调学习率和选最佳模型；测试集在这一阶段不参与训练或模型选择。
    val_loader = DataLoader(datasets['val'], batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = InverseBiLSTM(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # 验证误差连续多轮不改善时降低学习率，帮助后续优化使用更小的步长。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                   patience=config.lr_patience, min_lr=config.min_learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    # 推理需要保留网络配置、尺寸上下界和波长网格，不能仅保存一组裸权重。
    metadata = {
        'config': config.metadata(), 'dataset_sha256': dataset_hash,
        'dataset_config_fingerprint': datasets['train'].metadata['config_fingerprint'],
        'parameter_names': list(datasets['train'].parameter_names), 'lower_nm': datasets['train'].lower_nm.tolist(),
        'upper_nm': datasets['train'].upper_nm.tolist(),
        'parameter_units': list(datasets['train'].parameter_units),
        'lower_values': datasets['train'].lower_values.tolist(),
        'upper_values': datasets['train'].upper_values.tolist(),
        'wavelength_nm': datasets['train'].wavelength_nm.tolist(),
        'source_dataset_metadata': datasets['train'].metadata,
        'split_sample_ids': {s: datasets[s].sample_ids.tolist() for s in datasets},
        'model_family': 'standalone_bilstm',
        'implementation_sha256': {name: checksum(PROJECT_ROOT / name) for name in ('model.py', 'train.py', 'dataset.py', 'schema.py', 'utils.py')},
        'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'pytorch_version': str(torch.__version__), 'model_parameters': sum(p.numel() for p in model.parameters()),
        'mixed_precision_active': use_amp, 'quality_level': datasets['train'].metadata['config']['quality_level'],
        'fp32_evaluation_tf32_enabled': False,
        'selection': 'Lowest validation normalized MSE; test unused until final evaluation',
    }
    history, best_loss, best_epoch, bad_epochs, total_steps, previous_seconds = [], math.inf, 0, 0, 0, 0.0
    if resume:
        # last.pt 包含上一次完整轮次后的训练状态，best.pt 则用于最终推理/评估。
        checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=True)
        saved = checkpoint['metadata']
        if not matching_training_config(saved['config'], metadata['config']) or saved['dataset_sha256'] != dataset_hash:
            raise ValueError('Resume configuration or dataset differs from saved run')
        if saved['implementation_sha256'] != metadata['implementation_sha256']:
            raise ValueError('Resume implementation differs from saved run')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        history = checkpoint['history']
        best_loss, best_epoch = checkpoint['best_val_mse'], checkpoint['best_epoch']
        bad_epochs, total_steps = checkpoint['bad_epochs'], checkpoint['optimizer_steps']
        previous_seconds = checkpoint['training_seconds']
        restore_rng(checkpoint['rng'], generator)
        LOG.info('Resuming after epoch %d', checkpoint['epoch'])
    atomic_json(run_dir / 'run.json', metadata)
    latest = config.output_dir / 'latest_run.json'
    atomic_json(latest, {'run_dir': str(run_dir), 'state': 'training', 'dataset': str(config.dataset)})
    LOG.info('Run: %s', run_dir)
    LOG.info('Device: %s; model parameters: %d; splits: %s', metadata['gpu_name'] or device,
             metadata['model_parameters'], {s: len(d) for s, d in datasets.items()})
    LOG.info('Dataset quality: %s; mixed precision: %s', metadata['quality_level'], use_amp)
    start = time.monotonic()
    stop_reason = 'early_stopping' if bad_epochs >= config.early_stopping_patience else 'maximum_epochs'
    for epoch in range(len(history) + 1, config.epochs + 1):
        if bad_epochs >= config.early_stopping_patience:
            break
        epoch_start = time.monotonic()
        used_lr = optimizer.param_groups[0]['lr']
        train_loss, steps = train_epoch(model, train_loader, optimizer, device, scaler, use_amp, config.gradient_clip_norm)
        total_steps += steps
        # 每轮先训练，再以不更新权重的方式计算验证误差，交给学习率调度器。
        val_loss, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        # Always save the actual best validation model; min_delta only controls patience.
        # 只要验证误差创新低就保存 best.pt；min_delta 仅判断是否重置早停计数。
        significant = val_loss < best_loss - config.min_delta
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            save_checkpoint(run_dir / 'best.pt', {'model_state': model.state_dict(), 'metadata': metadata,
                                                 'epoch': epoch, 'val_mse': val_loss})
        bad_epochs = 0 if significant else bad_epochs + 1
        history.append({'epoch': epoch, 'train_mse': train_loss, 'val_mse': val_loss,
                        'learning_rate': used_lr, 'optimizer_steps': steps,
                        'epoch_seconds': time.monotonic() - epoch_start})
        training_seconds = previous_seconds + time.monotonic() - start
        # 每轮保存权重、优化器、学习率调度器、AMP、随机数和历史，便于中断后继续。
        save_checkpoint(run_dir / 'last.pt', {
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
            'metadata': metadata, 'epoch': epoch, 'history': history, 'best_val_mse': best_loss,
            'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'optimizer_steps': total_steps,
            'training_seconds': training_seconds, 'rng': rng_snapshot(generator),
        })
        atomic_json(run_dir / 'history.json', history)
        status = {'state': 'training', 'epoch': epoch, 'max_epochs': config.epochs,
                  'best_epoch': best_epoch, 'best_val_mse': best_loss, 'train_mse': train_loss,
                  'val_mse': val_loss, 'optimizer_steps': total_steps, 'time': utc_now()}
        atomic_json(run_dir / 'status.json', status)
        LOG.info('Epoch %03d/%03d train=%.6f val=%.6f best=%.6f (epoch %d) lr=%.2g %.1fs',
                 epoch, config.epochs, train_loss, val_loss, best_loss, best_epoch, used_lr,
                 time.monotonic() - epoch_start)
        if bad_epochs >= config.early_stopping_patience:
            stop_reason = 'early_stopping'
            break
    # 恢复验证集选出的最佳权重，而非直接使用早停时的最后一轮权重。
    selected = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model_state'])
    # 常数基线只使用训练集标签均值，检查模型是否比不读取光谱的预测更好。
    train_mean = datasets['train'].labels.mean(dim=0).numpy()
    if config.model.angle_symmetry == 'square_mirror':
        # 等效角常数基线取角度绝对值均值，避免带符号平均互相抵消到零。
        train_mean[5] = .5 + (datasets['train'].labels[:, 5] - .5).abs().mean().item()
    scores, baselines = {}, {}
    for split, dataset in datasets.items():
        # 到这里才评估测试集，并导出各划分的逐样本预测和物理单位误差。
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        _, prediction, target = evaluate(model, loader, device)
        scores[split] = regression_metrics(prediction, target, dataset.lower_values, dataset.upper_values,
                                           dataset.parameter_names, config.model.angle_symmetry)
        constant = np.broadcast_to(train_mean, target.shape)
        baselines[split] = regression_metrics(constant, target, dataset.lower_values, dataset.upper_values,
                                              dataset.parameter_names, config.model.angle_symmetry)
        export_predictions(run_dir / (split + '_predictions.csv'), dataset, prediction, target)
    report = {**metadata, 'training_performed': True, 'completed_at': utc_now(),
              'epochs_completed': len(history), 'best_epoch': best_epoch, 'best_val_mse': best_loss,
              'stop_reason': stop_reason, 'optimizer_steps': total_steps,
              'training_seconds': previous_seconds + time.monotonic() - start,
              'metrics': scores, 'train_mean_baseline': baselines,
              'best_checkpoint': str(run_dir / 'best.pt'), 'best_checkpoint_sha256': checksum(run_dir / 'best.pt')}
    atomic_json(run_dir / 'metrics.json', report)
    from plot_training import plot_training
    # 保存静态训练曲线，show=False 不在训练结束时强制弹出绘图窗口。
    plot_training(run_dir, show=False)
    atomic_json(run_dir / 'status.json', {'state': 'completed', 'epochs_completed': len(history),
                'best_epoch': best_epoch, 'test_mse': scores['test']['normalized_mse'], 'time': utc_now()})
    atomic_json(latest, {'run_dir': str(run_dir), 'state': 'completed', 'dataset': str(config.dataset)})
    LOG.info('Finished. Best epoch %d; test MSE=%.6f; train-mean baseline=%.6f', best_epoch,
             scores['test']['normalized_mse'], baselines['test']['normalized_mse'])
    LOG.info('Best checkpoint: %s', run_dir / 'best.pt')
    return run_dir, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--resume', type=Path, help='Continue an existing run directory from last.pt')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    train(load_training_config(args.config), args.resume)


if __name__ == '__main__':
    main()
