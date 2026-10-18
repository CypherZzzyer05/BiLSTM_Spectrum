"""Draw the trained BiLSTM architecture and verify its real forward shapes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

from model import BiLSTMConfig, InverseBiLSTM
from utils import checksum


ROOT = Path(__file__).resolve().parent
INK = '#203650'
MUTED = '#617487'
COLORS = {'blue': '#E8F1FB', 'green': '#E5F4ED', 'orange': '#FFF0DC',
          'purple': '#EEE9FA', 'gray': '#EDF1F5', 'border': '#C5D0DD'}


def inspect(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = BiLSTMConfig.from_dict(saved['metadata']['config']['model'])
    model = InverseBiLSTM(config).eval()
    model.load_state_dict(saved['model_state'])
    observed = {}
    handles = []

    def capture(name):
        def hook(_module, _args, output):
            if name == 'bilstm':
                sequence, (h, c) = output
                observed[name] = list(sequence.shape)
                observed['h_n'] = list(h.shape)
                observed['c_n'] = list(c.shape)
            else:
                observed[name] = list(output.shape)
        return hook

    for name in ('patch_embedding', 'position_encoding', 'bilstm', 'final_norm',
                 'regressor.0', 'regressor.1', 'regressor.2', 'regressor.3'):
        handles.append(model.get_submodule(name).register_forward_hook(capture(name)))
    handles.append(model.regressor.register_forward_pre_hook(
        lambda _module, args: observed.update(mean_pool=list(args[0].shape))))
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            prediction = model(torch.zeros(2, config.spectrum_points))
    finally:
        for handle in handles:
            handle.remove()
    assert observed['patch_embedding'] == [2, config.embedding_dim, config.patch_count]
    assert observed['position_encoding'] == [2, config.patch_count, config.embedding_dim]
    assert observed['bilstm'] == [2, config.patch_count, 2*config.hidden_size]
    assert observed['h_n'] == [2*config.recurrent_layers, 2, config.hidden_size]
    assert observed['c_n'] == observed['h_n']
    assert observed['mean_pool'] == [2, 2*config.hidden_size]
    assert prediction.shape == (2, config.geometry_dim)
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters())
              for name in ('patch_embedding', 'bilstm', 'final_norm', 'regressor')}
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters()) == saved['metadata']['model_parameters']
    return config, {
        'model_config': asdict(config), 'module_shapes': observed,
        'wavelength_range_nm': [saved['metadata']['wavelength_nm'][0], saved['metadata']['wavelength_nm'][-1]],
        'parameter_count': sum(counts.values()), 'parameter_counts': counts,
        'checkpoint_sha256': checksum(checkpoint),
        'model_source_sha256': checksum(ROOT/'model.py'),
        'probe': 'CPU eval/inference_mode; two synthetic zero spectra only for shape verification',
        'training_performed': False,
    }


def draw(c, info, output):
    if (c.spectrum_points, c.patch_count, c.embedding_dim, c.hidden_size,
        c.recurrent_layers, c.mlp_dim, c.geometry_dim, c.position_encoding) != (1000, 100, 128, 64, 2, 256, 7, True):
        raise ValueError('This layout illustrates the published 100-sample model configuration')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 10.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 160), ylim=(0, 105))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def text(x, y, value, size=11, weight='normal', color=INK, ha='center'):
        return ax.text(x, y, value, ha=ha, va='center', fontsize=size, fontweight=weight,
                       color=color, linespacing=1.4, zorder=5)

    def box(x, y, w, h, value, fill='blue', size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1',
                                   facecolor=COLORS.get(fill, fill), edgecolor=COLORS['border'],
                                   linewidth=1.05, zorder=2))
        text(x+w/2, y+h/2, value, size)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                    mutation_scale=12, linewidth=1.3, color=color, zorder=3))

    def panel(x, w, title):
        ax.add_patch(FancyBboxPatch((x, 12), w, 59, boxstyle='round,pad=0.2,rounding_size=1.2',
                                   facecolor='#FAFBFD', edgecolor=COLORS['border'], linewidth=1, zorder=0))
        text(x+w/2, 67.5, title, 13, 'bold')

    text(80, 100.8, 'BiLSTM 光谱逆向模型 · 已训练实现', 23, 'bold')
    text(80, 96.1, f"参数量 {info['parameter_count']:,}  |  B 表示批量大小  |  R(λ) → 七个结构参数", 12, color=MUTED)
    band = info['wavelength_range_nm']
    stages = [
        (4, 20, f'原始反射光谱\n{band[0]:g}–{band[1]:g} nm\n[B, 1000]', 'blue'),
        (29, 23, '① 分块与投影\n100 块 × 10 点\n[B, 100, 128]', 'purple'),
        (57, 22, '加入位置编码\n固定 sin / cos\n[B, 100, 128]', 'purple'),
        (84, 24, '② BiLSTM × 2 层\n每方向 64 维\n[B, 100, 128]', 'orange'),
        (113, 20, '③ 归一化与池化\nLayerNorm → Mean\n[B, 128]', 'blue'),
        (138, 18, '回归头\n128 → 256 → 7\n[B, 7]', 'green'),
    ]
    for i, (x, w, label, fill) in enumerate(stages):
        box(x, 80, w, 12, label, fill, 10.5)
        if i < len(stages)-1:
            arrow(x+w+.3, 86, stages[i+1][0]-.3, 86)
    text(80, 75.8, '光谱顺序在分块后保留；双向 LSTM 沿 100 个分块递推，每条光谱使用独立的初始状态。', 10.5, color=MUTED)

    panel(4, 44, '① 光谱分块与嵌入')
    box(9, 53.5, 34, 9, '1000 个点 → 100 个不重叠分块\n每块包含 10 个反射率值', 'blue', 10.5)
    arrow(26, 53.1, 26, 49.7)
    box(9, 39.5, 34, 10, '共享线性投影：10 → 128\nConv1d：1 → 128\nkernel = stride = 10', 'purple', 10.5)
    arrow(26, 39.1, 26, 35.7)
    box(9, 25.5, 34, 10, '分块向量 + 正弦 / 余弦位置编码\n分块位置 p = 0, …, 99\n[B, 100, 128]', 'purple', 10.5)
    text(26, 18.2, '这是分块投影，未添加卷积特征前端。', 9.5, color=MUTED)

    panel(53, 61, '② 两层双向 LSTM：每层均拼接两个方向')
    text(83.5, 61.5, '输入：z1, z2, …, z100（每个为 128 维）', 10.5)
    for base, number in ((43.7, 1), (18.7, 2)):
        box(57, base, 53, 13, '', 'orange')
        text(62, base+10, f'第 {number} 层', 10, 'bold')
        text(70, base+8.1, '正向：短波 → 长波', 10)
        arrow(88, base+8.1, 106, base+8.1, '#247A62')
        text(70, base+3.7, '反向：长波 → 短波', 10)
        arrow(106, base+3.7, 88, base+3.7, '#AB6B22')
    arrow(83.5, 60, 83.5, 57.1)
    arrow(83.5, 43.3, 83.5, 40.9)
    box(57, 33.9, 53, 6.5, '每位置拼接 64 + 64 = 128 维\nDropout p=0.1（仅两层之间，训练时启用）', 'gray', 10)
    arrow(83.5, 33.5, 83.5, 32.1)
    text(83.5, 14.8, '输出完整序列 [B, 100, 128]，用于后续池化', 10, color=MUTED)

    panel(119, 37, '③ 聚合与七参数回归')
    box(123, 53.5, 29, 9, 'LayerNorm(128)\n每个位置的特征归一化', 'blue', 10.5)
    arrow(137.5, 53.1, 137.5, 50.2)
    box(123, 41, 29, 9, '沿 100 个位置取平均\n[B, 100, 128] → [B, 128]', 'blue', 10)
    arrow(137.5, 40.6, 137.5, 37.7)
    box(123, 27.5, 29, 10, 'Linear 128 → 256\nGELU\nLinear 256 → 7', 'green', 10.5)
    arrow(137.5, 27.1, 137.5, 24.2)
    box(123, 14.7, 29, 9, 'Sigmoid → [0, 1]\n七个归一化结构参数', 'green', 10.5)

    text(80, 8.3, '输出：[P, L, H, dX, dY, θ奇, θ偶]；前五项为 nm，后两项为 deg，物理单位还原在网络外完成。', 10.5)
    counts = info['parameter_counts']
    text(80, 4.3, f"参数：分块投影 {counts['patch_embedding']:,}  |  BiLSTM {counts['bilstm']:,}  |  LayerNorm {counts['final_norm']:,}  |  回归头 {counts['regressor']:,}", 10, color=MUTED)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'bilstm_architecture.{suffix}', dpi=180, facecolor='white')
        if suffix == 'svg':
            svg_path = output/'bilstm_architecture.svg'
            svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',
                                encoding='utf-8', newline='\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'models/au100_seed42.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/figures')
    args = parser.parse_args()
    config, info = inspect(args.checkpoint)
    draw(config, info, args.output)
    info['diagram_source_sha256'] = checksum(Path(__file__))
    (args.output/'bilstm_architecture.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'parameter_count': info['parameter_count'],
                      'verified_shapes': info['module_shapes']}, indent=2))


if __name__ == '__main__':
    main()
