"""PyCharm viewer for saved training curves and held-out geometry predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from schema import PARAMETER_NAMES, PROJECT_ROOT


def latest_run(output_root=None):
    root = Path(output_root).resolve() if output_root else PROJECT_ROOT / 'runs/au100_bilstm_100'
    pointer = root / 'latest_run.json'
    return Path(json.loads(pointer.read_text(encoding='utf-8'))['run_dir'])


def plot_training(run_dir, show=True, save=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    history = json.loads((run_dir / 'history.json').read_text(encoding='utf-8'))
    report = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    parameter_names = report.get('parameter_names', list(PARAMETER_NAMES))
    rotated = len(parameter_names) == 6
    mirror = report['config']['model'].get('angle_symmetry', 'none') == 'square_mirror'
    predictions = np.genfromtxt(run_dir / 'test_predictions.csv', delimiter=',', names=True)
    predictions = np.atleast_1d(predictions)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='#f5f7fb')
    fig.canvas.manager.set_window_title('BiLSTM | 训练结果')
    fig.subplots_adjust(left=.08, right=.95, top=.85, bottom=.09, hspace=.4, wspace=.3)
    counts = {name: len(ids) for name, ids in report['split_sample_ids'].items()}
    model_name = 'Standalone BiLSTM'
    fig.suptitle(f'{model_name} · {sum(counts.values())} 个真实 FDTD 样本训练', x=.08, ha='left', y=.965,
                 fontsize=19, fontweight='bold', color='#152849')
    fig.text(.08, .905, f"{report['device'].upper()} · 训练/验证/测试 {counts['train']}/{counts['val']}/{counts['test']}"
             f" · 完成 {report['epochs_completed']} 轮 · 最佳验证模型：第 {report['best_epoch']} 轮", color='#50617c')
    epochs = [h['epoch'] for h in history]
    ax = axes[0, 0]
    ax.plot(epochs, [h['train_mse'] for h in history], color='#2667d8', label='训练（含 dropout）')
    ax.plot(epochs, [h['val_mse'] for h in history], color='#049b86', label='验证')
    ax.axvline(report['best_epoch'], color='#d58b23', ls='--', label='选中模型')
    ax.set(title='训练过程', xlabel='Epoch', ylabel='归一化尺寸与等效角 MSE' if mirror else '归一化参数 MSE')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    x = np.arange(5)
    ax.bar(x - .18, [report['metrics']['test']['mae_nm'][p] for p in PARAMETER_NAMES], .35,
           color='#2667d8', label='BiLSTM')
    ax.bar(x + .18, [report['train_mean_baseline']['test']['mae_nm'][p] for p in PARAMETER_NAMES], .35,
           color='#b6c3d9', label='训练集均值基线')
    ax.set(title='留出测试集：各参数绝对误差', ylabel='MAE (nm)', xticks=x, xticklabels=PARAMETER_NAMES)
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    colors = ('#2667d8', '#049b86', '#d58b23', '#8b68bd', '#d96779', '#9d631c', '#146d7a')
    for p, color in zip(parameter_names, colors):
        truth, predicted = predictions['true_01_' + p], predictions['pred_01_' + p]
        if p == 'rotation_deg' and mirror:
            truth, predicted = np.abs(2 * truth - 1), np.abs(2 * predicted - 1)
        ax.scatter(truth, predicted, s=30,
                   color=color, label='|θ| / 45°' if p == 'rotation_deg' and mirror else p, alpha=.8)
    ax.plot([0, 1], [0, 1], color='#7f8ba0', ls='--', lw=1)
    ax.set(title='测试集：真实值与预测值', xlabel='真实参数 [0,1]', ylabel='预测参数 [0,1]',
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(ncol=3 if rotated else 5, fontsize=8, loc='upper center')
    axes[1, 1].axis('off')
    model_mse = report['metrics']['test']['normalized_mse']
    baseline = report['train_mean_baseline']['test']['normalized_mse']
    text = (f"结果记录\n\n测试 MSE：{model_mse:.6f}\n训练均值基线 MSE：{baseline:.6f}\n"
            f"模型参数：{report['model_parameters']:,}\n实际优化步骤：{report['optimizer_steps']}\n"
            f"停止原因：{report['stop_reason']}\n\n"
            '仅用验证集选择模型，测试集用于最终评估。\n'
            f"数据级别：{report['quality_level']}。\n"
            f"测试样本 {counts['test']} 个，结果不代表论文精度。\n\n"
            f"已保存 best.pt、训练日志和预测 CSV。\n运行：{run_dir.name}")
    if rotated:
        angle = report['metrics']['test']['mae_deg']['rotation_deg']
        raw = report['metrics']['test']['raw_signed_mae_deg']
        text = (f"六参数结果\n\n测试 MSE：{model_mse:.6f}；基线：{baseline:.6f}\n"
                f"等效角 MAE：{angle:.2f}°\n原始有符号角 MAE：{raw:.2f}°\n"
                "当前输入下 ±θ 互为镜像等效候选。\n长度柱图单位 nm，角度另用 deg。\n\n"
                f"参数量：{report['model_parameters']:,}；优化步骤：{report['optimizer_steps']}\n"
                f"停止原因：{report['stop_reason']}\n数据级别：{report['quality_level']}\n"
                f"测试 {counts['test']} 个样本，精度需更多数据验证。\n"
                f"运行：{run_dir.name}")
    if len(parameter_names) == 7:
        angle_errors = report['metrics']['test']['mae_deg']
        text = (f"七参数结果\n\n测试 MSE：{model_mse:.6f}；基线：{baseline:.6f}\n"
                f"奇数象限角 MAE：{angle_errors['theta_odd_deg']:.2f}°\n"
                f"偶数象限角 MAE：{angle_errors['theta_even_deg']:.2f}°\n"
                "角度误差按正方形 90° 周期计算。\n\n"
                f"优化步骤：{report['optimizer_steps']}\n"
                f"数据级别：{report['quality_level']}\n运行：{run_dir.name}")
    axes[1, 1].text(0, 1, text, va='top', color='#334969', linespacing=1.3 if rotated else 1.5)
    output = run_dir / 'training_preview.png'
    if save:
        fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
        print(f'Training preview: {output}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Run directory; defaults to latest saved training run')
    parser.add_argument('--output-root', type=Path, help='Training output root containing latest_run.json')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-save', action='store_true', help='View an archived result without rewriting its image')
    args = parser.parse_args()
    plot_training(args.run or latest_run(args.output_root), show=not args.no_show, save=not args.no_save)


if __name__ == '__main__':
    main()
