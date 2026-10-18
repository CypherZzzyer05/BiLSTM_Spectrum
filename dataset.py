"""PyTorch dataset adapter for real FDTD spectra -> normalized geometry labels."""
# 数据桥接层：读取真实反射光谱、五个尺寸及一个公共角或两个象限角。
# spectra 使用原始反射率；结构参数按各自采样上下界归一化到 0～1。

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from schema import PARAMETER_NAMES, ROTATION_PARAMETER_NAMES, QUADRANT_PARAMETER_NAMES


class SpectrumGeometryDataset(Dataset):
    def __init__(self, path, split="train", allow_validation_data=False):
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be train, val, test or all")
        with np.load(Path(path), allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata_json"].item()))
            # 标签顺序必须与网络输出一致，避免把 p 的预测与 l 等其他尺寸比较。
            self.parameter_names = tuple(str(name) for name in data["parameter_names"])
            if self.parameter_names not in (PARAMETER_NAMES, ROTATION_PARAMETER_NAMES, QUADRANT_PARAMETER_NAMES):
                raise ValueError("Dataset geometry parameter order differs from BiLSTM")
            rotated = len(self.parameter_names) > 5
            self.parameter_units = tuple(str(unit) for unit in data["parameter_units"]) if rotated else ("nm",) * 5
            if self.parameter_units != ("nm",) * 5 + ("deg",) * (len(self.parameter_names)-5):
                raise ValueError("Dataset parameter units must be five nm lengths and optional degree angle")
            if self.metadata["config"]["quality_level"] == "pipeline_validation" and not allow_validation_data:
                raise ValueError("This is a pipeline-validation dataset. Explicitly allow it for a smoke test only.")
            # pipeline_validation 表示粗网格工程流程验证数据，显式授权后才允许读取。
            # 该标记与 train/val/test 的划分无关；训练入口另外严格检查 status=2。
            spectra = data["spectra"].copy()
            labels = data["geometry_01"].copy()
            self.lower_nm = data["lower_nm"].copy()
            self.upper_nm = data["upper_nm"].copy()
            # 混合单位标签单独保存，geometry_nm/lower_nm/upper_nm 始终仅含五个长度。
            self.lower_values = data["lower_values"].copy() if rotated else self.lower_nm.copy()
            self.upper_values = data["upper_values"].copy() if rotated else self.upper_nm.copy()
            self.wavelength_nm = data["wavelength_nm"].copy()
            self.sample_ids = data["sample_ids"].copy()
            self.geometry_nm = data["geometry_nm"].copy()
            self.geometry_values = data["geometry_values"].copy() if rotated else self.geometry_nm.copy()
            if self.parameter_names == ROTATION_PARAMETER_NAMES and (not np.array_equal(self.geometry_values[:, :5], self.geometry_nm)
                            or not np.array_equal(self.geometry_values[:, 5], data["rotation_deg"])):
                raise ValueError("Mixed-unit geometry disagrees with length/angle fields")
            if rotated and not np.array_equal(self.geometry_values[:, :5], self.geometry_nm):
                raise ValueError("Mixed-unit geometry disagrees with lengths")
            # 复用文件中已保存的划分索引；all 仅用于完整数据检查或结果展示。
            indices = np.arange(len(spectra)) if split == "all" else data[split + "_indices"].copy()
            if np.any(data["status"] != 2) and not allow_validation_data:
                raise ValueError("Dataset contains simulations without auto-shutoff convergence")
        dimensions = len(self.parameter_names)
        if spectra.shape != (len(labels), 1000) or labels.shape != (len(spectra), dimensions):
            raise ValueError(f"Expected spectra [N,1000] and labels [N,{dimensions}]")
        if self.geometry_nm.shape != (len(spectra), 5) or self.geometry_values.shape != (len(spectra), dimensions):
            raise ValueError("Physical geometry shapes differ from the parameter schema")
        if (self.lower_nm.shape != (5,) or self.upper_nm.shape != (5,)
                or not np.array_equal(self.lower_values[:5], self.lower_nm)
                or not np.array_equal(self.upper_values[:5], self.upper_nm)):
            raise ValueError("Mixed-unit bounds disagree with the five length bounds")
        if not np.isfinite(spectra).all() or not np.isfinite(labels).all():
            raise ValueError("Dataset contains nonfinite spectra or labels")
        if np.any(labels < 0) or np.any(labels > 1):
            raise ValueError("Geometry labels must be in [0,1]")
        if (self.lower_values.shape != (dimensions,) or self.upper_values.shape != (dimensions,)
                or not np.isfinite(self.lower_values).all() or not np.isfinite(self.upper_values).all()
                or np.any(self.upper_values <= self.lower_values)):
            raise ValueError("Invalid normalization bounds")
        if rotated and (np.any(self.lower_values[5:] != -45) or np.any(self.upper_values[5:] != 45)):
            raise ValueError("Rotation labels must use [-45,45] degree bounds")
        # 上下界来自采样配置，而不是由训练集/验证集各自重新估计。
        # 再计算一次归一化标签，确认保存的物理尺寸与训练目标确实一一对应。
        expected = (self.geometry_values - self.lower_values) / (self.upper_values - self.lower_values)
        if not np.allclose(labels, expected, atol=1e-7):
            raise ValueError("Physical and normalized labels disagree")
        if self.wavelength_nm.shape != (1000,) or not np.all(np.diff(self.wavelength_nm) > 0):
            raise ValueError("Wavelength grid must have 1000 increasing points")
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(spectra)):
            raise ValueError("Invalid split indices")
        # 按固定划分切片后转成 float32；标签为五、六或七列。
        # DataLoader 堆叠为 [B,1000] 和 [B,D]；这里不做 token 分块。
        self.spectra = torch.from_numpy(spectra[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.float32))
        self.geometry_nm = self.geometry_nm[indices]
        self.geometry_values = self.geometry_values[indices]
        self.sample_ids = self.sample_ids[indices]

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        # 交给网络的是反射率序列及归一化尺寸；波长网格作为元信息另行保存。
        return self.spectra[index], self.labels[index]

    def denormalize(self, geometry_01):
        # 支持 Tensor 和 NumPy；Tensor 的上下界与预测保持相同设备及数值类型。
        if torch.is_tensor(geometry_01):
            lower = torch.as_tensor(self.lower_values, dtype=geometry_01.dtype, device=geometry_01.device)
            upper = torch.as_tensor(self.upper_values, dtype=geometry_01.dtype, device=geometry_01.device)
        else:
            lower, upper = self.lower_values, self.upper_values
        # 前五列还原为 nm；一个或两个角度列还原为 deg。
        return lower + geometry_01 * (upper - lower)
