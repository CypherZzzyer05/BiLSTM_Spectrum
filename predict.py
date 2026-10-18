"""Predict geometry for one saved spectrum using a standalone BiLSTM checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import BiLSTMConfig, InverseBiLSTM


def predict(checkpoint_path, dataset_path, sample_id):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if metadata.get("model_family") != "standalone_bilstm":
        raise ValueError("Checkpoint is not from this standalone BiLSTM")
    model = InverseBiLSTM(BiLSTMConfig.from_dict(metadata["config"]["model"])).eval()
    model.load_state_dict(saved["model_state"])
    with np.load(dataset_path, allow_pickle=False) as d:
        rows = np.flatnonzero(d["sample_ids"] == sample_id)
        if len(rows) != 1:
            raise ValueError("Sample ID must identify exactly one spectrum")
        if not np.array_equal(d["wavelength_nm"], np.array(metadata["wavelength_nm"])):
            raise ValueError("Spectrum wavelength grid differs from training")
        spectrum = d["spectra"][int(rows[0])].astype(np.float32)
    if not np.isfinite(spectrum).all():
        raise ValueError("Spectrum contains nonfinite values")
    with torch.inference_mode():
        normalized = model(torch.from_numpy(spectrum)[None])[0].numpy()
    lower = np.array(metadata["lower_values"])
    upper = np.array(metadata["upper_values"])
    physical = lower + normalized*(upper-lower)
    return {
        "sample_id": sample_id,
        "prediction": {name: {"value": float(value), "unit": unit}
                       for name, value, unit in zip(metadata["parameter_names"], physical, metadata["parameter_units"])},
        "normalized_prediction": normalized.tolist(),
        "training_dataset_sha256": metadata["dataset_sha256"],
        "quality_level": metadata["quality_level"],
        "note": "One candidate geometry; no FDTD verification of this prediction has been performed.",
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "models/au100_seed42.pt")
    parser.add_argument("--dataset", type=Path, default=root / "examples/au100_metal8_100/dataset.npz")
    parser.add_argument("--sample-id", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.dataset, args.sample_id), indent=2))


if __name__ == "__main__":
    main()
