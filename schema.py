"""Parameter schema; independent of any simulator or Metaformer project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARAMETER_NAMES = ("p", "l", "h", "alpha_x", "alpha_y")
ROTATION_PARAMETER_NAMES = PARAMETER_NAMES + ("rotation_deg",)
QUADRANT_PARAMETER_NAMES = PARAMETER_NAMES + ("theta_odd_deg", "theta_even_deg")
