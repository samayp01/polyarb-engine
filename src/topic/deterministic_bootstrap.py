"""
Deterministic bootstrap for reproducible experiments.

Sets environment variables and seeds for deterministic behavior across
numpy, sklearn, torch, and BLAS libraries.
"""

import os
import random


def init_deterministic(seed=42):
  """Initialize deterministic behavior for reproducibility."""
  # Set BLAS/LAPACK threading to single thread for determinism
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"

  # Seed Python random
  random.seed(seed)

  # Seed numpy
  try:
    import numpy as np
    np.random.seed(seed)
  except ImportError:
    pass

  # Seed torch if available
  try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  except ImportError:
    pass
