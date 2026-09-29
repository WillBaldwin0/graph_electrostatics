import os

# e3nn 0.4.4 loads its Wigner-matrix constants with torch.load, which fails under
# the weights_only default of torch >= 2.6 unless this variable is set.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
