import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
data_dir = "help-docs"
output_dir = "database"
resource_file = "command_resources"

quantization_config = {
    "quant_method": "awq",
    "zero_point": True,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
}   