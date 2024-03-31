import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
data_dir = "help-docs"
output_dir = "database"
resource_file = os.path.join(data_dir, "command_resources")