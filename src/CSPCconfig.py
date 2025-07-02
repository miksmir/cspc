import json
from pathlib import Path

"""
Loads global config variables from "../config/config.json"
"""

# Load JSON only once from current working directory (cspc) into cspc/config
with open(Path(__file__).parent.parent / "config" / "config.json") as f:
    CONFIG = json.load(f)

# Directory of original point cloud
INPUT_PATH_LAS = str(Path(CONFIG["input_path_pclas"]))

# Directory of output (reconstructed) point cloud
OUTPUT_PATH_LAS = str(Path(CONFIG["output_path_pclas"]))

# Directory of output (reconstructed) point cloud plots
OUTPUT_PATH_PLOTS = str(Path(CONFIG["output_path_plots"]))
