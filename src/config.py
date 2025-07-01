import json
from pathlib import Path

"""
Loads global config variables from "../config/config.json"
"""

# Load JSON only once from current working directory (cspc) into cspc/config
with open(Path(__file__).parent.parent / "config" / "config.json") as f:
    CONFIG = json.load(f)

INPUT_PATH_LAS = str(Path(CONFIG["input_path_pclas"]))
OUTPUT_PATH_LAS = str(Path(CONFIG["output_path_pclas"]))
OUTPUT_PATH_PLOTS = str(Path(CONFIG["output_path_plots"]))
