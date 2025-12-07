import json
import pickle
import os
import glob
from pathlib import Path
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

def load_history_data(filepath: str) -> Dict[str, Any]:
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")

def get_run_name(filepath: Path, config: Dict[str, Any]) -> str:
    """
    Creates a concise run name for the folder structure.
    We generally don't put the whole 'note' here to avoid invalid file paths.
    """
    run_name = filepath.stem
    
    # Add key parameters to the name for easy filtering in the UI
    keys_to_include = ['algorithm', 'participation_ratio']
    params = []
    for key in keys_to_include:
        if key in config:
            short_key = key.replace('participation_', '')
            params.append(f"{short_key}={config[key]}")
            
    if params:
        run_name = f"{run_name}_" + "_".join(params)
    return run_name

def write_experiment_to_tensorboard(data: Dict[str, Any], run_name: str, base_log_dir: str):
    log_dir = os.path.join(base_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Writing logs to: {log_dir}")

    # =========================================================
    # NEW: Log 'note' and 'Config' to the Text Tab
    # =========================================================
    config = data.get('Config', {})
    
    # 1. Extract the note
    experiment_note = config.get('note', 'No description provided.')
    
    # 2. format everything as Markdown
    # We create a bold header for the note, then a table for the rest of the config
    md_text = f"### Experiment Note\n> **{experiment_note}**\n\n"
    md_text += "### Configuration Details\n"
    md_text += "| Parameter | Value |\n|---|---|\n"
    
    for key, value in config.items():
        if key == 'note': continue # We already displayed the note at the top
        md_text += f"| {key} | {value} |\n"

    # 3. Write to TensorBoard
    # 'global_step=0' ensures it appears at the start
    writer.add_text("Experiment Info", md_text, global_step=0)
    # =========================================================

    # Helper to write list of (round, value) tuples
    def log_tuples(tag, tuple_list):
        if not tuple_list: return
        for item in tuple_list:
            # Handle potential [round, value] (JSON) or (round, value) (Pickle)
            if len(item) == 2:
                r, v = item
                if isinstance(v, (int, float)):
                    writer.add_scalar(tag, v, r)

    # Log Losses
    log_tuples("Loss/Distributed", data.get('losses_distributed', []))
    log_tuples("Loss/Centralized", data.get('losses_centralized', []))

    # Log Metrics
    metric_groups = {
        "Metrics_Distributed_Fit": data.get('metrics_distributed_fit', {}),
        "Metrics_Distributed_Eval": data.get('metrics_distributed', {}),
        "Metrics_Centralized": data.get('metrics_centralized', {})
    }

    for group_name, metrics_dict in metric_groups.items():
        if not metrics_dict: continue
        for metric_name, values in metrics_dict.items():
            if metric_name != 'similarity_scores':
                for item in values:
                    if len(item) == 2:
                        r, v = item
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"{group_name}/{metric_name}", v, r)

    writer.flush()
    writer.close()

def main():
    input_folder = "./results (client evaluation on val loader)" 
    tensorboard_dir = "./runs" 

    # Find files
    files = glob.glob(os.path.join(input_folder, "*.json")) + \
            glob.glob(os.path.join(input_folder, "*.pkl"))

    if not files:
        print(f"No files found in {input_folder}")
        return

    for file_path in files:
        try:
            data = load_history_data(file_path)
            config = data.get('Config', {})
            run_name = get_run_name(Path(file_path), config)
            write_experiment_to_tensorboard(data, run_name, tensorboard_dir)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\nDone. Run: tensorboard --logdir=./runs")

if __name__ == "__main__":
    main()