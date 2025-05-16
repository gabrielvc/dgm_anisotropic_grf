import os
from pathlib import Path
import json


def save_experiment(save_folder, results, log_data):

    os.makedirs(save_folder, exist_ok=True)

    run_number = 1
    while os.path.exists(os.path.join(save_folder, f"run{run_number}")):
        run_number += 1

    run_folder = save_folder / f"run{run_number}"
    os.makedirs(run_folder)
    results_path = os.path.join(run_folder, "results.json")
    log_path = os.path.join(run_folder, "log_data.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
