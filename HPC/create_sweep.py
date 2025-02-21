import wandb
import argparse

# ### Train Model

sweep_config = {
    "method": "grid",
    # "method": "random",
    "name": "changed targets",
    "metric": {"goal": "maximize", "name": "f1_percentage"},
    "parameters": {
        "epochs": {"values": [6000]},
        "max_iters": {"values": [95]}, # 95 = len of iris casebase
        "use_symmetric_attacks": {"values": [False]},
        "lr": {"values": [2e-2]},
        "temperature": {"values": [5e-2]},
        "use_blockers": {"values": [True]},
        "initialisation_method": {"values": ["Xavier_uniform"]},
        "alpha": {"values": [0]},
        "beta": {"values": [0]},
        "gamma": {"values": [5e-3]},
        "gamma_prime": {"values": [5e-3]},
        "post_process_func": {"values": ["uni_directional"]},
        "use_supports": {"values": [True]},
    }
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for project configuration.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
        help="dataset name"
    )


    args = parser.parse_args()

    sweep_id = wandb.sweep(sweep=sweep_config,  project=f"gradual-aa-cbr-{args.dataset}")

    print(f"Sweep ID: {sweep_id}")