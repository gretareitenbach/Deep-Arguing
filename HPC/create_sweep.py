import wandb

# ### Train Model

sweep_config = {
    "method": "grid",
    # "method": "random",
    "name": "best_params_support",
    "metric": {"goal": "maximize", "name": "f1_percentage"},
    "parameters": {
        "epochs": {"values": [4500]},
        "max_iters": {"values": [95]}, # 95 = len of iris casebase
        "use_symmetric_attacks": {"values": [False]},
        "lr": {"values": [2e-3]},
        "temperature": {"values": [5e-2]},
        "use_blockers": {"values": [True]},
        "initialisation_method": {"values": ["Xavier_uniform"]},
        "alpha": {"values": [0]},
        "beta": {"values": [0]},
        "gamma": {"values": [5e-2, 5e-3, 5e-4, 5e-8]},
        "gamma_prime": {"values": [5e-2, 5e-3, 5e-4, 5e-8]},
        "post_process_func": {"values": ["uni_directional"]},
        "use_supports": {"values": [True]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="gradual-aa-cbr-iris")

print(f"Sweep ID: {sweep_id}")