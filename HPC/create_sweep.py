import wandb

# ### Train Model

sweep_config = {
    "method": "grid",
    # "method": "random",
    "name": "fixed_support",
    "metric": {"goal": "maximize", "name": "f1_percentage"},
    "parameters": {
        "epochs": {"values": [3000]},
        "max_iters": {"values": [95]}, # 95 = len of iris casebase
        "use_symmetric_attacks": {"values": [False]},
        "lr": {"values": [2e-2]},
        "temperature": {"values": [5e-2]},
        "bs_temperature": {"values": [5e-1, 5e-2, 5e-3, 5e-4]},
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

sweep_id = wandb.sweep(sweep=sweep_config, project="gradual-aa-cbr-iris")

print(f"Sweep ID: {sweep_id}")