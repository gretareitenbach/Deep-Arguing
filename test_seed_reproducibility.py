import yaml
with open("tuning/mnist/data_mnist.yaml", "r") as f:
    config = yaml.safe_load(f)
print("Finished checking")
