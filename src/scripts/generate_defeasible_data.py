import numpy as np
import pandas as pd

np.random.seed(123)

# Number of samples
n = 10_000

# --- Generate synthetic features ---
data = pd.DataFrame(
    {
        "temperature": np.random.uniform(0, 40, n),
        "humidity": np.random.uniform(15, 95, n),
        "wind_speed": np.random.uniform(0, 20, n),
        "cloud_cover": np.random.uniform(0, 1, n),
        "pollution": np.random.uniform(5, 160, n),
        "uv_index": np.random.uniform(0, 11, n),
        "visibility": np.random.uniform(0.5, 12, n),
        "precipitation": np.random.uniform(0, 20, n),
        "pressure": np.random.uniform(970, 1040, n),
        "hour": np.random.uniform(0, 23, n),
        "green_space": np.random.uniform(0, 1, n),
    }
)


# --- Classification function ---
def classify(row):
    score = 0

    # DEFAULT RULE
    score += 0.1

    # LEVEL 1 RULES
    if 18 < row.temperature < 30:
        score += 2.0

    if row.cloud_cover > 0.75:
        score -= 1.5

    if row.precipitation > 8:
        score -= 1.0

    if row.pollution > 110:
        score -= 1.0

    # LEVEL 2 EXCEPTIONS
    if row.cloud_cover > 0.75 and row.uv_index > 7:
        score += 2.0

    if row.precipitation > 8 and row.visibility > 9:
        score += 1.8

    if row.pollution > 110 and row.green_space > 0.7:
        score += 2.0

    # LEVEL 3 EXCEPTIONS
    if row.cloud_cover > 0.75 and row.uv_index > 7 and row.wind_speed > 14:
        score -= 1.0

    if row.precipitation > 8 and row.visibility > 9 and row.humidity > 80:
        score -= 0.5

    # ADDITIONAL RULES
    if row.temperature < 16:
        score -= 1.0

    if 10 < row.temperature < 16 and row.wind_speed < 3:
        score += 1.5

    if row.visibility > 10:
        score += 2.5

    if row.pressure < 990:
        score -= 0.5

    # FINAL CLASSIFICATION
    if score >= 1:
        return "Favourable"
    elif score <= -1:
        return "Unfavourable"
    else:
        return "Neutral"


# --- STEP 1: Initial labels (clean) ---
data["class"] = data.apply(classify, axis=1)


# --- STEP 2: Gaussian noise ---
def add_gaussian_noise(df, noise_level=0.08):
    noisy_df = df.copy()

    for col in df.columns:
        if col != "class":
            std = df[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(df))
            noisy_df[col] += noise

    return noisy_df


data_noisy = add_gaussian_noise(data, noise_level=0.08)


# --- STEP 3: Structured noise ---
data_noisy["precipitation"] += data_noisy["humidity"] * 0.015
data_noisy["visibility"] -= data_noisy["wind_speed"] * 0.05

# Rare anomalies
mask_pollution = np.random.rand(n) < 0.02
data_noisy.loc[mask_pollution, "pollution"] *= 1.4

mask_visibility = np.random.rand(n) < 0.015
data_noisy.loc[mask_visibility, "visibility"] *= 0.6


# --- STEP 4: Clip to valid bounds ---
bounds = {
    "temperature": (0, 40),
    "humidity": (15, 95),
    "wind_speed": (0, 20),
    "cloud_cover": (0, 1),
    "pollution": (5, 160),
    "uv_index": (0, 11),
    "visibility": (0.5, 12),
    "precipitation": (0, 20),
    "pressure": (970, 1040),
    "hour": (0, 23),
    "green_space": (0, 1),
}

for col, (low, high) in bounds.items():
    data_noisy[col] = data_noisy[col].clip(low, high)



# --- STEP 5: Add label noise ---
flip_mask = np.random.rand(n) < 0.04
data_noisy.loc[flip_mask, "class"] = np.random.choice(
    ["Favourable", "Neutral", "Unfavourable"], size=flip_mask.sum()
)


# --- Save ---
file_path = "data/defeasible/defeasible_noisy.csv"
data_noisy.to_csv(file_path, index=False)

print(data_noisy["class"].value_counts())
