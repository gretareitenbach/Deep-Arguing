import numpy as np
import pandas as pd

np.random.seed(123)

# Number of samples
n = 10_000

# Generate synthetic features
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


def classify(row):
    score = 0

    # DEFAULT RULE
    score += 0.1

    # LEVEL 1 RULES
    if row.temperature > 18 and row.temperature < 30:
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

    # LEVEL 3 EXCEPTIONS TO EXCEPTIONS
    if row.cloud_cover > 0.75 and row.uv_index > 7 and row.wind_speed > 14:
        score -= 1.0

    if row.precipitation > 8 and row.visibility > 9 and row.humidity > 80:
        score -= 0.5

    # ADDITIONAL PRIORITY-ORDERED RULES
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


data["class"] = data.apply(classify, axis=1)

file_path = "data/defeasible/defeasible.csv"
data.to_csv(file_path, index=False)

print(data["class"].value_counts())
