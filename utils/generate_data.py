import pandas as pd
import numpy as np

data = []

for i in range(500):
    vibration = np.random.uniform(0.1, 1.0)
    tilt = np.random.randint(1, 10)
    crack_width = np.random.uniform(0.01, 0.6)
    rainfall = np.random.randint(20, 200)
    temperature = np.random.randint(20, 40)

    rockfall = 1 if (
        vibration > 0.6 or
        crack_width > 0.3 or
        rainfall > 120
    ) else 0

    data.append([vibration, tilt, crack_width, rainfall, temperature, rockfall])

df = pd.DataFrame(data, columns=[
    "vibration","tilt","crack_width","rainfall","temperature","rockfall"
])

df.to_csv("data/rockfall_dataset.csv", index=False)

print("Dataset generated successfully (500 rows)")