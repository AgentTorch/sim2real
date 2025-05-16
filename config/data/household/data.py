import pandas as pd
import numpy as np

# Configuration
NUM_FACILITIES = 3435
OUTPUT_PREFIX = 'facility'

# Boston bounding box (approx.)
lat_min, lat_max = 42.23, 42.40
lon_min, lon_max = -71.18, -70.92

# Generate fake data
np.random.seed(42)
ids = np.arange(1, NUM_FACILITIES + 1)
latitudes = np.random.uniform(lat_min, lat_max, NUM_FACILITIES)
longitudes = np.random.uniform(lon_min, lon_max, NUM_FACILITIES)
has_solar = np.random.choice([True, False], NUM_FACILITIES)
adoption_propensity = np.random.rand(NUM_FACILITIES)
annual_savings = np.zeros(NUM_FACILITIES)

# Build DataFrame
df = pd.DataFrame({
    'id': ids,
    'latitude': latitudes,
    'longitude': longitudes,
    'has_solar': has_solar,
    'adoption_propensity': adoption_propensity,
    'annual_savings': annual_savings
})

# Export each column to its own CSV (no header, one value per line)
columns = ['id', 'latitude', 'longitude', 'has_solar', 'adoption_propensity', 'annual_savings']
for col in columns:
    filename = f"{col}.csv"
    # For booleans, write as True/False strings
    if df[col].dtype == bool or df[col].dtype == object:
        df[col].astype(str).to_csv(filename, index=False, header=False)
    else:
        df[col].to_csv(filename, index=False, header=False)
    print(f"Generated {filename}")

