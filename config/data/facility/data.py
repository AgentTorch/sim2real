import pandas as pd

# Configuration
INPUT_CSV = 'data.csv'  # path to the input file

# Output files
LATITUDE_CSV = 'latitude.csv'
LONGITUDE_CSV = 'longitude.csv'
SOLAR_CSV = 'solar_production_kwh.csv'
DEMAND_CSV = 'energy_demand_kwh.csv'

# Read and clean input
raw = pd.read_csv(INPUT_CSV)
clean = raw[pd.to_numeric(raw['year'], errors='coerce').notnull()].copy()
clean['month'] = clean['month'].astype(int)
clean['latitude'] = clean['latitude'].astype(float)
clean['longitude'] = clean['longitude'].astype(float)
clean['solar_production_kwh'] = pd.to_numeric(clean['solar_production_kwh'], errors='coerce').fillna(0.0)
clean['energy_demand_kwh'] = pd.to_numeric(clean['energy_demand_kwh'], errors='coerce').fillna(0.0)

# Identify unique facilities
facilities = clean[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
facilities.insert(0, 'id', range(1, len(facilities) + 1))

# Function to pick first non-zero value or zero
def first_nonzero(series):
    nonzeros = series[series != 0]
    return nonzeros.iloc[0] if not nonzeros.empty else 0.0

# Solar: for each facility and month, pick first non-zero value across years
solar_grouped = (
    clean.groupby(['latitude', 'longitude', 'month'])['solar_production_kwh']
         .apply(first_nonzero)
         .reset_index()
)
# Pivot to months 1-12
solar_pivot = (
    solar_grouped.pivot(index=['latitude', 'longitude'], columns='month', values='solar_production_kwh')
    .reindex(columns=range(1, 13), fill_value=0)
)
# Merge IDs
solar = facilities.merge(solar_pivot.reset_index(), on=['latitude', 'longitude'], how='left').sort_values('id')
# Build list of month arrays
solar_array = solar.loc[:, range(1, 13)].values.tolist()

# Demand: mean value per month across all years
demand_pivot = (
    clean.pivot_table(
        index=['latitude', 'longitude'],
        columns='month',
        values='energy_demand_kwh',
        aggfunc='mean'
    )
    .reindex(columns=range(1, 13), fill_value=0)
)
demand = facilities.merge(demand_pivot.reset_index(), on=['latitude', 'longitude'], how='left').sort_values('id')
demand_array = demand.loc[:, range(1, 13)].values.tolist()

# Export latitude & longitude
facilities['latitude'].to_csv(LATITUDE_CSV, index=False, header=False)
facilities['longitude'].to_csv(LONGITUDE_CSV, index=False, header=False)

# Export solar and demand arrays (12 comma-separated values per line)
pd.DataFrame(solar_array, columns=range(1, 13)).to_csv(SOLAR_CSV, index=False, header=False)
pd.DataFrame(demand_array, columns=range(1, 13)).to_csv(DEMAND_CSV, index=False, header=False)

print("Conversion complete. Generated files:")
print(f" - {LATITUDE_CSV}")
print(f" - {LONGITUDE_CSV}")
print(f" - {SOLAR_CSV}")
print(f" - {DEMAND_CSV}")

