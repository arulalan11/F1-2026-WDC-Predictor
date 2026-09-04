import pandas as pd

DATA_DIR = "F1 Dataset"  
START_YEAR = 2014
END_YEAR = 2026

# 1. Load only the tables we actually need for the first version

races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv", na_values="\\N")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
status = pd.read_csv(f"{DATA_DIR}/status.csv")
driver_standings = pd.read_csv(f"{DATA_DIR}/driver_standings.csv", na_values="\\N")
constructor_standings = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv", na_values="\\N")
qualifying = pd.read_csv(f"{DATA_DIR}/qualifying.csv", na_values="\\N")

# 2. Filter races to the target window (2014-2026) 

races = races[(races["year"] >= START_YEAR) & (races["year"] <= END_YEAR)].copy()
race_cols = ["raceId", "year", "round", "circuitId", "name", "date"]
races = races[race_cols].rename(columns={"name": "race_name"})

# 3. Start from results (one row per driver per race) and keep only races within our window

df = results.merge(races, on="raceId", how="inner")

# 4. Bring in driver and constructor names

drivers_small = drivers[["driverId", "driverRef", "code", "forename", "surname", "dob", "nationality"]]
drivers_small = drivers_small.rename(columns={"nationality": "driver_nationality"})
df = df.merge(drivers_small, on="driverId", how="left")

constructors_small = constructors[["constructorId", "constructorRef", "name", "nationality"]]
constructors_small = constructors_small.rename(
    columns={"name": "constructor_name", "nationality": "constructor_nationality"}
)
df = df.merge(constructors_small, on="constructorId", how="left")

# 5. Decode DNF / finish status (statusId -> readable text + a clean flag)

df = df.merge(status, on="statusId", how="left")
df["dnf"] = ~df["status"].isin(["Finished"]) & ~df["status"].str.contains(r"^\+\d+ Lap", na=False)

# 6. Attach standings AFTER this race (points, position, wins so far)

ds = driver_standings[["raceId", "driverId", "points", "position", "wins"]].rename(
    columns={"points": "driver_points_after", "position": "driver_standing_pos_after", "wins": "driver_wins_after"}
)
df = df.merge(ds, on=["raceId", "driverId"], how="left")

cs = constructor_standings[["raceId", "constructorId", "points", "position", "wins"]].rename(
    columns={
        "points": "constructor_points_after",
        "position": "constructor_standing_pos_after",
        "wins": "constructor_wins_after",
    }
)
df = df.merge(cs, on=["raceId", "constructorId"], how="left")


# 7. Attach qualifying result (grid-deciding position + Q1/Q2/Q3 times)

qual = qualifying[["raceId", "driverId", "position", "q1", "q2", "q3"]].rename(
    columns={"position": "quali_position", "q1": "q1_time", "q2": "q2_time", "q3": "q3_time"}
)
df = df.merge(qual, on=["raceId", "driverId"], how="left")


# 8. Tidy up columns and types

df = df.rename(
    columns={
        "grid": "grid_position",
        "position": "finish_position",
        "points": "race_points",
        "positionOrder": "finish_position_order",
    }
)

keep_cols = [
    "raceId", "year", "round", "race_name", "date", "circuitId",
    "driverId", "driverRef", "code", "forename", "surname", "driver_nationality",
    "constructorId", "constructorRef", "constructor_name", "constructor_nationality",
    "grid_position", "quali_position", "q1_time", "q2_time", "q3_time",
    "finish_position", "finish_position_order", "race_points", "laps",
    "status", "dnf",
    "driver_points_after", "driver_standing_pos_after", "driver_wins_after",
    "constructor_points_after", "constructor_standing_pos_after", "constructor_wins_after",
]
df = df[keep_cols].sort_values(["year", "round", "finish_position_order"]).reset_index(drop=True)

# 9. Save

out_path = f"{DATA_DIR}/f1_master_{START_YEAR}_{END_YEAR}.csv"
df.to_csv(out_path, index=False)

print(f"Rows: {len(df):,}  |  Races: {df['raceId'].nunique()}  |  Years: {df['year'].min()}-{df['year'].max()}")
print(f"Saved to: {out_path}")
print(df.head(10).to_string(index=False))