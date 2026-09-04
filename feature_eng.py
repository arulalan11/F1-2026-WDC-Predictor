import pandas as pd

IN_PATH = "F1 Dataset/f1_master_2014_2026.csv"
OUT_PATH = "F1 Dataset/f1_features_2014_2026.csv"
ROLL_WINDOW = 5  

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["driverId", "date"]).reset_index(drop=True)

# 1. Rolling driver form — average finish position & points over the last N races, computed PER DRIVER, using only prior races.

grp = df.groupby("driverId")

df["form_avg_finish_last5"] = grp["finish_position_order"].transform(
    lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
)
df["form_avg_points_last5"] = grp["race_points"].transform(
    lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
)
df["form_dnf_rate_last5"] = grp["dnf"].transform(
    lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
)
# 2. Constructor (car) strength — rolling average points scored by the  TEAM (both drivers combined), per race, using only prior races.

team_race_points = (
    df.groupby(["constructorId", "raceId", "date"])["race_points"]
    .sum()
    .reset_index()
    .sort_values(["constructorId", "date"])
)
team_race_points["constructor_form_points_last5"] = team_race_points.groupby("constructorId")[
    "race_points"
].transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())

df = df.merge(
    team_race_points[["constructorId", "raceId", "constructor_form_points_last5"]],
    on=["constructorId", "raceId"],
    how="left",
)

# 3. Teammate-relative qualifying gap

quali_small = df[["raceId", "constructorId", "driverId", "quali_position"]]
paired = quali_small.merge(quali_small, on=["raceId", "constructorId"], suffixes=("", "_team"))
paired = paired[paired["driverId"] != paired["driverId_team"]]
paired["gap"] = paired["quali_position"] - paired["quali_position_team"]

teammate_gaps = (
    paired.groupby(["raceId", "constructorId", "driverId"])["gap"]
    .mean()
    .reset_index()
    .rename(columns={"gap": "teammate_quali_gap"})
)
df = df.merge(teammate_gaps, on=["raceId", "constructorId", "driverId"], how="left")

df = df.sort_values(["driverId", "date"])
df["teammate_gap_avg_last5"] = df.groupby("driverId")["teammate_quali_gap"].transform(
    lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
)
# 4. Championship context — points gap to the CURRENT leader, and how many races are left in the season.

season_leader_points = df.groupby("raceId")["driver_points_after"].transform("max")
df["points_gap_to_leader"] = season_leader_points - df["driver_points_after"]

full_races = pd.read_csv("F1 Dataset/races.csv")
season_length = full_races.groupby("year")["round"].max().rename("season_length")
df = df.merge(season_length, on="year", how="left")
df["races_remaining"] = df["season_length"] - df["round"]
df = df.drop(columns=["season_length"])

# 5. Save

df = df.sort_values(["year", "round", "finish_position_order"]).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)

feature_cols = [
    "form_avg_finish_last5", "form_avg_points_last5", "form_dnf_rate_last5",
    "constructor_form_points_last5", "teammate_quali_gap", "teammate_gap_avg_last5",
    "points_gap_to_leader", "races_remaining",
]
print(f"Rows: {len(df):,}")
print(f"Saved to: {OUT_PATH}")
print("\nNew feature columns:", feature_cols)
print("\nSample (a recent race):")
sample = df[df["year"] == df["year"].max()].tail(10)
print(sample[["year", "round", "code", "constructor_name", "finish_position_order"] + feature_cols].to_string(index=False))