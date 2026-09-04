import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, top_k_accuracy_score

IN_PATH = "F1 Dataset/f1_features_2014_2026.csv"

df = pd.read_csv(IN_PATH)

FEATURES = [
    "driver_points_after", "driver_standing_pos_after",
    "constructor_points_after", "constructor_standing_pos_after",
    "form_avg_finish_last5", "form_avg_points_last5", "form_dnf_rate_last5",
    "constructor_form_points_last5",
    "teammate_quali_gap", "teammate_gap_avg_last5",
    "points_gap_to_leader", "races_remaining",
]

# 1. Figure out each season's champion (driver in P1 of the standings at that season's FINAL completed round).

last_round_per_year = df.groupby("year")["round"].max().rename("final_round")
df = df.merge(last_round_per_year, on="year", how="left")

final_standings = df[df["round"] == df["final_round"]]
champions = (
    final_standings[final_standings["driver_standing_pos_after"] == 1]
    .set_index("year")["driverId"]
    .to_dict()
)
print("Champion (or current leader for 2026) found per season:")
for yr, did in sorted(champions.items()):
    name = df.loc[df["driverId"] == did, "code"].iloc[0]
    print(f"  {yr}: {name}")

# 2. Label every row: 1 if that driver was that season's eventual

df["is_champion"] = df.apply(lambda r: int(champions.get(r["year"]) == r["driverId"]), axis=1)

train_df = df[df["year"] < 2026].dropna(subset=FEATURES).copy()
predict_df = df[df["year"] == 2026].copy()
predict_round = predict_df["round"].max()
predict_df = predict_df[predict_df["round"] == predict_round].dropna(subset=FEATURES).copy()

print(f"\nTraining rows (2014-2025): {len(train_df):,}")
print(f"Predicting on 2026, round {predict_round}: {len(predict_df)} drivers")

X_train = train_df[FEATURES]
y_train = train_df["is_champion"]

# 3. Cross-validate BY SEASON (not randomly) -- so we're always testing

groups = train_df["year"]
gkf = GroupKFold(n_splits=5)
fold_scores = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
    model = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    proba = model.predict_proba(X_train.iloc[te_idx])[:, 1]
    ll = log_loss(y_train.iloc[te_idx], proba, labels=[0, 1])
    fold_scores.append(ll)
    print(f"  Fold {fold}: log-loss = {ll:.3f}  (seasons held out: {sorted(groups.iloc[te_idx].unique())})")

print(f"Average cross-validated log-loss: {sum(fold_scores)/len(fold_scores):.3f}  (lower is better; 0 = perfect)")

# 4. Train the FINAL model on all historical data, then predict 2026

final_model = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
final_model.fit(X_train, y_train)

predict_df["champion_probability_raw"] = final_model.predict_proba(predict_df[FEATURES])[:, 1]
# Normalize so probabilities across the 2026 grid sum to 100% (one champion)
total = predict_df["champion_probability_raw"].sum()
predict_df["champion_probability"] = predict_df["champion_probability_raw"] / total

result = predict_df[["code", "forename", "surname", "constructor_name", "driver_points_after", "champion_probability"]]
result = result.sort_values("champion_probability", ascending=False).reset_index(drop=True)
result["champion_probability_pct"] = (result["champion_probability"] * 100).round(1)

print(f"\n=== 2026 WDC win probability, as of round {predict_round} ===")
print(result[["code", "forename", "surname", "constructor_name", "driver_points_after", "champion_probability_pct"]].head(10).to_string(index=False))

result.to_csv("wdc_2026_predictions.csv", index=False)
print("\nSaved full results to wdc_2026_predictions.csv")

# Feature importance -- what the model actually relied on
importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nWhat the model weighted most heavily:")
print(importances.to_string())