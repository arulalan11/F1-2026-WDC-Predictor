import pandas as pd
import matplotlib.pyplot as plt

# Chart 1: Championship win probability (bar chart)

preds = pd.read_csv("wdc_2026_predictions.csv")
top10 = preds.head(10).copy()
top10["label"] = top10["code"]

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#E10600" if i == 0 else "#4C72B0" for i in range(len(top10))]
bars = ax.bar(top10["label"], top10["champion_probability_pct"], color=colors)

for bar, pct in zip(bars, top10["champion_probability_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{pct}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Modeled chance of winning 2026 WDC (%)")
ax.set_title("2026 F1 World Drivers' Championship — Win Probability", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(top10["champion_probability_pct"]) * 1.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("championship_probability.png", dpi=150)
plt.close()
print("Saved championship_probability.png")

# Chart 2: Points progression through the season (line chart)

df = pd.read_csv("F1 Dataset/f1_features_2014_2026.csv")
season = df[df["year"] == 2026].copy()

top_codes = top10["code"].head(5).tolist()
season_top = season[season["code"].isin(top_codes)]

fig, ax = plt.subplots(figsize=(9, 5.5))
color_map = {
    "ANT": "#00D2BE", "RUS": "#00A19C", "HAM": "#DC0000",
    "NOR": "#FF8700", "LEC": "#DC0000",
}
for code in top_codes:
    driver_data = season_top[season_top["code"] == code].sort_values("round")
    ax.plot(driver_data["round"], driver_data["driver_points_after"],
            marker="o", linewidth=2, label=code,
            color=color_map.get(code))

ax.set_xlabel("Round")
ax.set_ylabel("Cumulative championship points")
ax.set_title("2026 WDC Points Progression — Top 5 Contenders", fontsize=13, fontweight="bold")
ax.legend(title="Driver", loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("points_progression.png", dpi=150)
plt.close()
print("Saved points_progression.png")

# Chart 3: Top five championship predictions as a table

table_data = top10.head(5).copy()
table_data["driver"] = table_data["forename"] + " " + table_data["surname"]
table_data["points"] = table_data["driver_points_after"].round(0).astype(int).astype(str)
table_data["title_chance"] = table_data["champion_probability_pct"].map(lambda pct: f"{pct:.1f}%")

fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.axis("off")
table = ax.table(
    cellText=table_data[["driver", "constructor_name", "points", "title_chance"]].values,
    colLabels=["Driver", "Team", "Points", "Title chance"],
    colWidths=[0.25, 0.25, 0.17, 0.23],
    cellLoc="left",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.05)

team_colors = {
    "Mercedes": "#00A19C",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
}

for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.6)
    cell.PAD = 0.04
    cell.get_text().set_color("#000000")
    cell.get_text().set_fontfamily("DejaVu Serif")
    if row == 0:
        cell.set_facecolor("#F0F0F0")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
        if column in (2, 3):
            cell.get_text().set_ha("right")
        if column == 3:
            cell.get_text().set_fontweight("bold")
        if column == 1:
            team = table_data.iloc[row - 1]["constructor_name"]
            cell.get_text().set_color(team_colors.get(team, "#000000"))

fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout(pad=0.4)
plt.savefig("championship_predictions_table.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("Saved championship_predictions_table.png")