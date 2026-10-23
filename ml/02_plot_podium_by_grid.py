"""
Draw a bar chart of the podium rate by starting grid position.
Read-only for MongoDB. Writes one PNG file to docs/images/.

Run from the project root with the virtual environment active:
    python ml/02_plot_podium_by_grid.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FIRST_SEASON, LAST_SEASON = 2014, 2025
OUTPUT = PROJECT_ROOT / "docs" / "images" / "podium_rate_by_grid.png"

BACKGROUND = "#141517"
TEXT = "#e8e8e8"
MUTED = "#9aa0a6"
GRIDLINE = "#2a2d31"
ACCENT = "#d13b30"

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "f1_intelligence")
if not uri or uri.startswith("replace-with"):
    sys.exit("MONGODB_URI is missing or still a placeholder. Check your .env file.")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
    rows = list(
        client[db_name]["race_results"].find(
            {"season": {"$gte": FIRST_SEASON, "$lte": LAST_SEASON}, "started": True},
            {"_id": 0, "season": 1, "round": 1, "grid": 1, "position": 1},
        )
    )
except PyMongoError as error:
    print(f"Database error: {type(error).__name__}")
    print(str(error)[:400])
    sys.exit(1)
finally:
    client.close()

if not rows:
    sys.exit("No race results found. Run the season pipeline first.")

df = pd.DataFrame(rows)
df["podium"] = df["position"].isin([1, 2, 3]).astype(int)

race_count = df[["season", "round"]].drop_duplicates().shape[0]
grid_zero = int((df["grid"] == 0).sum())
grid_above_20 = int((df["grid"] > 20).sum())

shown = df[(df["grid"] >= 1) & (df["grid"] <= 20)]
by_grid = shown.groupby("grid")["podium"].agg(starters="count", rate="mean")

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
bars = ax.bar(by_grid.index, by_grid["rate"] * 100, color=ACCENT, width=0.7)
ax.bar_label(
    bars,
    labels=[f"{value * 100:.0f}" for value in by_grid["rate"]],
    color=MUTED,
    fontsize=7,
    padding=2,
)

ax.set_xticks(range(1, 21))
ax.set_ylim(0, 100)
ax.set_xlabel("Starting grid position", color=MUTED)
ax.set_ylabel("Starters finishing in the top 3 (%)", color=MUTED)
ax.set_title(
    f"Podium rate by starting grid position, {FIRST_SEASON} to {LAST_SEASON}",
    color=TEXT,
    loc="left",
    fontsize=12,
)
ax.tick_params(colors=MUTED)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRIDLINE)
ax.yaxis.grid(True, color=GRIDLINE)
ax.set_axisbelow(True)

fig.text(
    0.01,
    0.01,
    f"{race_count} races, {len(df)} starters. Excluded: {grid_zero} rows with grid 0 "
    f"and {grid_above_20} with grid above 20. Data: Jolpica-F1 (CC BY-NC-SA 4.0).",
    color=MUTED,
    fontsize=7,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=150, facecolor=BACKGROUND)
plt.close(fig)

print(f"Saved chart to {OUTPUT}")
print(by_grid.round(3).to_string())