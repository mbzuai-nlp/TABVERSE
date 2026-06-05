import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "/home/momina.ahsan/TABVERSE/results/scores/task/task_category.csv"
OUTPUT_DIR = "/home/momina.ahsan/TABVERSE/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "task_category_radar_best_models.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "task_category_radar_best_models.pdf")


# ============================================================
# Task QA categories
# ============================================================

METRICS = ["SL", "CL", "MIL", "ACA", "CE", "SBV", "MBV"]

METRIC_LABELS = {
    "SL": "Simple\nLookup",
    "CL": "Conditional\nLookup",
    "MIL": "Multi-item\nLookup",
    "ACA": "Aggregation /\nCounting /\nArithmetic",
    "CE": "Comparison &\nExtremum",
    "SBV": "Single-step\nBinary\nVerification",
    "MBV": "Multi-hop\nBinary\nVerification",
}


# ============================================================
# Best model from each pipeline
# ============================================================

SELECTED_MODELS = [
    ("VLM", "gemini-3-flash-preview", "Gemini-3 Flash (VLM)"),
    ("VLM-TEXT", "gemini-3-flash-preview", "Gemini-3 Flash (VLM-Text)"),
    ("LLM", "Qwen3-30B-A3B-Instruct-2507", "Qwen3-30B-A3B (LLM)"),
]


# ============================================================
# Helper
# ============================================================

def close_loop(values):
    return values + values[:1]


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df.dropna(subset=["pipeline", "model"]).copy()

df["pipeline"] = df["pipeline"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()

for metric in METRICS:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")


# ============================================================
# Select rows to plot
# ============================================================

rows = []

for pipeline, model, label in SELECTED_MODELS:
    selected = df[(df["pipeline"] == pipeline) & (df["model"] == model)].copy()

    if selected.empty:
        available_models = df[df["pipeline"] == pipeline]["model"].tolist()
        raise ValueError(
            f"\nCould not find selected model.\n"
            f"Pipeline: {pipeline}\n"
            f"Model: {model}\n\n"
            f"Available models for {pipeline}:\n{available_models}\n"
        )

    selected["label"] = label
    rows.append(selected)

plot_df = pd.concat(rows, ignore_index=True)


# ============================================================
# Radar setup
# ============================================================

n_metrics = len(METRICS)

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles = close_loop(angles)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 7.2),
    subplot_kw={"polar": True},
)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)

ax.tick_params(axis="x", pad=12)

ax.set_ylim(0, 80)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8)

ax.grid(True, linewidth=0.6, alpha=0.65)
ax.spines["polar"].set_linewidth(0.9)

linestyles = ["-", "--", "-."]
markers = ["o", "s", "^"]

for i, (_, row) in enumerate(plot_df.iterrows()):
    values = row[METRICS].astype(float).tolist()
    values = close_loop(values)

    ax.plot(
        angles,
        values,
        linewidth=2.4,
        linestyle=linestyles[i],
        marker=markers[i],
        markersize=4,
        label=row["label"],
    )

    ax.fill(
        angles,
        values,
        alpha=0.06,
    )

# No title inside the figure. Put explanation in LaTeX caption instead.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.34),
    ncol=1,
    frameon=False,
    fontsize=9,
)

plt.tight_layout(rect=[0, 0, 1, 0.82])

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close()


# ============================================================
# Print summary
# ============================================================

print(f"Saved PNG: {OUTPUT_PNG}")
print(f"Saved PDF: {OUTPUT_PDF}")

print("\nModels plotted:")
for _, row in plot_df.iterrows():
    values = row[METRICS].astype(float)
    mean_score = values.mean()
    print(f"{row['label']}: mean = {mean_score:.2f}")