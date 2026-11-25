from __future__ import annotations

import json

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    find_dataset,
    iter_transaction_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "results" / "plots"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

CHUNKSIZE = 250_000
NEGATIVE_SAMPLE_PER_CHUNK = 500
RANDOM_SEED = 42


def run_exploration(path: Path | None = DEFAULT_DATA_PATH) -> dict[str, object]:
    """Summarize the full dataset and create plots from bounded samples."""
    csv_path = find_dataset(path)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    target_counts: Counter[int] = Counter()
    missing_counts: Counter[str] = Counter()
    payment_format_counts: Counter[str] = Counter()
    receiving_currency_counts: Counter[str] = Counter()
    payment_currency_counts: Counter[str] = Counter()
    volume_by_month: Counter[str] = Counter()
    laundering_volume_by_month: Counter[str] = Counter()
    plot_samples: list[pd.DataFrame] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for chunk in iter_transaction_chunks(path=csv_path, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        target_counts.update(
            {int(label): int(count) for label, count
             in chunk[TARGET_COLUMN].value_counts().items()}
        )
        missing_counts.update(chunk.isna().sum().astype(int).to_dict())
        payment_format_counts.update(
            chunk["Payment Format"].dropna().astype(str).value_counts().to_dict()
        )
        receiving_currency_counts.update(
            chunk["Receiving Currency"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )
        payment_currency_counts.update(
            chunk["Payment Currency"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        timestamps = pd.to_datetime(chunk["Timestamp"], errors="coerce")
        valid_timestamps = timestamps.notna()
        month_labels = timestamps.loc[valid_timestamps].dt.to_period("M").astype(str)
        volume_by_month.update(month_labels.value_counts().to_dict())
        laundering_months = month_labels.loc[
            chunk.loc[valid_timestamps, TARGET_COLUMN].to_numpy() == 1
        ]
        laundering_volume_by_month.update(
            laundering_months.value_counts().to_dict()
        )

        # Keep every positive row for class-specific plots, but sample negatives
        # to keep memory use bounded despite the extreme class imbalance.
        positive = chunk.loc[chunk[TARGET_COLUMN] == 1]
        negative = chunk.loc[chunk[TARGET_COLUMN] == 0]
        if not positive.empty:
            plot_samples.append(positive.copy())
        if not negative.empty:
            n = min(NEGATIVE_SAMPLE_PER_CHUNK, len(negative))
            sampled_negative = negative.sample(
                n=n,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            plot_samples.append(sampled_negative)

    if not plot_samples:
        raise ValueError("No rows were read from the dataset.")

    sample = pd.concat(plot_samples, ignore_index=True)
    positives = target_counts.get(1, 0)
    negatives = target_counts.get(0, 0)
    prevalence = positives / total_rows if total_rows else 0.0

    sns.set_theme(style="whitegrid")

    # Exact target counts from the full dataset.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["Legitimate (0)", "Labeled suspicious (1)"],
        [negatives, positives],
        color=["steelblue", "darkorange"],
    )
    ax.set_yscale("log")
    ax.set_title("Target distribution (full dataset; logarithmic scale)")
    ax.set_ylabel("Transaction count")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    # Missing values were counted across all chunks.
    nonzero_missing = {
        name: count for name, count in missing_counts.items() if count > 0
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    if nonzero_missing:
        names = list(nonzero_missing)
        ax.bar(names, [nonzero_missing[name] for name in names])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Missing values")
        ax.set_title("Missing values (full dataset)")
    else:
        ax.text(
            0.5, 0.5, "No missing values found in the full dataset",
            ha="center", va="center", fontsize=13,
        )
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    # Amount plots use the documented bounded sample, not the full dataset.
    amount_column = "Amount Received"
    amounts = pd.to_numeric(sample[amount_column], errors="coerce")
    valid_amounts = amounts.notna() & amounts.ge(0)
    sample_for_plot = sample.loc[valid_amounts].copy()
    sample_for_plot["log_amount_received"] = np.log1p(
        amounts.loc[valid_amounts].to_numpy()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=sample_for_plot,
        x="log_amount_received",
        hue=TARGET_COLUMN,
        bins=60,
        stat="count",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set_title(
        "Log transaction amount by label "
        "(all positives + sampled negatives)"
    )
    ax.set_xlabel("log(1 + Amount Received)")
    ax.set_ylabel("Sampled transaction count")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "amount_by_target.png", dpi=150)
    plt.close(fig)

    # Only transaction amounts and the label are included here. Bank IDs are
    # identifiers, so their numeric representation is not treated as a
    # meaningful continuous measurement.
    numeric_for_corr = sample[
        ["Amount Received", "Amount Paid", TARGET_COLUMN]
    ].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_for_corr.corr(numeric_only=True)
    correlation_summary = {
        row: {
            column: (
                None
                if pd.isna(value)
                else float(value)
            )
            for column, value in values.items()
        }
        for row, values in correlation.to_dict().items()
    }

    amount_bins = np.linspace(
        float(sample_for_plot["log_amount_received"].min()),
        float(sample_for_plot["log_amount_received"].max()),
        41,
    )
    amount_distribution: dict[str, dict[str, list[float]]] = {}
    for label, name in ((0, "normal"), (1, "laundering")):
        values = sample_for_plot.loc[
            sample_for_plot[TARGET_COLUMN] == label,
            "log_amount_received",
        ].to_numpy(dtype=float)
        counts, _ = np.histogram(values, bins=amount_bins)
        shares = counts / counts.sum() if counts.sum() else counts
        amount_distribution[name] = {
            "bin_left": amount_bins[:-1].tolist(),
            "bin_right": amount_bins[1:].tolist(),
            "share": shares.tolist(),
        }

    summary: dict[str, object] = {
        "dataset_path": str(csv_path),
        "rows": total_rows,
        "columns": sample.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "target_counts": {"0": negatives, "1": positives},
        "positive_prevalence": prevalence,
        "negative_to_positive_ratio": (
            negatives / positives if positives else None
        ),
        "missing_values_by_column": {
            name: count for name, count in missing_counts.items() if count
        },
        "payment_format_counts": dict(payment_format_counts),
        "receiving_currency_counts": dict(receiving_currency_counts),
        "payment_currency_counts": dict(payment_currency_counts),
        "volume_by_month": dict(sorted(volume_by_month.items())),
        "laundering_volume_by_month": dict(
            sorted(laundering_volume_by_month.items())
        ),
        "amount_received_log_distribution": amount_distribution,
        "numeric_correlations": correlation_summary,
        "plot_sample_rows": len(sample),
        "plot_sample_positive_rows": int((sample[TARGET_COLUMN] == 1).sum()),
        "plot_sample_negative_rows": int((sample[TARGET_COLUMN] == 0).sum()),
        "notes": [
            "Target counts, missing-value counts, and payment-format and "
            "currency counts are accumulated over all CSV chunks.",
            "Monthly transaction counts are accumulated over all rows with "
            "a parseable timestamp.",
            "Amount distributions and correlations use a bounded plotting "
            "sample: all positive rows plus a random sample of negative rows "
            "from each chunk.",
            "Do not interpret the sampled class proportions as the dataset "
            "class proportions.",
        ],
    }

    report_path = REPORT_DIR / "eda_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Numeric feature correlation (bounded plotting sample)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "numeric_correlation_sample.png", dpi=150)
    plt.close(fig)

    # Payment-format counts are exact full-dataset counts.
    formats = payment_format_counts.most_common()
    fig, ax = plt.subplots(figsize=(10, 5))
    if formats:
        labels, counts = zip(*formats)
        ax.bar(labels, counts, color="slateblue")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Transaction count")
        ax.set_title("Payment format counts (full dataset)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "payment_format_counts.png", dpi=150)
    plt.close(fig)

    print(f"Rows processed: {total_rows:,}")
    print(f"Target counts: legitimate={negatives:,}, suspicious={positives:,}")
    print(f"Positive prevalence: {prevalence:.6%}")
    print(f"Negative-to-positive ratio: {negatives / positives:,.1f}:1")
    print(f"Plot sample rows: {len(sample):,}")
    print(f"Summary: {report_path}")
    print(f"Plots: {PLOT_DIR}")

    return summary


if __name__ == "__main__":
    run_exploration()
