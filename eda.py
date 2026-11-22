
"""Complete EDA for the FastAPI House Price Predictor.

Run:
    python eda.py

Input:
    data/Housing.csv

Output:
    eda_output/ (plots, CSV reports, and cleaning summary)
"""   
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a display

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Housing.csv"
OUTPUT_DIR = ROOT / "eda_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

TARGET = "price"
REQUIRED = NUMERIC + CATEGORICAL + [TARGET]

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------
# 2. LOAD DATA AND VALIDATE COLUMNS
# ---------------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Make sure Housing.csv is inside the data folder."
        )

    raw = pd.read_csv(DATA_PATH)

    missing_columns = sorted(set(REQUIRED) - set(raw.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work only with the features used by the model.
    df = raw[REQUIRED].copy()

    print("\n" + "=" * 65)
    print("INITIAL DATASET INFORMATION")
    print("=" * 65)
    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")
    print(f"Duplicate rows initially: {df.duplicated().sum()}")

    print("\nData types before conversion:")
    print(df.dtypes)

    print("\nFirst five rows:")
    print(df.head())

    return df


# ---------------------------------------------------------
# 3. CLEAN DATA AND RECORD EACH CLEANING STEP
# ---------------------------------------------------------

def clean_data(raw):
    df = raw.copy()
    cleaning_steps = []

    def record(step, before, after, reason):
        cleaning_steps.append({
            "step": step,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "reason": reason,
        })

    initial_rows = len(df)

    # Convert numerical fields; invalid strings become NaN.
    for column in NUMERIC + [TARGET]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Standardize categorical fields.
    for column in CATEGORICAL:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
        )

    # Replace infinite numeric values with missing values.
    df[NUMERIC + [TARGET]] = df[
        NUMERIC + [TARGET]
    ].replace([np.inf, -np.inf], np.nan)

    # Step A: Remove rows without a usable target price.
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    record(
        "Remove missing/invalid price",
        before,
        len(df),
        "A known price is needed to construct the supervised target.",
    )

    # Step B: Remove impossible measurements.
    before = len(df)
    valid = (
        (df["price"] > 0)
        & (df["area"] > 0)
        & (df["bedrooms"] >= 0)
        & (df["bathrooms"] >= 0)
        & (df["stories"] >= 0)
        & (df["parking"] >= 0)
    )
    df = df.loc[valid].copy()
    record(
        "Remove invalid measurements",
        before,
        len(df),
        "Require positive price/area and non-negative counts.",
    )

    # Step C: Remove exact duplicate records.
    before = len(df)
    df = df.drop_duplicates().copy()
    record(
        "Remove duplicate rows",
        before,
        len(df),
        "Keep one copy of each identical selected record.",
    )

    cleaning_report = pd.DataFrame(cleaning_steps)
    cleaning_report.to_csv(
        OUTPUT_DIR / "cleaning_report.csv", index=False
    )

    cleaning_report.to_string(
        OUTPUT_DIR / "cleaning_report.txt", index=False
    )

    print("\n" + "=" * 65)
    print("CLEANING REPORT")
    print("=" * 65)
    print(cleaning_report.to_string(index=False))
    print(f"\nStarting rows: {initial_rows}")
    print(f"Final cleaned rows: {len(df)}")
    print(f"Total rows removed: {initial_rows - len(df)}")

    # Keep missing feature values for EDA reporting.
    # The model pipeline imputes these values during training.
    return df


# ---------------------------------------------------------
# 4. DATASET OVERVIEW AND MISSING-VALUE ANALYSIS
# ---------------------------------------------------------

def dataset_overview(raw, clean):
    print("\n" + "=" * 65)
    print("CLEANED DATASET INFORMATION")
    print("=" * 65)

    print(f"Rows: {len(clean)}")
    print(f"Columns: {len(clean.columns)}")

    print("\nNumerical descriptive statistics:")
    print(clean[NUMERIC + [TARGET]].describe().round(2))

    numeric_summary = clean[
        NUMERIC + [TARGET]
    ].describe().T
    numeric_summary.to_csv(
        OUTPUT_DIR / "numeric_summary.csv"
    )

    print("\nCategorical value counts:")
    for column in CATEGORICAL:
        print(f"\n{column}:")
        print(
            clean[column]
            .value_counts(dropna=False)
            .head(20)
        )

    categorical_rows = []

    for column in CATEGORICAL:
        counts = clean[column].value_counts(dropna=False)

        for category, count in counts.items():
            categorical_rows.append({
                "feature": column,
                "category": str(category),
                "count": int(count),
                "percentage": round(
                    100 * count / len(clean), 2
                ) if len(clean) else 0,
            })

    pd.DataFrame(categorical_rows).to_csv(
        OUTPUT_DIR / "categorical_summary.csv",
        index=False,
    )

    # Missingness report before and after cleaning.
    missing_report = pd.DataFrame({
        "missing_before": raw.isna().sum(),
        "missing_after": clean.isna().sum(),
    })

    missing_report["percent_before"] = (
        100 * missing_report["missing_before"] / len(raw)
        if len(raw) else 0
    )

    missing_report["percent_after"] = (
        100 * missing_report["missing_after"] / len(clean)
        if len(clean) else 0
    )

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values_report.csv"
    )

    print("\nMissing-value report:")
    print(missing_report.round(2))

    # Visualize missingness before and after cleaning.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        raw.isna(),
        cbar=False,
        yticklabels=False,
        ax=axes[0],
    )
    axes[0].set_title("Missing Values — Before Cleaning")
    axes[0].set_xlabel("Features")

    sns.heatmap(
        clean.isna(),
        cbar=False,
        yticklabels=False,
        ax=axes[1],
    )
    axes[1].set_title("Missing Values — After Cleaning")
    axes[1].set_xlabel("Features")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "01_missing_values_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------
# 5. HISTOGRAMS AND DISTRIBUTION ANALYSIS
# ---------------------------------------------------------

def plot_histograms(df):
    columns = NUMERIC + [TARGET]

    fig, axes = plt.subplots(
        2, 3, figsize=(17, 10)
    )
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.histplot(
            data=df,
            x=column,
            kde=(column in ["area", "price"]),
            bins=25,
            ax=axes[i],
        )
        axes[i].set_title(
            f"Distribution of {column.title()}"
        )
        axes[i].set_xlabel(column.title())
        axes[i].set_ylabel("Number of houses")

    fig.suptitle(
        "Numerical Feature Distributions",
        fontsize=17,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "02_numeric_histograms.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------
# 6. BOXPLOTS AND IQR OUTLIER ANALYSIS
# ---------------------------------------------------------

def plot_boxplots(df):
    fig, axes = plt.subplots(
        2, 3, figsize=(17, 10)
    )
    axes = axes.flatten()

    outlier_rows = []

    for i, column in enumerate(NUMERIC + [TARGET]):
        values = df[column].dropna()

        sns.boxplot(
            data=df,
            y=column,
            ax=axes[i],
        )

        axes[i].set_title(
            f"Boxplot of {column.title()}"
        )

        if len(values):
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(
                ((values < lower) | (values > upper)).sum()
            )

            outlier_rows.append({
                "feature": column,
                "q1": q1,
                "median": values.median(),
                "q3": q3,
                "iqr": iqr,
                "lower_iqr_bound": lower,
                "upper_iqr_bound": upper,
                "outlier_count": count,
                "outlier_percentage": round(
                    100 * count / len(values), 2
                ),
            })

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "03_numeric_boxplots.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    pd.DataFrame(outlier_rows).to_csv(
        OUTPUT_DIR / "outlier_report.csv",
        index=False,
    )

    print("\nIQR-based outlier report:")
    print(pd.DataFrame(outlier_rows).round(2).to_string(index=False))


# ---------------------------------------------------------
# 7. CORRELATION HEATMAP
# ---------------------------------------------------------

def plot_correlation_heatmap(df):
    # Pearson correlation for numerical features.
    corr = df[NUMERIC + [TARGET]].corr(
        method="pearson"
    )

    corr.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
    )

    plt.title(
        "Correlation Heatmap: Numerical Features and Price",
        fontsize=15,
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_correlation_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nCorrelation with price:")
    print(
        corr[TARGET]
        .sort_values(ascending=False)
        .round(3)
    )


# ---------------------------------------------------------
# 8. CATEGORICAL COUNT PLOTS
# ---------------------------------------------------------

def plot_categorical_counts(df):
    fig, axes = plt.subplots(
        3, 3, figsize=(18, 14)
    )
    axes = axes.flatten()

    for i, column in enumerate(CATEGORICAL):
        order = (
            df[column]
            .fillna("missing")
            .value_counts()
            .index
        )

        sns.countplot(
            data=df.assign(
                **{column: df[column].fillna("missing")}
            ),
            x=column,
            order=order,
            ax=axes[i],
        )

        axes[i].set_title(
            f"Count of {column.title()}"
        )
        axes[i].set_xlabel(column.title())
        axes[i].set_ylabel("Number of houses")
        axes[i].tick_params(
            axis="x", labelrotation=25
        )

    # Hide unused subplot panels.
    for j in range(len(CATEGORICAL), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Categorical Feature Distributions",
        fontsize=17,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "05_categorical_counts.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------
# 9. SCATTERPLOTS: AREA VS PRICE
# ---------------------------------------------------------

def plot_area_vs_price(df):
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6)
    )

    sns.scatterplot(
        data=df,
        x="area",
        y="price",
        alpha=0.65,
        ax=axes[0],
    )
    axes[0].set_title("Area vs House Price")
    axes[0].set_xlabel("Area")
    axes[0].set_ylabel("Price")

    sns.regplot(
        data=df,
        x="area",
        y="price",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=axes[1],
    )
    axes[1].set_title(
        "Area vs Price with Linear Trend"
    )
    axes[1].set_xlabel("Area")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_area_vs_price.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------
# 10. PRICE DISTRIBUTION BY CATEGORICAL FEATURES
# ---------------------------------------------------------

def plot_price_by_category(df):
    fig, axes = plt.subplots(
        3, 3, figsize=(19, 15)
    )
    axes = axes.flatten()

    for i, column in enumerate(CATEGORICAL):
        plot_df = df[[column, TARGET]].copy()
        plot_df[column] = plot_df[column].fillna("missing")

        order = (
            plot_df.groupby(column, observed=True)[TARGET]
            .median()
            .sort_values()
            .index
        )

        sns.boxplot(
            data=plot_df,
            x=column,
            y=TARGET,
            order=order,
            ax=axes[i],
        )

        axes[i].set_title(
            f"Price by {column.title()}"
        )
        axes[i].set_xlabel(column.title())
        axes[i].set_ylabel("Price")
        axes[i].tick_params(
            axis="x", labelrotation=25
        )

    for j in range(len(CATEGORICAL), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "House Price Across Categorical Groups",
        fontsize=17,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "07_price_by_category.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------
# 11. OPTIONAL: PRICE BANDS FOR EDA VISUALIZATION
# ---------------------------------------------------------

def plot_price_bands(df):
    # This is for exploratory visualization only.
    # The actual model defines its own target bands.
    plot_df = df.copy()

    try:
        plot_df["price_band_eda"] = pd.qcut(
            plot_df[TARGET],
            q=4,
            labels=[
                "Budget",
                "Standard",
                "Premium",
                "Luxury",
            ],
        )
    except ValueError:
        print(
            "Could not create four EDA price bands. "
            "Prices may contain too many tied values."
        )
        return

    plt.figure(figsize=(9, 6))

    sns.countplot(
        data=plot_df,
        x="price_band_eda",
        order=["Budget", "Standard", "Premium", "Luxury"],
    )

    plt.title("Distribution of House Price Bands")
    plt.xlabel("Price Band")
    plt.ylabel("Number of houses")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "08_price_band_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plot_df.groupby(
        "price_band_eda", observed=True
    )[TARGET].agg(
        ["count", "min", "median", "max"]
    ).to_csv(
        OUTPUT_DIR / "price_band_summary.csv"
    )


# ---------------------------------------------------------
# 12. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    print("Starting House Price EDA...")

    raw = load_data()
    clean = clean_data(raw)

    if clean.empty:
        raise ValueError(
            "No valid rows remain after cleaning."
        )

    dataset_overview(raw, clean)

    plot_histograms(clean)
    plot_boxplots(clean)
    plot_correlation_heatmap(clean)
    plot_categorical_counts(clean)
    plot_area_vs_price(clean)
    plot_price_by_category(clean)
    plot_price_bands(clean)

    print("\n" + "=" * 65)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"All reports and graphs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
