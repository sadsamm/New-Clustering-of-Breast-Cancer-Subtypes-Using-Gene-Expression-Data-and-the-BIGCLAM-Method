"""
Generate paper-ready figures comparing BIGCLAM communities with clinical annotations.

Figures:
1. Confusion heatmap (BIGCLAM community vs PAM50) for TCGA and GSE96058
2. Side-by-side community distributions (dominant cluster vs PAM50)
3. Cluster-to-PAM50 heatmaps per cohort
4. Survival summary bars (TCGA & GSE)
5. Pathway enrichment summary bubble plot (top pathways per cohort)
6. Classifier accuracy bars (MLP vs SVM)
7. Graph layout/pca of expression colored by community + PAM50

Usage:
    python scripts/paper_figures.py --output figures/paper

Requires the *_target_added.csv files, cluster assignments under data/clusterings/,
and result summaries for survival/enrichment/classification if available.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle


def load_config(path="config/config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_target_df(dataset_name, config):
    target_path = Path(config["dataset_preparation"][dataset_name]["output"])
    df = pd.read_csv(target_path, index_col=0)
    pam50_row = df.index[-1]
    pam50_series = df.iloc[-1]
    pam50_series = pam50_series.astype(str).str.replace("pam50 subtype:", "").str.strip()
    samples = df.columns.tolist()
    return pd.DataFrame(
        {"sample_id": samples, "PAM50": pam50_series.values}, index=samples
    )


def load_cluster_assignments(dataset_name):
    cluster_file = Path(f"data/clusterings/{dataset_name}_communities.npy")
    if not cluster_file.exists():
        raise FileNotFoundError(f"{cluster_file} is missing. Run clustering first.")
    communities = np.load(cluster_file)
    if communities.ndim > 1:
        communities = np.argmax(communities, axis=1)
    communities = communities.flatten()
    return communities


def create_summary_df(dataset_name, config):
    df = load_target_df(dataset_name, config)
    communities = load_cluster_assignments(dataset_name)
    if len(communities) != len(df):
        raise ValueError("Cluster assignment length mismatch.")
    df["community"] = communities
    df["dataset"] = dataset_name
    return df.reset_index(drop=True)


def plot_confusion_pam50(df, dataset_name, ax):
    matrix = pd.crosstab(df["community"], df["PAM50"], normalize="index")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="rocket_r", ax=ax)
    ax.set_title(f"{dataset_name.upper()}: Community vs PAM50")
    ax.set_xlabel("PAM50 subtype")
    ax.set_ylabel("BIGCLAM community")


def plot_distribution(df, dataset_name, ax):
    sns.countplot(x="community", hue="PAM50", data=df, ax=ax)
    ax.set_title(f"{dataset_name.upper()}: PAM50 overlap per community")
    ax.set_xlabel("BIGCLAM community")
    ax.set_ylabel("Sample count")
    ax.legend(title="PAM50", bbox_to_anchor=(1.05, 1), loc="upper left")


def make_cluster_pam50_heatmap(output_dir, summary):
    heatmap_dir = Path("results/cluster_pam50_mapping")
    figs = []
    for dataset in ["tcga", "gse96058"]:
        csv_path = heatmap_dir / dataset / "cluster_pam50_mapping.csv"
        if not csv_path.exists():
            print(f"[SKIP] No cluster-to-PAM50 CSV for {dataset} ({csv_path})")
            continue
        matrix = pd.read_csv(csv_path)
        pivot = matrix.pivot_table(
            index="bigclam_cluster", columns="pam50_subtype", values="count", fill_value=0
        )
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            pivot,
            cmap="viridis",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Sample count"},
        )
        plt.title(f"{dataset.upper()}: Cluster vs PAM50 (counts)")
        plt.ylabel("BIGCLAM community")
        plt.xlabel("PAM50 subtype")
        path = Path(output_dir) / f"heatmap_cluster_pam50_{dataset}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[OK] Saved cluster-to-PAM50 heatmap for {dataset} to {path}")


def parse_survival_summary():
    summary_path = Path("results/survival/SURVIVAL_ANALYSIS_SUMMARY.md")
    if not summary_path.exists():
        print("[SKIP] Survival summary not available.")
        return None
    with open(summary_path) as fh:
        text = fh.read()
    datasets = {}
    for ds in ["GSE96058", "TCGA-BRCA"]:
        segment = text.split(f"## {ds} Dataset Results")[-1].split("---")[0]
        percentages = []
        for line in segment.splitlines():
            if "At" in line and "%" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    pct = parts[1].split("%")[0].strip("~ ")
                    try:
                        percentages.append(float(pct))
                    except ValueError:
                        continue
        datasets[ds] = percentages[:3] if percentages else []
    return datasets


def make_survival_summary(output_dir, summary):
    data = parse_survival_summary()
    if not data:
        return
    dfs = []
    for ds, values in data.items():
        for idx, val in enumerate(values):
            dfs.append({"dataset": ds.replace("-BRCA", ""), "index": idx, "survival%": val})
    if not dfs:
        return
    df = pd.DataFrame(dfs)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="dataset", y="survival%", hue="index", data=df)
    plt.title("Key survival estimates (visual observation)")
    plt.ylabel("Approx. survival (%)")
    plt.xlabel("Dataset")
    legend = plt.legend(title="observation")
    path = Path(output_dir) / "survival_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Saved survival summary figure to {path}")


def make_pathway_enrichment(output_dir):
    bio_dir = Path("results/biological_interpretation")
    rows = []
    for dataset in ["tcga", "gse96058"]:
        dataset_dir = bio_dir / dataset
        if not dataset_dir.exists():
            continue
        csvs = list(dataset_dir.glob("cluster_*_pathways_*.csv"))
        for csv in csvs:
            try:
                df = pd.read_csv(csv).copy()
            except pd.errors.EmptyDataError:
                continue
            df = df.dropna(subset=["Adjusted P-value"], errors="ignore")
            if "Adjusted P-value" not in df.columns:
                continue
            df = df.sort_values("Adjusted P-value").head(3)
            cluster = csv.name.split("_")[1]
            for _, row in df.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "cluster": cluster,
                        "pathway": row.iloc[0],
                        "score": -np.log10(row["Adjusted P-value"]),
                    }
                )
    if not rows:
        print("[SKIP] No pathway enrichment files found.")
        return
    pf = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=pf,
        x="score",
        y="pathway",
        hue="dataset",
        style="cluster",
        s=120,
        palette="Set2",
    )
    plt.xlabel("-log10(FDR)")
    plt.ylabel("Pathway")
    plt.title("Top enriched pathways per community")
    plt.tight_layout()
    path = Path(output_dir) / "pathway_enrichment_summary.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Saved pathway enrichment summary to {path}")


def make_classifier_bars(output_dir):
    path = Path("results/classification/gse96058_data_classification_results.pkl")
    if not path.exists():
        print("[SKIP] Classification summary missing.")
        return
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except ModuleNotFoundError as exc:
        print("[SKIP] Cannot load classification pickle (missing dependency).", exc)
        return
    metrics = []
    for model in ["mlp", "svm"]:
        entry = data.get(model, {})
        acc = entry.get("accuracy")
        auc = entry.get("roc_auc")
        if acc is not None:
            metrics.append({"model": model.upper(), "metric": "Accuracy", "value": acc})
        if auc is not None:
            metrics.append({"model": model.upper(), "metric": "ROC AUC", "value": auc})
    if not metrics:
        print("[SKIP] No accuracy/AUC values found in classification results.")
        return
    df = pd.DataFrame(metrics)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="model", y="value", hue="metric", data=df)
    plt.ylim(0, 1)
    plt.title("Classifier performance (GSE96058)")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.tight_layout()
    path = Path(output_dir) / "classifier_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Saved classifier performance figure to {path}")


def load_expression_pca(dataset_name, config, n_genes=500):
    target_path = Path(config["dataset_preparation"][dataset_name]["output"])
    df = pd.read_csv(target_path, index_col=0)
    expr = df.iloc[:-1]
    if expr.shape[0] > n_genes:
        expr = expr.iloc[:n_genes]
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0).T
    pca = PCA(n_components=2)
    coords = pca.fit_transform(expr)
    pam50 = df.iloc[-1].astype(str).str.replace("pam50 subtype:", "").str.strip()
    aligned = pam50.loc[expr.index]
    return pd.DataFrame(
        {
            "sample_id": expr.index,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "PAM50": aligned.values,
        },
        index=expr.index,
    )


def make_graph_layout(output_dir, summary, config):
    dataset = "tcga"
    expr_df = load_expression_pca(dataset, config)
    merged = expr_df.merge(
        summary[summary["dataset"] == dataset][["sample_id", "community"]],
        on="sample_id",
        how="inner",
    )
    merged["community"] = merged["community"].astype(str)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=merged,
        x="PC1",
        y="PC2",
        hue="community",
        style="PAM50",
        palette="tab10",
        alpha=0.7,
        s=30,
    )
    plt.title("Expression PCA (TCGA) colored by community + PAM50")
    plt.tight_layout()
    path = Path(output_dir) / "graph_layout_tcga.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Saved PCA-based graph layout to {path}")

def make_figures(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["tcga", "gse96058"]
    summary = pd.concat(
        [create_summary_df(ds, config) for ds in datasets], ignore_index=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, dataset in enumerate(datasets):
        data = summary[summary["dataset"] == dataset]
        plot_confusion_pam50(data, dataset, axes[idx, 0])
        plot_distribution(data, dataset, axes[idx, 1])
    plt.tight_layout()
    fig_path = output_dir / "figure_cluster_vs_pam50.png"
    fig.savefig(fig_path, dpi=300)
    print(f"[OK] Saved cluster vs PAM50 figure to {fig_path}")

    make_cluster_pam50_heatmap(output_dir, summary)
    make_survival_summary(output_dir, summary)
    make_pathway_enrichment(output_dir)
    make_classifier_bars(output_dir)
    make_graph_layout(output_dir, summary, config)


def main():
    parser = argparse.ArgumentParser(description="Paper figure generation helper")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/paper",
        help="Directory for generated figures",
    )
    args = parser.parse_args()

    config = load_config()
    make_figures(args.output, config)


if __name__ == "__main__":
    main()

