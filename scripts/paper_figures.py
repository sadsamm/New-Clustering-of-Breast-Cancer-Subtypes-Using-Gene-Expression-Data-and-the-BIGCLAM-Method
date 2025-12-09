"""
Generate comprehensive paper-ready figures comparing BIGCLAM communities with clinical annotations.

Figures Generated (for both TCGA-BRCA and GSE96058 datasets where applicable):

1. Confusion heatmap (BIGCLAM community vs PAM50) for both datasets
2. Side-by-side community distributions (dominant cluster vs PAM50)
3. t-SNE plots colored by BIGCLAM clusters and PAM50 subtypes
4. UMAP plots colored by BIGCLAM clusters and PAM50 subtypes (if UMAP available)
5. Sankey diagrams for cluster-PAM50 mapping (if Plotly available)
6. Differential expression and pathway enrichment heatmaps
7. Kaplan-Meier survival curves by cluster (TCGA-BRCA only, requires lifelines)
8. ARI/NMI comparison barplots vs other clustering methods
9. Cross-dataset correlation heatmap and overlap analysis
10. Cluster-to-PAM50 heatmaps per cohort
11. Survival summary bars (TCGA & GSE)
12. Pathway enrichment summary bubble plots
13. Classifier accuracy bars (MLP vs SVM)
14. Graph layout/PCA colored by community + PAM50

Usage:
    python scripts/paper_figures.py --output figures/paper

Requirements:
- *_target_added.csv files in data/
- Cluster assignments in data/clusterings/
- Processed expression data in data/processed/
- Biological interpretation results in results/biological_interpretation/
- Survival analysis results in results/survival/
- Method comparison results in results/method_comparison/

Optional dependencies:
- umap-learn (for UMAP plots)
- plotly (for Sankey diagrams)
- lifelines (for Kaplan-Meier curves)
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[Warning] Plotly not available - Sankey diagrams will be skipped")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[Warning] UMAP not available - UMAP plots will be skipped")

try:
    import lifelines
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("[Warning] Lifelines not available - Survival analysis will be skipped")


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
    # Map short dataset names to actual file names
    name_mapping = {
        'tcga': 'tcga_brca_data',
        'gse96058': 'gse96058_data'
    }
    actual_name = name_mapping.get(dataset_name, dataset_name)
    cluster_file = Path(f"data/clusterings/{actual_name}_communities.npy")

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

        # Read the existing format and parse PAM50 distributions
        df = pd.read_csv(csv_path)
        heatmap_data = []

        for _, row in df.iterrows():
            cluster = row['cluster']
            distribution = row['pam50_distribution']

            # Parse the distribution string
            # Format: "Luminal A: 69 (43.9%), Luminal B: 36 (22.9%), ..." or
            # "pam50 subtype: LumA: 688 (50.1%), pam50 subtype: LumB: 311 (22.6%), ..."
            parts = distribution.split(', ')
            for part in parts:
                if ': ' in part:
                    # Handle both formats: with/without "pam50 subtype: " prefix
                    if 'pam50 subtype: ' in part:
                        # GSE96058 format: "pam50 subtype: LumA: 688 (50.1%)"
                        _, subtype_part = part.split('pam50 subtype: ', 1)
                        pam50_type, count_part = subtype_part.split(': ', 1)
                    else:
                        # TCGA format: "Luminal A: 69 (43.9%)"
                        pam50_type, count_part = part.split(': ', 1)

                    count = int(count_part.split(' ')[0])  # Extract number before percentage
                    heatmap_data.append({
                        'bigclam_cluster': cluster,
                        'pam50_subtype': pam50_type,
                        'count': count
                    })

        if heatmap_data:
            matrix = pd.DataFrame(heatmap_data)
            pivot = matrix.pivot_table(
                index="bigclam_cluster", columns="pam50_subtype", values="count", fill_value=0
            )

            # Ensure pivot table contains integers
            pivot = pivot.astype(int)

            plt.figure(figsize=(10, 6))
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
        else:
            print(f"[SKIP] Could not parse PAM50 distribution data for {dataset}")


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
            df = df.dropna(subset=["Adjusted P-value"])
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


def make_tsne_umap_plots(output_dir, config):
    """Generate t-SNE and UMAP plots colored by BIGCLAM clusters for both datasets."""
    output_dir = Path(output_dir)

    for dataset in ["tcga", "gse96058"]:
        # Load expression data and communities
        expr_df = load_expression_pca(dataset, config, n_genes=2000)  # Use more genes for better embedding
        summary_df = create_summary_df(dataset, config)
        merged = expr_df.merge(
            summary_df[["sample_id", "community"]],
            on="sample_id",
            how="inner",
        )

        # Get raw expression data for better embeddings
        target_path = Path(config["dataset_preparation"][dataset]["output"])
        df = pd.read_csv(target_path, index_col=0)
        expr_raw = df.iloc[:-1].apply(pd.to_numeric, errors="coerce").fillna(0).T

        # Subset to common samples
        common_samples = merged["sample_id"].values
        expr_subset = expr_raw.loc[expr_raw.index.isin(common_samples)]

        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            tsne_coords = tsne.fit_transform(expr_subset.values)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # t-SNE colored by communities
            scatter1 = ax1.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                 c=merged["community"], cmap="tab10", alpha=0.7, s=20)
            ax1.set_title(f"{dataset.upper()}: t-SNE colored by BIGCLAM clusters")
            ax1.set_xlabel("t-SNE 1")
            ax1.set_ylabel("t-SNE 2")
            plt.colorbar(scatter1, ax=ax1, label="Cluster")

            # t-SNE colored by PAM50
            pam50_colors = pd.Categorical(merged["PAM50"]).codes
            scatter2 = ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                 c=pam50_colors, cmap="Set1", alpha=0.7, s=20)
            ax2.set_title(f"{dataset.upper()}: t-SNE colored by PAM50 subtypes")
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")

            # Create legend for PAM50
            unique_pam50 = merged["PAM50"].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_pam50)))
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=colors[i], markersize=8,
                                         label=unique_pam50[i]) for i in range(len(unique_pam50))]
            ax2.legend(handles=legend_elements, title="PAM50", bbox_to_anchor=(1.05, 1))

            plt.tight_layout()
            path = output_dir / f"tsne_clusters_{dataset}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved t-SNE plots for {dataset} to {path}")

        except Exception as e:
            print(f"[SKIP] t-SNE failed for {dataset}: {e}")

        # UMAP (if available)
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                umap_coords = reducer.fit_transform(expr_subset.values)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # UMAP colored by communities
                scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                     c=merged["community"], cmap="tab10", alpha=0.7, s=20)
                ax1.set_title(f"{dataset.upper()}: UMAP colored by BIGCLAM clusters")
                ax1.set_xlabel("UMAP 1")
                ax1.set_ylabel("UMAP 2")
                plt.colorbar(scatter1, ax=ax1, label="Cluster")

                # UMAP colored by PAM50
                scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                     c=pam50_colors, cmap="Set1", alpha=0.7, s=20)
                ax2.set_title(f"{dataset.upper()}: UMAP colored by PAM50 subtypes")
                ax2.set_xlabel("UMAP 1")
                ax2.set_ylabel("UMAP 2")
                ax2.legend(handles=legend_elements, title="PAM50", bbox_to_anchor=(1.05, 1))

                plt.tight_layout()
                path = output_dir / f"umap_clusters_{dataset}.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[OK] Saved UMAP plots for {dataset} to {path}")

            except Exception as e:
                print(f"[SKIP] UMAP failed for {dataset}: {e}")


def make_sankey_diagram(output_dir, summary):
    """Create Sankey diagram for cluster-PAM50 mapping."""
    if not PLOTLY_AVAILABLE:
        print("[SKIP] Plotly not available for Sankey diagram")
        return

    output_dir = Path(output_dir)

    for dataset in ["tcga", "gse96058"]:
        data = summary[summary["dataset"] == dataset]

        # Create contingency table
        contingency = pd.crosstab(data["community"], data["PAM50"])

        # Prepare Sankey data
        sources = []
        targets = []
        values = []
        labels = []

        # Add cluster nodes
        cluster_labels = [f"Cluster {i}" for i in contingency.index]
        pam50_labels = contingency.columns.tolist()
        labels = cluster_labels + pam50_labels

        # Create flows
        for i, cluster in enumerate(contingency.index):
            for j, pam50 in enumerate(contingency.columns):
                count = contingency.loc[cluster, pam50]
                if count > 0:
                    sources.append(i)  # cluster index
                    targets.append(len(cluster_labels) + j)  # pam50 index
                    values.append(count)

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["lightblue"] * len(cluster_labels) + ["lightcoral"] * len(pam50_labels)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])

        fig.update_layout(
            title_text=f"{dataset.upper()}: BIGCLAM Clusters → PAM50 Subtypes",
            font_size=10
        )

        path = output_dir / f"sankey_cluster_pam50_{dataset}.png"
        try:
            fig.write_image(str(path), format='png')
            print(f"[OK] Saved Sankey diagram for {dataset} to {path}")
        except Exception as e:
            print(f"[Warning] Could not save PNG (kaleido not available?): {e}")
            # Fallback to HTML
            html_path = output_dir / f"sankey_cluster_pam50_{dataset}.html"
            fig.write_html(str(html_path))
            print(f"[OK] Saved Sankey diagram for {dataset} to {html_path} (HTML fallback)")


def make_differential_expression_heatmap(output_dir):
    """Create heatmap of differential expression and pathway enrichment results."""
    output_dir = Path(output_dir)
    bio_dir = Path("results/biological_interpretation")

    for dataset in ["tcga", "gse96058"]:
        dataset_dir = bio_dir / dataset
        if not dataset_dir.exists():
            continue

        # Find differential expression files
        de_files = list(dataset_dir.glob("cluster_*_differential_expression*.csv"))
        pathway_files = list(dataset_dir.glob("cluster_*_pathways*.csv"))

        if not de_files:
            continue

        # Collect top DE genes and pathways for each cluster
        cluster_data = {}

        for de_file in de_files:
            try:
                df = pd.read_csv(de_file)
                if df.empty:
                    continue

                # Extract cluster number from filename
                cluster_match = de_file.name.split("_")[1]
                cluster = f"Cluster {cluster_match}"

                # Get top 10 most significant genes
                if "padj" in df.columns:
                    df = df.sort_values("padj").head(10)
                elif "pvalue" in df.columns:
                    df = df.sort_values("pvalue").head(10)
                else:
                    df = df.head(10)

                cluster_data[cluster] = df

            except Exception as e:
                print(f"[Warning] Could not read {de_file}: {e}")
                continue

        if not cluster_data:
            continue

        # Create a comprehensive heatmap
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

        # Differential expression heatmap
        all_genes = set()
        for df in cluster_data.values():
            if "gene" in df.columns:
                all_genes.update(df["gene"].head(5))

        all_genes = list(all_genes)[:20]  # Limit to top 20 genes

        if all_genes:
            # Create expression matrix for top genes
            de_matrix = pd.DataFrame(index=all_genes, columns=cluster_data.keys())

            for cluster, df in cluster_data.items():
                if "gene" in df.columns and "log2FoldChange" in df.columns:
                    for _, row in df.iterrows():
                        if row["gene"] in all_genes:
                            de_matrix.loc[row["gene"], cluster] = row["log2FoldChange"]

            de_matrix = de_matrix.fillna(0).astype(float)

            sns.heatmap(de_matrix.T, cmap="RdYlBu_r", center=0, ax=axes[0],
                       cbar_kws={'label': 'Log2 Fold Change'})
            axes[0].set_title(f"{dataset.upper()}: Top Differentially Expressed Genes by Cluster")
            axes[0].set_xlabel("Gene")
            axes[0].set_ylabel("BIGCLAM Cluster")

        # Pathway enrichment summary
        pathway_data = []
        for pathway_file in pathway_files:
            try:
                df = pd.read_csv(pathway_file)
                if df.empty:
                    continue

                cluster_match = pathway_file.name.split("_")[1]
                cluster = f"Cluster {cluster_match}"

                # Get top pathways
                if "Adjusted P-value" in df.columns:
                    df = df.sort_values("Adjusted P-value").head(3)
                    for _, row in df.iterrows():
                        pathway_data.append({
                            'cluster': cluster,
                            'pathway': str(row.iloc[0])[:50],  # Truncate long names
                            'score': -np.log10(row["Adjusted P-value"])
                        })

            except Exception as e:
                continue

        if pathway_data:
            pathway_df = pd.DataFrame(pathway_data)
            pathway_pivot = pathway_df.pivot_table(
                index='cluster', columns='pathway', values='score', fill_value=0
            )

            sns.heatmap(pathway_pivot, cmap="Reds", ax=axes[1],
                       cbar_kws={'label': '-log10(FDR)'})
            axes[1].set_title("Top Enriched Pathways by Cluster")
            axes[1].set_xlabel("Pathway")
            axes[1].set_ylabel("BIGCLAM Cluster")

        plt.tight_layout()
        path = output_dir / f"de_pathway_heatmap_{dataset}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved DE/pathway heatmap for {dataset} to {path}")


def make_kaplan_meier_curves(output_dir):
    """Generate Kaplan-Meier survival curves by cluster."""
    if not LIFELINES_AVAILABLE:
        print("[SKIP] Lifelines not available for survival analysis")
        return

    output_dir = Path(output_dir)

    # Load TCGA clinical data (has survival data)
    survival_dir = Path("results/survival")
    clinical_file = survival_dir / "tcga_clinical_data_processed.csv"

    if not clinical_file.exists():
        print("[SKIP] TCGA clinical data not found for survival analysis")
        return

    try:
        clinical_df = pd.read_csv(clinical_file)

        # Load cluster assignments
        cluster_file = Path("data/clusterings/tcga_brca_data_communities.npy")
        if not cluster_file.exists():
            print("[SKIP] TCGA cluster assignments not found")
            return

        communities = np.load(cluster_file)
        if communities.ndim > 1:
            communities = np.argmax(communities, axis=1)

        # Merge with clinical data
        clinical_df['community'] = communities

        # Required columns for survival analysis
        required_cols = ['OS.time', 'OS', 'community']
        if not all(col in clinical_df.columns for col in required_cols):
            print("[SKIP] Required survival columns not found")
            return

        # Filter out missing survival data
        survival_data = clinical_df.dropna(subset=['OS.time', 'OS'])

        if len(survival_data) < 10:
            print("[SKIP] Insufficient survival data")
            return

        # Create Kaplan-Meier plot
        fig, ax = plt.subplots(figsize=(10, 8))

        kmf = KaplanMeierFitter()
        colors = plt.cm.tab10(np.linspace(0, 1, len(survival_data['community'].unique())))

        # Plot KM curve for each cluster
        for i, cluster in enumerate(sorted(survival_data['community'].unique())):
            cluster_data = survival_data[survival_data['community'] == cluster]

            if len(cluster_data) < 5:  # Need minimum samples
                continue

            kmf.fit(cluster_data['OS.time'], cluster_data['OS'],
                   label=f'Cluster {cluster} (n={len(cluster_data)})')

            kmf.plot(ax=ax, color=colors[i], ci_show=False)

        ax.set_title('TCGA-BRCA: Kaplan-Meier Survival Curves by BIGCLAM Cluster')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        path = output_dir / "kaplan_meier_tcga.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved Kaplan-Meier curves to {path}")

        # Statistical comparison (log-rank test)
        unique_clusters = sorted(survival_data['community'].unique())
        if len(unique_clusters) >= 2:
            # Compare first two clusters as example
            c1_data = survival_data[survival_data['community'] == unique_clusters[0]]
            c2_data = survival_data[survival_data['community'] == unique_clusters[1]]

            if len(c1_data) >= 5 and len(c2_data) >= 5:
                results = logrank_test(
                    c1_data['OS.time'], c2_data['OS.time'],
                    c1_data['OS'], c2_data['OS']
                )
                print(".3f")

    except Exception as e:
        print(f"[SKIP] Survival analysis failed: {e}")


def make_method_comparison_barplot(output_dir):
    """Create barplot comparing ARI/NMI scores vs other clustering methods."""
    output_dir = Path(output_dir)

    # Look for method comparison results
    comparison_files = [
        "results/method_comparison/method_comparison_results.csv",
        "results/method_comparison/comprehensive_method_comparison.csv",
        "results/evaluation/tcga_brca_data_evaluation_results.csv",
        "results/evaluation/gse96058_data_evaluation_results.csv"
    ]

    data_found = False

    for file_path in comparison_files:
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)

                # Look for relevant columns
                ari_cols = [col for col in df.columns if 'ari' in col.lower()]
                nmi_cols = [col for col in df.columns if 'nmi' in col.lower()]
                method_cols = [col for col in df.columns if 'method' in col.lower() or 'algorithm' in col.lower()]

                if ari_cols and nmi_cols and method_cols:
                    data_found = True

                    # Create comparison plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # ARI comparison
                    methods = df[method_cols[0]].values
                    ari_values = df[ari_cols[0]].values

                    bars1 = ax1.bar(range(len(methods)), ari_values, color='skyblue', alpha=0.7)
                    ax1.set_xlabel('Clustering Method')
                    ax1.set_ylabel('Adjusted Rand Index (ARI)')
                    ax1.set_title('ARI Comparison Across Methods')
                    ax1.set_xticks(range(len(methods)))
                    ax1.set_xticklabels(methods, rotation=45, ha='right')
                    ax1.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, val in zip(bars1, ari_values):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                '.3f', ha='center', va='bottom')

                    # NMI comparison
                    nmi_values = df[nmi_cols[0]].values

                    bars2 = ax2.bar(range(len(methods)), nmi_values, color='lightcoral', alpha=0.7)
                    ax2.set_xlabel('Clustering Method')
                    ax2.set_ylabel('Normalized Mutual Information (NMI)')
                    ax2.set_title('NMI Comparison Across Methods')
                    ax2.set_xticks(range(len(methods)))
                    ax2.set_xticklabels(methods, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, val in zip(bars2, nmi_values):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                '.3f', ha='center', va='bottom')

                    plt.suptitle('BIGCLAM vs Other Clustering Methods', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    path = output_dir / "method_comparison_barplot.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"[OK] Saved method comparison barplot to {path}")
                    break

            except Exception as e:
                print(f"[Warning] Could not read {file_path}: {e}")
                continue

    if not data_found:
        print("[SKIP] No method comparison data found")


def make_cross_dataset_analysis(output_dir):
    """Create cross-dataset correlation heatmap and overlap analysis."""
    output_dir = Path(output_dir)

    # Load cluster assignments for both datasets
    tcga_clusters_file = Path("data/clusterings/tcga_brca_data_communities.npy")
    gse_clusters_file = Path("data/clusterings/gse96058_data_communities.npy")

    if not tcga_clusters_file.exists() or not gse_clusters_file.exists():
        print("[SKIP] Cluster assignment files not found for cross-dataset analysis")
        return

    try:
        tcga_clusters = np.load(tcga_clusters_file)
        gse_clusters = np.load(gse_clusters_file)

        if tcga_clusters.ndim > 1:
            tcga_clusters = np.argmax(tcga_clusters, axis=1)
        if gse_clusters.ndim > 1:
            gse_clusters = np.argmax(gse_clusters, axis=1)

        # Load PAM50 labels
        config = load_config()
        tcga_summary = create_summary_df("tcga", config)
        gse_summary = create_summary_df("gse96058", config)

        # Create contingency table
        contingency = pd.crosstab(tcga_summary['community'], gse_summary['community'])

        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Cross-dataset cluster correlation heatmap
        sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   cbar_kws={'label': 'Sample count'})
        ax1.set_title('BIGCLAM Clusters: TCGA-BRCA vs GSE96058')
        ax1.set_xlabel('GSE96058 Clusters')
        ax1.set_ylabel('TCGA-BRCA Clusters')

        # PAM50 distribution comparison
        tcga_pam50 = tcga_summary['PAM50'].value_counts()
        gse_pam50 = gse_summary['PAM50'].value_counts()

        pam50_comparison = pd.DataFrame({
            'TCGA-BRCA': tcga_pam50,
            'GSE96058': gse_pam50
        }).fillna(0)

        pam50_comparison.plot(kind='bar', ax=ax2, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax2.set_title('PAM50 Subtype Distribution Across Datasets')
        ax2.set_xlabel('PAM50 Subtype')
        ax2.set_ylabel('Sample Count')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        path = output_dir / "cross_dataset_analysis.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved cross-dataset analysis to {path}")

        # Print summary statistics
        print(f"Cross-dataset analysis summary:")
        print(f"  TCGA-BRCA: {len(tcga_clusters)} samples, {len(set(tcga_clusters))} clusters")
        print(f"  GSE96058: {len(gse_clusters)} samples, {len(set(gse_clusters))} clusters")
        print(f"  Total overlap analysis completed")

    except Exception as e:
        print(f"[SKIP] Cross-dataset analysis failed: {e}")

def make_figures(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["tcga", "gse96058"]
    summary = pd.concat(
        [create_summary_df(ds, config) for ds in datasets], ignore_index=True
    )

    print("Generating comprehensive paper figures...")
    print("=" * 60)

    # 1. Original cluster vs PAM50 figures
    print("\n1. Creating cluster vs PAM50 comparison figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, dataset in enumerate(datasets):
        data = summary[summary["dataset"] == dataset]
        plot_confusion_pam50(data, dataset, axes[idx, 0])
        plot_distribution(data, dataset, axes[idx, 1])
    plt.tight_layout()
    fig_path = output_dir / "figure_cluster_vs_pam50.png"
    fig.savefig(fig_path, dpi=300)
    print(f"[OK] Saved cluster vs PAM50 figure to {fig_path}")

    # 2. t-SNE and UMAP plots colored by BIGCLAM clusters
    print("\n2. Creating t-SNE/UMAP plots colored by BIGCLAM clusters...")
    make_tsne_umap_plots(output_dir, config)

    # 3. Cluster-PAM50 mapping (Sankey diagrams)
    print("\n3. Creating Sankey diagrams for cluster-PAM50 mapping...")
    make_sankey_diagram(output_dir, summary)

    # 4. Differential expression and pathway heatmaps
    print("\n4. Creating differential expression and pathway heatmaps...")
    make_differential_expression_heatmap(output_dir)

    # 5. Kaplan-Meier survival curves by cluster
    print("\n5. Creating Kaplan-Meier survival curves...")
    make_kaplan_meier_curves(output_dir)

    # 6. ARI/NMI comparison barplot vs other methods
    print("\n6. Creating method comparison barplots...")
    make_method_comparison_barplot(output_dir)

    # 7. Cross-dataset correlation heatmap and overlap analysis
    print("\n7. Creating cross-dataset analysis...")
    make_cross_dataset_analysis(output_dir)

    # 8. Original additional figures
    print("\n8. Creating additional original figures...")
    make_cluster_pam50_heatmap(output_dir, summary)
    make_survival_summary(output_dir, summary)
    make_pathway_enrichment(output_dir)
    make_classifier_bars(output_dir)
    make_graph_layout(output_dir, summary, config)

    print("\n" + "=" * 60)
    print("All paper figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


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

