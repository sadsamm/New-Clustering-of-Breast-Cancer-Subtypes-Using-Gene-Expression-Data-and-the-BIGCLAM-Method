"""
Graph Construction Module (Mutual kNN on PCA)

Builds mutual kNN graphs from preprocessed expression data for BIGCLAM input.
- PCA dimensionality reduction (default 50 PCs)
- Mutual kNN (default k=20) in PCA space, Euclidean distance
- Sparse adjacency output
"""

import numpy as np
from pathlib import Path
import gc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def build_mutual_knn_graph(
    expression_data,
    pca_components=50,
    k=20,
    metric="euclidean",
    use_sparse=True,
):
    """
    Build a mutual k-nearest-neighbor graph on PCA-reduced expression data.
    
    Args:
        expression_data: numpy array (n_samples x n_features)
        pca_components: number of PCA components to keep before kNN
        k: number of neighbors for mutual kNN (10–30 recommended)
        metric: distance metric for kNN in PCA space
        use_sparse: whether to return a sparse adjacency
        
    Returns:
        tuple: (adjacency_matrix, pca_embedding)
    """
    print("\n[Graph Construction - Mutual kNN on PCA]...")
    print(f"    Data shape: {expression_data.shape}")
    print(f"    PCA components: {pca_components}, kNN k: {k}, metric: {metric}")
    
    n_samples = expression_data.shape[0]
    if n_samples < 2:
        raise ValueError("Not enough samples to build a graph.")

    k_eff = min(k, max(1, n_samples - 1))
    pca_n = min(pca_components, expression_data.shape[1], n_samples)

    pca = PCA(n_components=pca_n, random_state=42)
    data_pca = pca.fit_transform(expression_data)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(data_pca)
    _, indices = nn.kneighbors(data_pca)

    rows = []
    cols = []
    for i in range(n_samples):
        # skip self at position 0
        for j in indices[i, 1:]:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.int8)
    directed = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    mutual = directed.multiply(directed.T)
    mutual.setdiag(0)
    mutual.eliminate_zeros()

    if not use_sparse:
        mutual = mutual.toarray()

    n_edges = mutual.nnz if hasattr(mutual, "nnz") else (mutual > 0).sum()
    density = n_edges / (n_samples * n_samples) * 100
    print(f"    Edges (mutual): {n_edges:,} ({density:.2f}% density)")
    
    gc.collect()
    return mutual, data_pca

    


def save_graph_data(adjacency, output_file):
    """
    Save graph adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (sparse or dense)
        output_file: Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(adjacency, 'nnz'):
        # Sparse matrix
        import scipy.sparse
        scipy.sparse.save_npz(output_file, adjacency, compressed=True)
        print(f"    Saved sparse graph to: {output_file}")
    else:
        # Dense matrix
        np.save(output_file, adjacency)
        print(f"    Saved dense graph to: {output_file}")


def load_graph_data(input_file):
    """
    Load graph adjacency matrix.
    
    Args:
        input_file: Input file path
        
    Returns:
        Adjacency matrix
    """
    input_file = Path(input_file)
    
    if input_file.suffix == '.npz':
        import scipy.sparse
        adjacency = scipy.sparse.load_npz(input_file)
        print(f"    Loaded sparse graph from: {input_file}")
    else:
        adjacency = np.load(input_file)
        print(f"    Loaded dense graph from: {input_file}")
    
    return adjacency


def construct_graphs(
    input_dir='data/processed',
    output_dir='data/graphs',
    use_sparse=True,
    save_embedding=False,
    mutual_knn_params=None,
):
    """
    Construct graphs for all processed datasets.
    
    Args:
        input_dir: Directory containing processed data
        output_dir: Directory to save graphs
        use_sparse: Whether to use sparse matrices
        mutual_knn_params: dict with PCA/kNN settings
        
    Returns:
        dict: Graph data for each dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mutual_knn_params = mutual_knn_params or {}
    
    results = {}
    
    # Find all processed files
    processed_files = list(input_dir.glob('*_processed.npy'))
    
    if not processed_files:
        print(f"No processed files found in {input_dir}")
        return results
    
    for processed_file in processed_files:
        print("\n" + "="*80)
        dataset_name = processed_file.stem.replace('_processed', '')
        print(f"CONSTRUCTING GRAPH: {dataset_name}")
        print("="*80)
        # Load expression data
        expression_data = np.load(processed_file)
        
        pca_components = int(mutual_knn_params.get("pca_components", 50))
        k_neighbors = int(mutual_knn_params.get("k", 20))
        metric = mutual_knn_params.get("metric", "euclidean")

        adjacency, embedding_final = build_mutual_knn_graph(
            expression_data, 
            pca_components=pca_components,
            k=k_neighbors,
            metric=metric,
            use_sparse=use_sparse,
        )
        similarity = None
        if save_embedding:
            emb_file = output_dir / f"{dataset_name}_pca_embedding.npy"
            np.save(emb_file, embedding_final)
            print(f"    Saved PCA embedding to: {emb_file}")
        
        # Save graph
        graph_file = output_dir / f"{dataset_name}_adjacency"
        save_graph_data(adjacency, graph_file)
        
        results[dataset_name] = {
            'adjacency': adjacency,
            'similarity': similarity,
            'n_samples': expression_data.shape[0]
        }
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Construct similarity graphs')
    parser.add_argument('--input_dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='data/graphs', help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file for dataset-specific thresholds')
    parser.add_argument('--use_sparse', action='store_true', default=True, help='Use sparse matrices')
    parser.add_argument('--no_sparse', action='store_false', dest='use_sparse', help='Use dense matrices')
    parser.add_argument('--pca_components', type=int, default=50, help='PCA components before kNN')
    parser.add_argument('--k', type=int, default=20, help='k for mutual kNN')
    parser.add_argument('--metric', type=str, default='euclidean', help='Distance metric for mutual kNN')
    parser.add_argument('--save_embedding', action='store_true', help='Save PCA embedding as .npy alongside graphs')
    
    args = parser.parse_args()
    
    construct_graphs(
        args.input_dir,
        args.output_dir,
        use_sparse=args.use_sparse,
        save_embedding=args.save_embedding,
        mutual_knn_params={
            "pca_components": args.pca_components,
            "k": args.k,
            "metric": args.metric,
        },
    )

