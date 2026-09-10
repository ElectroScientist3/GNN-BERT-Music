import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing import Dict, List, Optional, Union

class MusicGraphBuilder:
    def __init__(self, config: Dict):
        """
        Initialize the graph builder.
        
        Args:
            config: Dictionary containing graph parameters or full config.
        """
        graph_cfg = config.get('graph', {}) if isinstance(config, dict) else {}
        if not graph_cfg and isinstance(config, dict):
            graph_cfg = config

        self.similarity_threshold = graph_cfg.get('similarity_threshold', 0.7)
        self.max_edges_per_node = graph_cfg.get('max_edges_per_node', graph_cfg.get('max_edges', 10))
        self.node_feature_dim = graph_cfg.get('node_feature_dim', 260)

    def build_segment_graph(self, segment_features: np.ndarray) -> Optional[Data]:
        """
        Build a graph from segmented audio features.
        
        Args:
            segment_features: np.ndarray of shape (num_segments, feature_dim)
            
        Returns:
            torch_geometric.data.Data or None if features are invalid.
        """
        if segment_features is None or len(segment_features) == 0:
            return None
            
        # Handle NaNs by replacing with zeros
        segment_features = np.nan_to_num(segment_features, nan=0.0)
        
        num_nodes = segment_features.shape[0]
        if num_nodes < 2:
            # Graph needs at least 2 nodes for edges
            return None
            
        edge_list = []
        
        # 1. Temporal edges (bidirectional between consecutive segments)
        for i in range(num_nodes - 1):
            edge_list.append((i, i + 1))
            edge_list.append((i + 1, i))
            
        # 2. Similarity edges
        # Calculate cosine similarity matrix
        # Distance = 1 - similarity -> similarity = 1 - distance
        dists = cdist(segment_features, segment_features, metric='cosine')
        sim_matrix = 1.0 - dists
        
        # Fill diagonal with -1 so we don't add self-loops here
        np.fill_diagonal(sim_matrix, -1.0)
        
        for i in range(num_nodes):
            # Find indices of segments that exceed similarity threshold
            sim_indices = np.where(sim_matrix[i] >= self.similarity_threshold)[0]
            
            if len(sim_indices) > 0:
                # Sort by highest similarity
                sorted_sim_indices = sim_indices[np.argsort(-sim_matrix[i, sim_indices])]
                
                # Cap at max_edges_per_node
                top_k = sorted_sim_indices[:self.max_edges_per_node]
                
                for j in top_k:
                    # Avoid duplicates and self loops
                    if i != j and (i, j) not in edge_list:
                        edge_list.append((i, j))
                        # Bidirectional similarity edge
                        if (j, i) not in edge_list:
                            edge_list.append((j, i))
                            
        if not edge_list:
            # Ensure edge index shape is correct even if no edges somehow
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
        x = torch.tensor(segment_features, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = num_nodes
        return data

    def build_all_graphs(self, features_dir: str, output_dir: str, annotations_df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Process all songs, build graphs, optionally attach labels, and save.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        csv_files = glob.glob(os.path.join(features_dir, "*.csv"))
        
        for file_path in tqdm(csv_files, desc="Building graphs"):
            try:
                song_id = os.path.splitext(os.path.basename(file_path))[0]
                # Load features - assumes OpenSMILE format where rows are timeframes
                df = pd.read_csv(file_path)
                
                # Drop non-feature columns if they exist (e.g., frameTime)
                if 'frameTime' in df.columns:
                    features = df.drop(columns=['frameTime']).values
                else:
                    features = df.values
                    
                data = self.build_segment_graph(features)
                
                if data is None:
                    continue
                    
                # Attach labels if provided
                if annotations_df is not None and song_id in annotations_df.index:
                    row = annotations_df.loc[song_id]
                    if 'valence' in row:
                        data.valence = torch.tensor([row['valence']], dtype=torch.float32)
                    if 'arousal' in row:
                        data.arousal = torch.tensor([row['arousal']], dtype=torch.float32)
                    if 'tags' in row:
                        data.tags = torch.tensor([row['tags']], dtype=torch.float32) # Assuming tags are binary vector
                        
                out_path = os.path.join(output_dir, f"{song_id}.pt")
                torch.save(data, out_path)
                saved_paths.append(out_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        return saved_paths

    def compute_graph_statistics(self, graphs: List[Data]) -> Dict:
        """Compute average nodes, edges, degree per graph."""
        if not graphs:
            return {}
            
        num_nodes = [g.num_nodes for g in graphs]
        num_edges = [g.edge_index.size(1) for g in graphs]
        degrees = [e / n if n > 0 else 0 for e, n in zip(num_edges, num_nodes)]
        
        return {
            "avg_nodes": np.mean(num_nodes),
            "max_nodes": np.max(num_nodes),
            "min_nodes": np.min(num_nodes),
            "avg_edges": np.mean(num_edges),
            "avg_degree": np.mean(degrees)
        }

    def load_graph(self, path: str) -> Data:
        """Load a saved .pt graph."""
        return torch.load(path)

def build_graph_from_opensmile(csv_path: str, config: Dict) -> Optional[Data]:
    """Convenience function to build a graph from a single OpenSMILE csv."""
    builder = MusicGraphBuilder(config)
    
    try:
        df = pd.read_csv(csv_path)
        if 'frameTime' in df.columns:
            features = df.drop(columns=['frameTime']).values
        else:
            features = df.values
            
        return builder.build_segment_graph(features)
    except Exception as e:
        print(f"Error building graph from {csv_path}: {e}")
        return None

if __name__ == '__main__':
    # Simple test
    print("Testing MusicGraphBuilder...")
    config = {
        'similarity_threshold': 0.8,
        'max_edges': 3,
        'node_feature_dim': 260
    }
    builder = MusicGraphBuilder(config)
    
    # Dummy features: 10 segments, 260 features
    dummy_features = np.random.rand(10, 260)
    
    graph = builder.build_segment_graph(dummy_features)
    print(f"Graph built successfully: {graph}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.edge_index.size(1)}")
    
    stats = builder.compute_graph_statistics([graph])
    print(f"Stats: {stats}")
