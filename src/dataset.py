import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyGDataset, Batch, Data
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Any

class MusicTagDataset(Dataset):
    """Dataset for Task 1: BERT based Music Tagging."""
    def __init__(self, texts: List[str], tags: np.ndarray, tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'tag_vector': torch.tensor(self.tags[idx], dtype=torch.float32)
        }


class MusicGraphDataset(PyGDataset):
    """Dataset for Task 2: GNN based Music Processing."""
    def __init__(self, graph_dir: str, song_ids: List[str], labels_dict: Dict[str, np.ndarray]):
        super().__init__()
        self.graph_dir = graph_dir
        self.song_ids = song_ids
        self.labels_dict = labels_dict

    def len(self):
        return len(self.song_ids)

    def get(self, idx):
        song_id = self.song_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{song_id}.pt")
        
        # Load the graph
        data = torch.load(graph_path)
        
        # Attach labels
        if song_id in self.labels_dict:
            data.y = torch.tensor(self.labels_dict[song_id], dtype=torch.float32)
            
        return data


class MusicFusionDataset(Dataset):
    """Dataset for Task 3: GNN+BERT Fusion (Predicting Valence/Arousal)."""
    def __init__(
        self, 
        graph_dir: str, 
        texts: List[str], 
        tags: np.ndarray,
        valence: List[float],
        arousal: List[float],
        tokenizer: Any, 
        song_ids: List[str], 
        max_length: int = 128
    ):
        self.graph_dir = graph_dir
        self.texts = texts
        self.tags = tags
        self.valence = valence
        self.arousal = arousal
        self.tokenizer = tokenizer
        self.song_ids = song_ids
        self.max_length = max_length

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]
        text = str(self.texts[idx])
        
        # Load Graph
        graph_path = os.path.join(self.graph_dir, f"{song_id}.pt")
        graph_data = torch.load(graph_path)
        
        # Tokenize Text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'graph': graph_data,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'tag_vector': torch.tensor(self.tags[idx], dtype=torch.float32),
            'valence': torch.tensor([self.valence[idx]], dtype=torch.float32),
            'arousal': torch.tensor([self.arousal[idx]], dtype=torch.float32)
        }


class MusicContrastiveDataset(Dataset):
    """Dataset for Task 4: Contrastive Learning (Graph-Text pairs)."""
    def __init__(self, graph_dir: str, texts: List[str], tokenizer: Any, song_ids: List[str], max_length: int = 128):
        self.graph_dir = graph_dir
        self.texts = texts
        self.tokenizer = tokenizer
        self.song_ids = song_ids
        self.max_length = max_length

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]
        text = str(self.texts[idx])
        
        # Load Graph
        graph_path = os.path.join(self.graph_dir, f"{song_id}.pt")
        graph_data = torch.load(graph_path)
        
        # Tokenize Text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'graph': graph_data,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def create_splits(song_ids: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42, splits_dir: str = None) -> Dict:
    """Random split into train/val/test and save to JSON."""
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    
    train_val_ids, test_ids = train_test_split(song_ids, test_size=test_ratio, random_state=seed)
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_rel_ratio, random_state=seed)
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    if splits_dir:
        os.makedirs(splits_dir, exist_ok=True)
        with open(os.path.join(splits_dir, 'splits.json'), 'w') as f:
            json.dump(splits, f)
            
    return splits

def load_splits(splits_dir: str) -> Dict:
    """Load saved splits from JSON."""
    with open(os.path.join(splits_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
    return splits


def fusion_collate_fn(batch):
    """Custom collate function for MusicFusionDataset and MusicContrastiveDataset."""
    graphs = [item['graph'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Batch graphs using PyTorch Geometric's Batch
    batched_graphs = Batch.from_data_list(graphs)
    
    result = {
        'graph': batched_graphs,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    if 'tag_vector' in batch[0]:
        result['tag_vector'] = torch.stack([item['tag_vector'] for item in batch])
    if 'valence' in batch[0]:
        result['valence'] = torch.stack([item['valence'] for item in batch])
    if 'arousal' in batch[0]:
        result['arousal'] = torch.stack([item['arousal'] for item in batch])
        
    return result


if __name__ == '__main__':
    print("Testing dataset module...")
    
    # Simple dummy testing
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {
                'input_ids': torch.zeros((1, 128), dtype=torch.long),
                'attention_mask': torch.ones((1, 128), dtype=torch.long)
            }
            
    tokenizer = DummyTokenizer()
    texts = ["happy song", "sad song"]
    tags = np.random.randint(0, 2, (2, 50))
    song_ids = ["song1", "song2"]
    
    dataset = MusicTagDataset(texts, tags, tokenizer)
    print("MusicTagDataset len:", len(dataset))
    print("Sample:", dataset[0]['input_ids'].shape)
    
    splits = create_splits(song_ids, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    print("Splits:", splits)
