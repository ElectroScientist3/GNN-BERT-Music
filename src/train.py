"""
Unified training script for GNN-BERT Music Context Understanding.
Supports all 4 tasks: BERT tagger, GNN classifier, GNN-BERT fusion, Contrastive dual-encoder.

Usage:
    python src/train.py --task 1 --epochs 20
    python src/train.py --task 2 --epochs 30
    python src/train.py --task 3 --epochs 30
    python src/train.py --task 4 --epochs 30
"""

import argparse
import yaml
import json
import os
import sys
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

# Project imports
from tag_generator import TagGenerator
from audio_features import AudioFeatureExtractor
from graph_builder import MusicGraphBuilder
from dataset import (
    MusicTagDataset, MusicGraphDataset, MusicFusionDataset,
    MusicContrastiveDataset, create_splits, load_splits, fusion_collate_fn
)
from bert_encoder import BERTTagClassifier
from gnn_model import GraphSAGEEncoder, GATEncoder, GNNClassifier
from cnn_baseline import CNNBaseline
from fusion_model import GNNBERTFusionModel
from contrastive import ContrastiveDualEncoder, compute_retrieval_metrics
from evaluate import Evaluator


def load_config(path='config.yaml'):
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using defaults.")
        return {
            'data': {'raw_dir': 'data/raw', 'processed_dir': 'data/processed',
                     'splits_dir': 'data/splits'},
            'graph': {'similarity_threshold': 0.7, 'max_edges_per_node': 10,
                      'node_feature_dim': 260},
            'bert': {'model_name': 'bert-base-uncased', 'max_length': 128,
                     'freeze_layers': 8},
            'gnn': {'hidden_dims': [128, 64, 32], 'dropout': 0.3},
            'fusion': {'fused_dim': 128, 'cross_attention_heads': 4,
                       'alpha': 0.3, 'beta': 0.3},
            'contrastive': {'embedding_dim': 256, 'temperature': 0.07},
            'training': {'batch_size': 16, 'lr': 2e-4, 'bert_lr': 2e-5,
                         'epochs': 30, 'patience': 10, 'weight_decay': 1e-4},
            'tags': {
                'mood_tags': ['happy', 'sad', 'angry', 'relaxed', 'energetic',
                              'calm', 'tense', 'peaceful'],
                'genre_tags': ['electronic', 'classical', 'rock', 'ambient',
                               'pop', 'jazz', 'folk', 'metal']
            }
        }
    with open(path) as f:
        return yaml.safe_load(f)


def extract_data(config):
    """Extract zip files if not already extracted."""
    raw_dir = config['data'].get('raw_dir', 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)

    # Try to find and extract zip files from parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_mappings = {
        'DEAM_Annotations.zip': os.path.join(raw_dir, 'annotations'),
        'features.zip': os.path.join(raw_dir, 'features'),
    }

    for zip_name, extract_check in zip_mappings.items():
        # Check multiple possible locations for the zip
        possible_paths = [
            os.path.join(parent_dir, zip_name),
            os.path.join(os.path.dirname(parent_dir), zip_name),
            os.path.join(raw_dir, zip_name),
        ]
        if os.path.exists(extract_check):
            continue

        for zip_path in possible_paths:
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path} -> {raw_dir}")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(raw_dir)
                break


def prepare_data(config):
    """
    Prepare all data: extract zips, generate tags, build graphs, create splits.
    Returns: tag_data dict, tag_vocabulary, splits dict
    """
    raw_dir = config['data'].get('raw_dir', 'data/raw')
    processed_dir = config['data'].get('processed_dir', 'data/processed')
    splits_dir = config['data'].get('splits_dir', 'data/splits')
    graph_dir = os.path.join(processed_dir, 'graphs')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # Step 1: Extract data
    extract_data(config)

    # Step 2: Generate tags from valence/arousal
    tag_cache = os.path.join(processed_dir, 'tag_data.json')
    tag_gen = TagGenerator(config)

    alt_tag_cache = os.path.join(processed_dir, 'tags_and_descriptions.json')
    tag_data = {}
    if os.path.exists(tag_cache) and os.path.getsize(tag_cache) > 10:
        with open(tag_cache, 'r') as f:
            tag_data = json.load(f)
    elif os.path.exists(alt_tag_cache) and os.path.getsize(alt_tag_cache) > 10:
        with open(alt_tag_cache, 'r') as f:
            tag_data = json.load(f)

    if not tag_data:
        print("Generating tags from valence/arousal annotations...")
        tag_data = tag_gen.generate_all()
        with open(tag_cache, 'w') as f:
            json.dump(tag_data, f)
        print(f"Generated tags for {len(tag_data)} songs.")

    tag_vocab = tag_gen.get_tag_vocabulary()

    # Step 3: Build graphs from OpenSMILE features
    features_dir = os.path.join(raw_dir, 'features')
    if not os.path.exists(features_dir):
        print("Warning: features directory not found. Looking in alternative paths...")
        for alt in [os.path.join(raw_dir, 'features'),
                    os.path.join(os.path.dirname(raw_dir), 'features')]:
            if os.path.exists(alt):
                features_dir = alt
                break

    existing_graphs = [f.replace('.pt', '') for f in os.listdir(graph_dir) if f.endswith('.pt')]

    if len(existing_graphs) < 10:
        print("Building music segment graphs...")
        builder = MusicGraphBuilder(config)
        extractor = AudioFeatureExtractor(config)

        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.csv')]
        built_count = 0

        for feat_file in tqdm(feature_files, desc="Building graphs"):
            song_id = feat_file.replace('.csv', '')
            graph_path = os.path.join(graph_dir, f"{song_id}.pt")

            if os.path.exists(graph_path):
                continue

            try:
                feat_path = os.path.join(features_dir, feat_file)
                raw_features = extractor.load_opensmile_features(feat_path)
                segments = extractor.segment_opensmile_features(raw_features)

                if len(segments) < 2:
                    continue

                segment_array = np.array(segments, dtype=np.float32)
                # Replace NaN/Inf with 0
                segment_array = np.nan_to_num(segment_array, nan=0.0, posinf=0.0, neginf=0.0)
                graph = builder.build_segment_graph(segment_array)

                # Attach labels if available
                if str(song_id) in tag_data:
                    info = tag_data[str(song_id)]
                    tag_vector = np.zeros(len(tag_vocab), dtype=np.float32)
                    for tag in info.get('mood_tags', []) + info.get('genre_tags', []):
                        if tag in tag_vocab:
                            tag_vector[tag_vocab.index(tag)] = 1.0
                    graph.y = torch.tensor(tag_vector)
                    graph.valence = torch.tensor([info['valence']], dtype=torch.float32)
                    graph.arousal = torch.tensor([info['arousal']], dtype=torch.float32)

                torch.save(graph, graph_path)
                built_count += 1
            except Exception as e:
                print(f"Error processing {feat_file}: {e}")
                continue

        print(f"Built {built_count} new graphs. Total: {len(os.listdir(graph_dir))} graphs.")
    else:
        print(f"Found {len(existing_graphs)} existing graphs.")

    # Step 4: Create or load splits
    splits_path = os.path.join(splits_dir, 'splits.json')
    graph_ids = [f.replace('.pt', '') for f in os.listdir(graph_dir) if f.endswith('.pt')]
    # Only keep songs that have both graph and tag data
    valid_ids = [sid for sid in graph_ids if str(sid) in tag_data]

    if os.path.exists(splits_path):
        print("Loading existing splits...")
        splits = load_splits(splits_dir)
        # Validate splits
        for key in splits:
            splits[key] = [sid for sid in splits[key] if sid in valid_ids]
    else:
        print(f"Creating train/val/test splits from {len(valid_ids)} songs...")
        splits = create_splits(valid_ids, splits_dir=splits_dir)

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return tag_data, tag_vocab, splits, graph_dir


def get_song_data(tag_data, tag_vocab, song_ids):
    """Extract texts, tag vectors, valence, arousal for a list of song IDs."""
    texts, tags, valences, arousals = [], [], [], []
    for sid in song_ids:
        info = tag_data[str(sid)]
        texts.append(info['text_description'])
        tag_vector = np.zeros(len(tag_vocab), dtype=np.float32)
        for tag in info.get('mood_tags', []) + info.get('genre_tags', []):
            if tag in tag_vocab:
                tag_vector[tag_vocab.index(tag)] = 1.0
        tags.append(tag_vector)
        valences.append(info['valence'])
        arousals.append(info['arousal'])
    return texts, np.array(tags), valences, arousals


# ========================
# Task 1: BERT Tag Classifier
# ========================
def train_task1(config, device):
    """Train BERT multi-label tag classifier."""
    print("\n" + "=" * 60)
    print("TASK 1: BERT Multi-Label Tag Classifier")
    print("=" * 60)

    from transformers import BertTokenizer

    tag_data, tag_vocab, splits, graph_dir = prepare_data(config)
    num_tags = len(tag_vocab)
    tokenizer = BertTokenizer.from_pretrained(config['bert']['model_name'])

    # Create datasets
    for split_name in ['train', 'val', 'test']:
        texts, tags, _, _ = get_song_data(tag_data, tag_vocab, splits[split_name])
        if split_name == 'train':
            train_ds = MusicTagDataset(texts, tags, tokenizer, config['bert']['max_length'])
        elif split_name == 'val':
            val_ds = MusicTagDataset(texts, tags, tokenizer, config['bert']['max_length'])
        else:
            test_ds = MusicTagDataset(texts, tags, tokenizer, config['bert']['max_length'])

    bs = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    # Model
    model = BERTTagClassifier(
        model_name=config['bert']['model_name'],
        num_tags=num_tags,
        freeze_layers=config['bert'].get('freeze_layers', 8)
    ).to(device)

    # Optimizer with separate LR for BERT and head
    bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'bert' not in n]
    optimizer = AdamW([
        {'params': bert_params, 'lr': config['training'].get('bert_lr', 2e-5)},
        {'params': head_params, 'lr': config['training']['lr']}
    ], weight_decay=config['training'].get('weight_decay', 1e-4))

    criterion = nn.BCEWithLogitsLoss()
    epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0
    patience_counter = 0
    train_f1s, val_f1s = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['tag_vector'].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        scheduler.step()
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        train_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)
        train_f1s.append(train_f1)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['tag_vector'].to(device)

                logits, _, _ = model(input_ids, attention_mask)
                val_loss += criterion(logits, labels).item()
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), average='macro', zero_division=0)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train F1: {train_f1:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'results/bert_tag_classifier.pt')
            print("  -> Saved best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training'].get('patience', 10):
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    model.load_state_dict(torch.load('results/bert_tag_classifier.pt', map_location=device))
    model.eval()
    test_preds, test_labels, test_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['tag_vector'].to(device)
            logits, _, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)
            test_preds.append((probs > 0.5).astype(int))
            test_labels.append(labels.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)
    test_probs = np.vstack(test_probs)

    evaluator = Evaluator(config, 'results')
    metrics = evaluator.compute_classification_metrics(test_labels, test_preds, test_probs)
    metrics['train_f1_history'] = train_f1s
    metrics['val_f1_history'] = val_f1s

    # Save results
    with open('results/metrics_task1.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    evaluator.plot_f1_curves(train_f1s, val_f1s, 'plots/task1_f1_curves.png')
    evaluator.plot_auc_pr_curve(test_labels, test_probs, tag_vocab, 'plots/task1_pr_curves.png')

    print(f"\nTask 1 Results: Macro-F1={metrics['macro_f1']:.4f}, Micro-F1={metrics['micro_f1']:.4f}, "
          f"AUC-PR={metrics.get('auc_pr_macro', 'N/A')}")

    # Print 5 example predictions
    print("\n5 Example Predictions:")
    for i in range(min(5, len(test_preds))):
        pred_tags = [tag_vocab[j] for j in range(len(tag_vocab)) if test_preds[i][j] == 1]
        true_tags = [tag_vocab[j] for j in range(len(tag_vocab)) if test_labels[i][j] == 1]
        print(f"  Sample {i+1}: Predicted={pred_tags}, True={true_tags}")

    return metrics


# ========================
# Task 2: GNN Classifier
# ========================
def train_task2(config, device):
    """Train GNN on music segment graphs."""
    print("\n" + "=" * 60)
    print("TASK 2: GNN on Music Structure Graphs")
    print("=" * 60)

    tag_data, tag_vocab, splits, graph_dir = prepare_data(config)
    num_tags = len(tag_vocab)

    # Create labels dict
    labels_dict = {}
    for sid in splits['train'] + splits['val'] + splits['test']:
        info = tag_data[str(sid)]
        tag_vector = np.zeros(num_tags, dtype=np.float32)
        for tag in info.get('mood_tags', []) + info.get('genre_tags', []):
            if tag in tag_vocab:
                tag_vector[tag_vocab.index(tag)] = 1.0
        labels_dict[sid] = tag_vector

    train_ds = MusicGraphDataset(graph_dir, splits['train'], labels_dict)
    val_ds = MusicGraphDataset(graph_dir, splits['val'], labels_dict)
    test_ds = MusicGraphDataset(graph_dir, splits['test'], labels_dict)

    bs = config['training']['batch_size']
    train_loader = GraphDataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = GraphDataLoader(val_ds, batch_size=bs)
    test_loader = GraphDataLoader(test_ds, batch_size=bs)

    # GNN Model
    encoder = GraphSAGEEncoder(
        in_channels=config['graph']['node_feature_dim'],
        hidden_channels=config['gnn']['hidden_dims'],
        dropout=config['gnn']['dropout']
    )
    model = GNNClassifier(encoder, graph_dim=config['gnn']['hidden_dims'][-1],
                          num_classes=num_tags).to(device)

    optimizer = Adam(model.parameters(), lr=config['training']['lr'],
                     weight_decay=config['training'].get('weight_decay', 1e-4))
    criterion = nn.BCEWithLogitsLoss()
    epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0
    train_f1s, val_f1s = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(logits.shape))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.view(logits.shape).cpu().numpy())

        scheduler.step()
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        train_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)
        train_f1s.append(train_f1)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch.y.view(logits.shape).cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), average='macro', zero_division=0)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'results/gnn_classifier.pt')
            print("  -> Saved best model!")

    # Test evaluation
    model.load_state_dict(torch.load('results/gnn_classifier.pt', map_location=device))
    model.eval()
    test_preds, test_labels_arr, test_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)
            test_preds.append((probs > 0.5).astype(int))
            test_labels_arr.append(batch.y.view(logits.shape).cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_labels_arr = np.vstack(test_labels_arr)
    test_probs = np.vstack(test_probs)

    evaluator = Evaluator(config, 'results')
    metrics = evaluator.compute_classification_metrics(test_labels_arr, test_preds, test_probs)

    with open('results/metrics_task2.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    evaluator.plot_f1_curves(train_f1s, val_f1s, 'plots/task2_f1_curves.png')
    evaluator.plot_auc_pr_curve(test_labels_arr, test_probs, tag_vocab, 'plots/task2_pr_curves.png')

    print(f"\nTask 2 Results: Macro-F1={metrics['macro_f1']:.4f}, AUC-PR={metrics.get('auc_pr_macro', 'N/A')}")
    return metrics


# ========================
# Task 3: GNN-BERT Fusion
# ========================
def train_task3(config, device):
    """Train GNN-BERT fusion model with multi-task loss."""
    print("\n" + "=" * 60)
    print("TASK 3: GNN-BERT Fusion for Multi-Context Understanding")
    print("=" * 60)

    from transformers import BertTokenizer

    tag_data, tag_vocab, splits, graph_dir = prepare_data(config)
    num_tags = len(tag_vocab)
    tokenizer = BertTokenizer.from_pretrained(config['bert']['model_name'])

    # Create datasets
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        texts, tags, valences, arousals = get_song_data(tag_data, tag_vocab, splits[split_name])
        datasets[split_name] = MusicFusionDataset(
            graph_dir, texts, tags, valences, arousals,
            tokenizer, splits[split_name], config['bert']['max_length']
        )

    bs = config['training']['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=bs, shuffle=True, collate_fn=fusion_collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=bs, collate_fn=fusion_collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=bs, collate_fn=fusion_collate_fn)

    alpha = config['fusion'].get('alpha', 0.3)
    beta = config['fusion'].get('beta', 0.3)

    # Run ablation: cross_attention and early_concat
    ablation_results = {}

    for fusion_type in ['cross_attention', 'early_concat']:
        print(f"\n--- Training fusion type: {fusion_type} ---")

        gnn_encoder = GraphSAGEEncoder(
            in_channels=config['graph']['node_feature_dim'],
            hidden_channels=config['gnn']['hidden_dims'],
            dropout=config['gnn']['dropout']
        )
        bert_encoder = BERTTagClassifier(
            model_name=config['bert']['model_name'],
            num_tags=num_tags,
            freeze_layers=config['bert'].get('freeze_layers', 8)
        )
        model = GNNBERTFusionModel(
            gnn_encoder, bert_encoder, fusion_type=fusion_type,
            graph_dim=config['gnn']['hidden_dims'][-1],
            num_tags=num_tags,
            fused_dim=config['fusion']['fused_dim'],
            num_heads=config['fusion']['cross_attention_heads']
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=config['training']['lr'],
                          weight_decay=config['training'].get('weight_decay', 1e-4))
        tag_criterion = nn.BCEWithLogitsLoss()
        reg_criterion = nn.MSELoss()
        epochs = config['training']['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_f1 = 0
        model_save_name = f'fusion_model_{fusion_type}.pt'

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                graph = batch['graph'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tags = batch['tag_vector'].to(device)
                valence = batch['valence'].to(device)
                arousal = batch['arousal'].to(device)

                optimizer.zero_grad()
                tag_logits, val_pred, aro_pred, _, _ = model(graph, input_ids, attention_mask)

                loss_tags = tag_criterion(tag_logits, tags)
                loss_val = reg_criterion(val_pred.squeeze(), valence.squeeze())
                loss_aro = reg_criterion(aro_pred.squeeze(), arousal.squeeze())
                loss = loss_tags + alpha * loss_val + beta * loss_aro

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            val_vals, val_aros, pred_vals, pred_aros = [], [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    graph = batch['graph'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    tag_logits, val_pred, aro_pred, _, _ = model(graph, input_ids, attention_mask)
                    val_preds.append(torch.sigmoid(tag_logits).cpu().numpy())
                    val_labels.append(batch['tag_vector'].numpy())
                    val_vals.append(batch['valence'].numpy())
                    val_aros.append(batch['arousal'].numpy())
                    pred_vals.append(val_pred.cpu().numpy())
                    pred_aros.append(aro_pred.cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_labels = np.vstack(val_labels)
            val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), average='macro', zero_division=0)

            val_vals_arr = np.concatenate(val_vals).flatten()
            pred_vals_arr = np.concatenate(pred_vals).flatten()
            val_mae = np.mean(np.abs(val_vals_arr - pred_vals_arr))

            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val F1: {val_f1:.4f} | Val MAE(V): {val_mae:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'results/{model_save_name}')

        # Test evaluation
        model.load_state_dict(torch.load(f'results/{model_save_name}', map_location=device))
        model.eval()
        test_preds, test_labels_arr, test_probs = [], [], []
        test_embeddings = []

        with torch.no_grad():
            for batch in test_loader:
                graph = batch['graph'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                tag_logits, _, _, fused_emb, _ = model(graph, input_ids, attention_mask)
                probs = torch.sigmoid(tag_logits).cpu().numpy()
                test_probs.append(probs)
                test_preds.append((probs > 0.5).astype(int))
                test_labels_arr.append(batch['tag_vector'].numpy())
                test_embeddings.append(fused_emb.cpu().numpy())

        test_preds = np.vstack(test_preds)
        test_labels_arr = np.vstack(test_labels_arr)
        test_probs = np.vstack(test_probs)
        test_embeddings = np.vstack(test_embeddings)

        evaluator = Evaluator(config, 'results')
        metrics = evaluator.compute_classification_metrics(test_labels_arr, test_preds, test_probs)
        ablation_results[fusion_type] = metrics

        print(f"\n{fusion_type} Results: Macro-F1={metrics['macro_f1']:.4f}")

        # t-SNE visualization for cross_attention model
        if fusion_type == 'cross_attention':
            # Get dominant tag for coloring
            dominant_tags = np.argmax(test_labels_arr, axis=1)
            evaluator.plot_tsne(test_embeddings, dominant_tags, tag_vocab,
                                'plots/task3_tsne.png', 't-SNE of Fused Embeddings')

    # Save all ablation results
    with open('results/metrics_task3.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Also save the cross-attention model as the main fusion model
    if os.path.exists('results/fusion_model_cross_attention.pt'):
        import shutil
        shutil.copy('results/fusion_model_cross_attention.pt', 'results/fusion_model.pt')

    print("\nAblation Results:")
    for ft, m in ablation_results.items():
        print(f"  {ft}: Macro-F1={m['macro_f1']:.4f}")

    return ablation_results


# ========================
# Task 4: Contrastive Dual-Encoder
# ========================
def train_task4(config, device):
    """Train contrastive dual-encoder for audio-text retrieval."""
    print("\n" + "=" * 60)
    print("TASK 4: Cross-Modal Contrastive Learning")
    print("=" * 60)

    from transformers import BertTokenizer

    tag_data, tag_vocab, splits, graph_dir = prepare_data(config)
    num_tags = len(tag_vocab)
    tokenizer = BertTokenizer.from_pretrained(config['bert']['model_name'])

    # Create datasets
    datasets = {}
    texts_by_split = {}
    for split_name in ['train', 'val', 'test']:
        texts, _, _, _ = get_song_data(tag_data, tag_vocab, splits[split_name])
        texts_by_split[split_name] = texts
        datasets[split_name] = MusicContrastiveDataset(
            graph_dir, texts, tokenizer, splits[split_name], config['bert']['max_length']
        )

    bs = min(config['training']['batch_size'], 16)  # Smaller batch for contrastive
    train_loader = DataLoader(datasets['train'], batch_size=bs, shuffle=True,
                              collate_fn=fusion_collate_fn, drop_last=True)
    val_loader = DataLoader(datasets['val'], batch_size=bs, collate_fn=fusion_collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=bs, collate_fn=fusion_collate_fn)

    # Model
    gnn_encoder = GraphSAGEEncoder(
        in_channels=config['graph']['node_feature_dim'],
        hidden_channels=config['gnn']['hidden_dims'],
        dropout=config['gnn']['dropout']
    )
    bert_encoder = BERTTagClassifier(
        model_name=config['bert']['model_name'],
        num_tags=num_tags,
        freeze_layers=config['bert'].get('freeze_layers', 8)
    )

    model = ContrastiveDualEncoder(
        gnn_encoder, bert_encoder,
        graph_dim=config['gnn']['hidden_dims'][-1],
        text_dim=768,
        projection_dim=config['contrastive'].get('embedding_dim', 256),
        temperature=config['contrastive'].get('temperature', 0.07)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['training']['lr'],
                      weight_decay=config['training'].get('weight_decay', 1e-4))
    epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_r5 = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            graph = batch['graph'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            graph_emb, text_emb = model(graph, input_ids, attention_mask)
            total_loss, g2t_loss, t2g_loss = model.compute_loss(graph_emb, text_emb)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()

        # Validation retrieval metrics
        model.eval()
        all_graph_embs, all_text_embs = [], []

        with torch.no_grad():
            for batch in val_loader:
                graph = batch['graph'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                g_emb, t_emb = model(graph, input_ids, attention_mask)
                all_graph_embs.append(g_emb.cpu())
                all_text_embs.append(t_emb.cpu())

        all_graph_embs = torch.cat(all_graph_embs)
        all_text_embs = torch.cat(all_text_embs)
        val_metrics = compute_retrieval_metrics(all_graph_embs, all_text_embs)

        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Val R@1={val_metrics['g2t_R@1']:.4f} | R@5={val_metrics['g2t_R@5']:.4f} | "
              f"R@10={val_metrics['g2t_R@10']:.4f}")

        if val_metrics['g2t_R@5'] > best_val_r5:
            best_val_r5 = val_metrics['g2t_R@5']
            torch.save(model.state_dict(), 'results/contrastive_model.pt')

    # Test evaluation
    model.load_state_dict(torch.load('results/contrastive_model.pt', map_location=device))
    model.eval()
    all_graph_embs, all_text_embs = [], []

    with torch.no_grad():
        for batch in test_loader:
            graph = batch['graph'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            g_emb, t_emb = model(graph, input_ids, attention_mask)
            all_graph_embs.append(g_emb.cpu())
            all_text_embs.append(t_emb.cpu())

    all_graph_embs = torch.cat(all_graph_embs)
    all_text_embs = torch.cat(all_text_embs)
    test_metrics = compute_retrieval_metrics(all_graph_embs, all_text_embs)

    # Generate retrieval examples
    evaluator = Evaluator(config, 'results')
    evaluator.generate_retrieval_examples(
        all_graph_embs.numpy(), all_text_embs.numpy(),
        texts_by_split['test'], splits['test'],
        'retrieval_examples'
    )

    with open('results/metrics_task4.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nTask 4 Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return test_metrics


# ========================
# Baseline: Random Predictor
# ========================
def train_baseline_random(config):
    """Compute random baseline metrics."""
    tag_data, tag_vocab, splits, _ = prepare_data(config)
    num_tags = len(tag_vocab)

    _, test_tags, _, _ = get_song_data(tag_data, tag_vocab, splits['test'])
    random_preds = np.random.randint(0, 2, size=test_tags.shape)
    random_probs = np.random.rand(*test_tags.shape)

    evaluator = Evaluator(config, 'results')
    metrics = evaluator.compute_classification_metrics(test_tags, random_preds, random_probs)
    print(f"Random Baseline: Macro-F1={metrics['macro_f1']:.4f}")
    return metrics


# ========================
# Main Entry Point
# ========================
def main():
    parser = argparse.ArgumentParser(description='GNN-BERT Music Context Training')
    parser.add_argument('--task', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Task number: 1=BERT, 2=GNN, 3=Fusion, 4=Contrastive')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    os.makedirs('results/plots', exist_ok=True)

    task_funcs = {
        1: train_task1,
        2: train_task2,
        3: train_task3,
        4: train_task4,
    }

    metrics = task_funcs[args.task](config, device)

    # Also run random baseline for comparison
    print("\n--- Random Baseline ---")
    random_metrics = train_baseline_random(config)

    # Compile all results
    all_results = {'random_baseline': random_metrics}
    for task_num in [1, 2, 3, 4]:
        path = f'results/metrics_task{task_num}.json'
        if os.path.exists(path):
            with open(path) as f:
                all_results[f'task{task_num}'] = json.load(f)

    with open('results/metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\nAll results saved to results/metrics.json")


if __name__ == '__main__':
    main()
