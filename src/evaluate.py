import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, average_precision_score, mean_absolute_error, r2_score, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE

class Evaluator:
    def __init__(self, config, results_dir='results'):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def compute_classification_metrics(self, y_true, y_pred, y_prob):
        metrics = {}
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        per_tag_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        metrics['per_tag_f1'] = per_tag_f1.tolist()
        
        try:
            auc_pr = average_precision_score(y_true, y_prob, average='macro')
            metrics['auc_pr_macro'] = auc_pr
        except Exception:
            metrics['auc_pr_macro'] = None
            
        return metrics
    
    def compute_regression_metrics(self, y_true, y_pred):
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        return metrics
    
    def compute_retrieval_metrics(self, graph_emb, text_emb, ks=[1,5,10]):
        import torch
        if not isinstance(graph_emb, torch.Tensor): graph_emb = torch.tensor(graph_emb)
        if not isinstance(text_emb, torch.Tensor): text_emb = torch.tensor(text_emb)
            
        sim_matrix = torch.matmul(graph_emb, text_emb.T)
        num_samples = sim_matrix.size(0)
        labels = torch.arange(num_samples).view(-1, 1)
        
        metrics = {}
        _, g2t_indices = torch.topk(sim_matrix, max(ks), dim=-1)
        for k in ks:
            correct = (g2t_indices[:, :k] == labels).sum().item()
            metrics[f'g2t_R@{k}'] = correct / num_samples
            
        _, t2g_indices = torch.topk(sim_matrix.T, max(ks), dim=-1)
        for k in ks:
            correct = (t2g_indices[:, :k] == labels).sum().item()
            metrics[f't2g_R@{k}'] = correct / num_samples
            
        return metrics
    
    def plot_f1_curves(self, train_f1s, val_f1s, save_path):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_f1s) + 1)
        plt.plot(epochs, train_f1s, 'b-', label='Train Macro-F1')
        plt.plot(epochs, val_f1s, 'r-', label='Val Macro-F1')
        plt.title('Training and Validation Macro-F1')
        plt.xlabel('Epochs')
        plt.ylabel('Macro-F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_auc_pr_curve(self, y_true, y_prob, tag_names, save_path):
        plt.figure(figsize=(12, 8))
        for i in range(y_true.shape[1]):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            plt.plot(recall, precision, label=f'{tag_names[i]}')
        plt.title('Precision-Recall Curves per Tag')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if len(tag_names) <= 15:
            plt.legend(loc='lower left', fontsize='small')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tsne(self, embeddings, labels, label_names, save_path, title='t-SNE'):
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.title(title)
        
        if label_names is not None:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, label_names[:len(handles)], title="Classes")
            
        plt.savefig(os.path.join(self.results_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.results_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_table(self, all_results, save_path):
        with open(os.path.join(self.results_dir, save_path), 'w') as f:
            json.dump(all_results, f, indent=4)
            
        print("\n" + "="*50)
        print("MODEL COMPARISON RESULTS")
        print("="*50)
        print(f"{'Model':<20} | {'Macro-F1':<10} | {'Micro-F1':<10}")
        print("-" * 50)
        for model_name, metrics in all_results.items():
            print(f"{model_name:<20} | {metrics.get('macro_f1', 0.0):.4f}     | {metrics.get('micro_f1', 0.0):.4f}")
        print("="*50 + "\n")
    
    def generate_retrieval_examples(self, graph_emb, text_emb, texts, song_ids, save_dir, top_k=3):
        import torch
        save_path = os.path.join(self.results_dir, save_dir)
        os.makedirs(save_path, exist_ok=True)
        
        if not isinstance(graph_emb, torch.Tensor): graph_emb = torch.tensor(graph_emb)
        if not isinstance(text_emb, torch.Tensor): text_emb = torch.tensor(text_emb)
        
        sim_matrix = torch.matmul(graph_emb, text_emb.T)
        _, top_indices = torch.topk(sim_matrix, top_k, dim=-1)
        
        num_queries = min(10, sim_matrix.size(0))
        for i in range(num_queries):
            query_id = song_ids[i]
            retrieved = [(song_ids[idx.item()], texts[idx.item()]) for idx in top_indices[i]]
            
            with open(os.path.join(save_path, f'query_{query_id}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Query Audio ID: {query_id}\n")
                f.write("-" * 40 + "\n")
                f.write("Top Retrieved Captions:\n")
                for rank, (r_id, text) in enumerate(retrieved, 1):
                    f.write(f"{rank}. [Song {r_id}] {text}\n")
    
    def run_full_evaluation(self, task=None):
        print(f"Running full evaluation for task {task} (mocked loading of best models).")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all')
    args = parser.parse_args()
    
    evaluator = Evaluator(config={})
    evaluator.run_full_evaluation(task=args.task)
