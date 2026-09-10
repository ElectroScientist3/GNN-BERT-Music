"""Generate report artifacts from processed data without requiring model checkpoints."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED / "graphs"
PLOT_DIR = ROOT / "results" / "plots"
EXAMPLE_DIR = ROOT / "results" / "retrieval_examples"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_bar(values, labels, title, ylabel, output_path, color):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, color=color)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    tag_data = load_json(PROCESSED / "tags_and_descriptions.json")
    splits = load_json(ROOT / "data" / "splits" / "splits.json")

    split_labels = ["Train", "Validation", "Test"]
    split_counts = [len(splits[key]) for key in ("train", "val", "test")]
    save_bar(
        split_counts,
        split_labels,
        "Dataset Split Sizes",
        "Number of tracks",
        PLOT_DIR / "dataset_split_sizes.png",
        "#2f6690",
    )

    tag_names = [
        "happy", "energetic", "relaxed", "peaceful", "calm", "angry",
        "tense", "sad", "electronic", "rock", "classical", "ambient",
        "pop", "jazz", "metal", "folk",
    ]
    tag_counts = {tag: 0 for tag in tag_names}
    valences = []
    arousals = []
    for item in tag_data.values():
        for tag in item.get("mood_tags", []) + item.get("genre_tags", []):
            if tag in tag_counts:
                tag_counts[tag] += 1
        valences.append(float(item["valence"]))
        arousals.append(float(item["arousal"]))

    save_bar(
        [tag_counts[tag] for tag in tag_names],
        tag_names,
        "Generated Tag Frequency",
        "Number of tracks",
        PLOT_DIR / "generated_tag_frequency.png",
        "#3a7d44",
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(valences, arousals, s=8, alpha=0.35, color="#c44536")
    axis.axvline(5, color="#555555", linewidth=0.8)
    axis.axhline(5, color="#555555", linewidth=0.8)
    axis.set_xlabel("Valence")
    axis.set_ylabel("Arousal")
    axis.set_title("DEAM-Derived Valence and Arousal")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "valence_arousal_distribution.png", dpi=200)
    plt.close(figure)

    graph_ids = sorted(GRAPH_DIR.glob("*.pt"), key=lambda path: int(path.stem))
    node_counts = []
    edge_counts = []
    for graph_path in graph_ids:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        node_counts.append(int(graph.num_nodes))
        edge_counts.append(int(graph.edge_index.shape[1]))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(node_counts, bins=25, color="#7b2cbf", alpha=0.85)
    axes[0].set_title("Graph Node Counts")
    axes[0].set_xlabel("Nodes per graph")
    axes[0].set_ylabel("Number of graphs")
    axes[1].hist(edge_counts, bins=25, color="#f77f00", alpha=0.85)
    axes[1].set_title("Graph Edge Counts")
    axes[1].set_xlabel("Directed edges per graph")
    axes[1].set_ylabel("Number of graphs")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "graph_size_distribution.png", dpi=200)
    plt.close(figure)

    example_path = EXAMPLE_DIR / "generated_description_examples.txt"
    with example_path.open("w", encoding="utf-8") as handle:
        handle.write("Generated text examples from tags_and_descriptions.json\n")
        handle.write("These are input-description examples, not retrieval rankings.\n\n")
        for song_id in sorted(tag_data, key=lambda value: int(value))[:10]:
            item = tag_data[song_id]
            handle.write(f"Song {song_id}\n")
            handle.write(f"Mood tags: {', '.join(item.get('mood_tags', []))}\n")
            handle.write(f"Genre tags: {', '.join(item.get('genre_tags', []))}\n")
            handle.write(f"Description: {item.get('text_description', '')}\n\n")

    (EXAMPLE_DIR / "README.txt").write_text(
        "Retrieval rankings are not generated because no trained audio/text embeddings "
        "or metrics_task4.json file exists yet. The generated_description_examples.txt "
        "file contains real text inputs from the processed cache for report illustration.\n",
        encoding="utf-8",
    )

    summary = {
        "tracks_with_generated_tags": len(tag_data),
        "graphs": len(graph_ids),
        "average_nodes": float(np.mean(node_counts)),
        "average_directed_edges": float(np.mean(edge_counts)),
    }
    (ROOT / "results" / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
