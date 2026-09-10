import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class TagGenerator:
    """
    Generate tags and text descriptions from DEAM valence/arousal annotations.
    """
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                'annotations_path_1': 'data/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv',
                'annotations_path_2': 'data/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv'
            }
        self.config = config
        self.vocabulary = self.get_tag_vocabulary()

    def get_tag_vocabulary(self) -> List[str]:
        return [
            'happy', 'energetic', 'relaxed', 'peaceful', 'calm',
            'angry', 'tense', 'sad', 'electronic', 'rock',
            'classical', 'ambient', 'pop', 'jazz', 'metal', 'folk'
        ]

    def load_annotations(self) -> pd.DataFrame:
        import glob
        path1 = self.config.get('annotations_path_1', '')
        path2 = self.config.get('annotations_path_2', '')
        
        possible_paths = [path1, path2]
        # Search recursively for static annotation CSVs in data directory
        search_dirs = ['data/raw', 'data/raw/annotations', 'data/annotations', '.']
        for sdir in search_dirs:
            if os.path.exists(sdir):
                found = glob.glob(os.path.join(sdir, '**', 'static_annotations_averaged_songs_*.csv'), recursive=True)
                possible_paths.extend(found)

        # Filter out duplicates and non-existing paths
        unique_paths = list(set([p for p in possible_paths if p and os.path.exists(p)]))
        
        df_list = []
        for path in unique_paths:
            try:
                # DEAM CSVs can be comma or semicolon separated
                df = pd.read_csv(path, sep=',')
                if 'valence_mean' not in df.columns and 'valence_mean' not in [c.strip() for c in df.columns]:
                    df = pd.read_csv(path, sep=';')
                # Clean column names
                df.columns = [c.strip() for c in df.columns]
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
                
            cols = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
            if all(c in df.columns for c in cols):
                df_list.append(df[cols])
            else:
                print(f"Warning: Expected columns missing in {path}. Columns: {df.columns.tolist()}")
                
        if not df_list:
            return pd.DataFrame(columns=['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std'])
            
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df

    def valence_arousal_to_mood_tags(self, valence: float, arousal: float) -> List[str]:
        # scale is 1-9, center is 5
        tags = []
        if valence >= 5.0 and arousal >= 5.0:
            tags.extend(['happy', 'energetic'])
        elif valence >= 5.0 and arousal < 5.0:
            tags.extend(['relaxed', 'peaceful', 'calm'])
        elif valence < 5.0 and arousal >= 5.0:
            tags.extend(['angry', 'tense'])
        elif valence < 5.0 and arousal < 5.0:
            tags.extend(['sad', 'calm'])
        return tags

    def valence_arousal_to_genre_tags(self, valence: float, arousal: float) -> List[str]:
        tags = []
        if arousal >= 5.0:
            tags.extend(['electronic', 'rock'])
        else:
            tags.extend(['classical', 'ambient'])
            
        if valence >= 5.0:
            tags.extend(['pop', 'jazz'])
        else:
            tags.extend(['metal', 'folk'])
            
        return tags

    def generate_text_description(self, mood_tags: List[str], genre_tags: List[str], valence: float, arousal: float) -> str:
        energy_desc = "high energy" if arousal >= 5.0 else "low energy"
        mood_desc = "positive mood" if valence >= 5.0 else "negative mood"
        mood_str = " and ".join(mood_tags[:2]) if mood_tags else "neutral"
        genre_str = genre_tags[0] if genre_tags else "music"
        
        description = f"A {mood_str} {genre_str} track with {energy_desc} and {mood_desc}."
        return description

    def generate_all(self) -> Dict:
        df = self.load_annotations()
        results = {}
        
        for _, row in df.iterrows():
            song_id = int(row['song_id'])
            v = row['valence_mean']
            a = row['arousal_mean']
            
            mood_tags = self.valence_arousal_to_mood_tags(v, a)
            genre_tags = self.valence_arousal_to_genre_tags(v, a)
            all_tags_list = mood_tags + genre_tags
            binary_vector = create_binary_tag_vector(all_tags_list, self.vocabulary)
            
            description = self.generate_text_description(mood_tags, genre_tags, v, a)
            
            results[str(song_id)] = {
                'mood_tags': mood_tags,
                'genre_tags': genre_tags,
                'all_tags': binary_vector.tolist(),
                'text_description': description,
                'valence': v,
                'arousal': a
            }
            
        return results

    def save(self, output_path: str):
        results = self.generate_all()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved generated tags to {output_path}")

def create_binary_tag_vector(tags: List[str], vocabulary: List[str]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=int)
    for tag in tags:
        if tag in vocabulary:
            idx = vocabulary.index(tag)
            vector[idx] = 1
    return vector

if __name__ == '__main__':
    generator = TagGenerator()
    generator.save('data/processed/tags_and_descriptions.json')
