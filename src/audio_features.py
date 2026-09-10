import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from typing import List

class AudioFeatureExtractor:
    def __init__(self, config: dict = None):
        defaults = {
            'sr': 22050,
            'n_mels': 128,
            'hop_length': 512,
            'n_fft': 2048,
            'segment_duration': 2.0, # in seconds
            'opensmile_frame_duration': 0.5 # OpenSMILE features frame interval
        }
        if config is not None:
            # Handle nested config from YAML (where audio settings are in config['audio'])
            audio_cfg = config.get('audio', {}) if isinstance(config, dict) else {}
            for k, v in defaults.items():
                if k in audio_cfg:
                    defaults[k] = audio_cfg[k]
                elif isinstance(config, dict) and k in config:
                    defaults[k] = config[k]
        self.config = defaults

    def extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(audio_path, sr=self.config['sr'])
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, 
                n_mels=self.config['n_mels'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Error extracting mel spectrogram from {audio_path}: {e}")
            return np.array([])

    def extract_chroma(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(audio_path, sr=self.config['sr'])
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr,
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            return chroma
        except Exception as e:
            print(f"Error extracting chromagram from {audio_path}: {e}")
            return np.array([])

    def segment_audio(self, features: np.ndarray, segment_duration: float) -> List[np.ndarray]:
        if features.size == 0:
            return []
        
        # Calculate how many frames correspond to segment_duration
        frames_per_sec = self.config['sr'] / self.config['hop_length']
        frames_per_segment = int(segment_duration * frames_per_sec)
        
        num_segments = features.shape[1] // frames_per_segment
        segments = []
        
        for i in range(num_segments):
            start = i * frames_per_segment
            end = start + frames_per_segment
            segment = features[:, start:end]
            segments.append(segment)
            
        return segments

    def load_opensmile_features(self, feature_csv_path: str) -> np.ndarray:
        try:
            # Usually semicolon separated, first column is frameTime
            df = pd.read_csv(feature_csv_path, sep=';')
            # Drop frameTime or name columns if present to only keep numerical features
            if 'frameTime' in df.columns:
                features = df.drop(columns=['frameTime', 'name'], errors='ignore').values
            else:
                features = df.values
            return features
        except Exception as e:
            print(f"Error loading OpenSMILE features from {feature_csv_path}: {e}")
            return np.array([])

    def segment_opensmile_features(self, features: np.ndarray) -> List[np.ndarray]:
        if features.size == 0:
            return []
        
        # Determine frames per segment based on OpenSMILE frame duration
        frames_per_segment = int(self.config['segment_duration'] / self.config['opensmile_frame_duration'])
        
        if frames_per_segment <= 0:
            frames_per_segment = 1
            
        num_segments = len(features) // frames_per_segment
        segments = []
        
        for i in range(num_segments):
            start = i * frames_per_segment
            end = start + frames_per_segment
            segment = features[start:end]
            # Compute mean within segment
            segment_mean = np.mean(segment, axis=0)
            segments.append(segment_mean)
            
        return segments

    def process_all_tracks(self, audio_dir: str, features_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        audio_files = glob.glob(os.path.join(audio_dir, '*.mp3')) + glob.glob(os.path.join(audio_dir, '*.wav'))
        
        for audio_path in tqdm(audio_files, desc="Processing tracks"):
            base_name = os.path.basename(audio_path)
            song_id = os.path.splitext(base_name)[0]
            
            mel_spec = self.extract_mel_spectrogram(audio_path)
            mel_segments = self.segment_audio(mel_spec, self.config['segment_duration'])
            
            # Save mel segments
            mel_out_path = os.path.join(output_dir, f"{song_id}_mel.npy")
            np.save(mel_out_path, np.array(mel_segments))
            
            # Process OpenSMILE features if available
            feature_csv_path = os.path.join(features_dir, f"{song_id}.csv")
            if os.path.exists(feature_csv_path):
                opensmile_features = self.load_opensmile_features(feature_csv_path)
                opensmile_segments = self.segment_opensmile_features(opensmile_features)
                
                os_out_path = os.path.join(output_dir, f"{song_id}_opensmile.npy")
                np.save(os_out_path, np.array(opensmile_segments))

if __name__ == '__main__':
    extractor = AudioFeatureExtractor()
    extractor.process_all_tracks(
        audio_dir='data/raw/audio',
        features_dir='data/raw/features',
        output_dir='data/processed/audio_features'
    )
