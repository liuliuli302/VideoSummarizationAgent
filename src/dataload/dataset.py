import torch
import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from src.utils.video_utils import load_video_frames

class VideoDataset(Dataset):
    def __init__(self, data_root, metadata_file=None, transform=None, num_frames=16):
        """
        Args:
            data_root (str): Root directory of videos.
            metadata_file (str, optional): Path to a csv/json file with video annotations.
            transform (callable, optional): Transform to apply to video tensor.
            num_frames (int): Number of frames to sample from each video.
        """
        self.data_root = data_root
        self.transform = transform
        self.num_frames = num_frames
        self.video_files = []
        self.labels = {} # map filename -> label/annotation

        # 1. Try loading from metadata
        if metadata_file and os.path.exists(metadata_file):
            print(f"Loading metadata from {metadata_file}")
            if metadata_file.endswith('.csv'):
                df = pd.read_csv(metadata_file)
                # Assumes 'video_path' or 'video_id' column exists
                for _, row in df.iterrows():
                    # Handle different potential column names
                    vid_name = row.get('video_path') or row.get('video_name') or row.get('id')
                    if vid_name:
                         self.video_files.append(vid_name)
                         # Load label if exists (e.g. 'label', 'summary')
                         if 'label' in row:
                             self.labels[vid_name] = row['label']
            
            # TODO: Add JSON support if needed
            
        # 2. If no metadata or empty, scan directory
        if not self.video_files:
            print(f"Scanning directory {data_root} for videos...")
            # Recursively find mp4 files
            self.video_files = sorted(glob.glob(os.path.join(data_root, "**/*.mp4"), recursive=True))
            # Store relative paths if possible, or just absolute
            self.video_files = [os.path.relpath(p, data_root) for p in self.video_files]
            
        print(f"Found {len(self.video_files)} videos in {data_root}")

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.data_root, video_name)
        
        # Load Video
        try:
            videos = load_video_frames(video_path, num_frames=self.num_frames)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return a dummy tensor in case of corruption
            videos = torch.zeros(self.num_frames, 3, 224, 224)

        # Get Label
        # If real annotation exists, load it.
        # Otherwise, generate mock frame-level labels for testing/pre-training.
        # [Time]
        label = self.labels.get(video_name)
        
        if label is None:
             # Random binary labels for self-supervised/test
             label = torch.randint(0, 2, (self.num_frames,)).float()
        elif not isinstance(label, torch.Tensor):
             label = torch.tensor(label).float()
             
             # Expand if scalar (video-level label) -> frame-level
             if label.dim() == 0:
                 label = label.expand(self.num_frames)
        
        if self.transform:
            videos = self.transform(videos)
            
        # Return dict matching model input
        return {"video": videos, "label": label, "name": video_name}
