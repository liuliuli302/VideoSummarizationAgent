import torch
from torch.utils.data import Dataset
from src.utils.video_utils import load_video_frames

class VideoDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        # 模拟文件列表
        self.video_files = [f"video_{i}.mp4" for i in range(10)]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        videos = load_video_frames(video_name)
        
        # 模拟标签：假设是一个二分类任务 (是否包含特定动作)
        label = torch.randint(0, 2, (1,)).float()
        
        if self.transform:
            # 可以在这里应用 self.transform(videos)
            pass
            
        return {"video": videos, "label": label, "name": video_name}
