import numpy as np
import torch

class VideoTools:
    def __init__(self):
        pass

    def extract_clip(self, video_tensor: torch.Tensor, start_frame: int, end_frame: int):
        """
        从视频张量中提取片段
        video_tensor: [Time, C, H, W]
        """
        t, c, h, w = video_tensor.shape
        if start_frame < 0: start_frame = 0
        if end_frame > t: end_frame = t
        
        return video_tensor[start_frame:end_frame]
    
    def detect_object(self, frame_tensor, object_name):
        """
        模拟在单帧中检测物体
        返回置信度
        """
        # 随机返回置信度，作为演示
        return np.random.rand()
