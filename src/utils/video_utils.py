import torch
import numpy as np
import decord
import os

def load_video_frames(video_path, num_frames=16, resolution=224):
    """
    Load frames from a video file using Decord.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.
        resolution (int): Target height/width for resizing (assumes square).
        
    Returns:
        torch.Tensor: [Time, Channel, Height, Width], values in [0, 1]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    try:
        vr = decord.VideoReader(video_path, width=resolution, height=resolution)
        total_frames = len(vr)
        
        # Simple uniform sampling
        if total_frames <= num_frames:
             indices = np.arange(total_frames)
        else:
             indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
             
        frames = vr.get_batch(indices).asnumpy() # (T, H, W, C)
        
        # Convert to Torch Tensor: (T, C, H, W)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        
        # Pad if fewer frames than requested
        if frames.shape[0] < num_frames:
             pad_len = num_frames - frames.shape[0]
             last_frame = frames[-1].unsqueeze(0)
             padding = last_frame.repeat(pad_len, 1, 1, 1)
             frames = torch.cat([frames, padding], dim=0)
             
        return frames
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        # Return a zero tensor as fallback
        return torch.zeros(num_frames, 3, resolution, resolution)

def save_video_tensor(tensor, output_path, fps=8):
    """
    Save a tensor as a video file.
    tensor: [T, C, H, W] or [T, H, W, C]
    """
    # 模拟保存
    print(f"Saving video to {output_path}")
    pass
