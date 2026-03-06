import cv2
import numpy as np
import os

def create_dummy_video(filename, frames=60, width=224, height=224):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    for i in range(frames):
        # Create a frame with moving shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # BGR format
        cv2.circle(frame, (i * 2 % width, i * 2 % height), 20, (255, 0, 0), -1)
        cv2.rectangle(frame, (width - i*2 % width, height - i*2 % height), (width - i*2 % width + 40, height - i*2 % height + 40), (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    os.makedirs("data/raw/egocentric", exist_ok=True)
    
    for i in range(3):
        create_dummy_video(f"data/raw/egocentric/test_video_{i}.mp4", frames=100)
