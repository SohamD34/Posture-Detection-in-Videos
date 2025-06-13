import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pycocotools.coco import COCO
import warnings
from cvzone.PoseModule import PoseDetector
from natsort import natsorted
import matplotlib.pyplot as plt
import subprocess
import random
import logging
warnings.filterwarnings("ignore")


class CocoKeypoints(Dataset):
    def __init__(self, root, annFile, target_size=(256, 192), flip_prob=0.5):
        self.root = root
        self.coco = COCO(annFile)
        self.target_size = target_size
        self.flip_prob = flip_prob
        self.image_ids = [img_id for img_id in self.coco.imgs.keys() 
                        if len(self.coco.getAnnIds(imgIds=img_id, catIds=1)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=1)
        anns = self.coco.loadAnns(ann_ids)
        ann = max(anns, key=lambda x: x['num_keypoints'])

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        kp = keypoints[:, :2]
        visibility = keypoints[:, 2]

        kp[:, 0] = kp[:, 0] * (self.target_size[1] / orig_w)
        kp[:, 1] = kp[:, 1] * (self.target_size[0] / orig_h)

        if np.random.rand() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            kp[:, 0] = self.target_size[1] - kp[:, 0]
            left = [1, 3, 5, 7, 9, 11, 13, 15]
            right = [2, 4, 6, 8, 10, 12, 14, 16]
            kp[left + right] = kp[right + left]

        img = np.ascontiguousarray(img)
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])

        heatmap_h, heatmap_w = self.target_size[0]//4, self.target_size[1]//4
        heatmaps = np.zeros((17, heatmap_h, heatmap_w), dtype=np.float32)
        
        for i in range(17):
            if visibility[i] > 0:
                x = (kp[i, 0] / self.target_size[1]) * heatmap_w
                y = (kp[i, 1] / self.target_size[0]) * heatmap_h
                heatmaps[i] = self._gaussian_kernel(heatmap_h, heatmap_w, x, y, 2)
        
        return img, torch.tensor(heatmaps, dtype=torch.float32)

    def _gaussian_kernel(self, height, width, x, y, sigma):
        xv, yv = np.meshgrid(np.arange(width), np.arange(height))
        d2 = (xv - x)**2 + (yv - y)**2
        return np.exp(-d2 / (2 * sigma**2))
    


def detect_keypoints(model, image_path, device='cuda'):
    # Load and preprocess image
    img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]
    
    img = cv2.resize(orig_img, (192, 256))
    img_tensor = transforms.functional.to_tensor(img)
    img_tensor = transforms.functional.normalize(img_tensor,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).unsqueeze(0)
 
    model.eval()
    with torch.no_grad():
        heatmaps = model(img_tensor.to(device)).cpu().numpy()[0]

    keypoints = []
    for i in range(17):
        hm = heatmaps[i]
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        x = (x / 48 * 192) * (orig_w / 192)
        y = (y / 64 * 256) * (orig_h / 256)
        keypoints.append((int(x), int(y)))
 
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img)
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, s=50, marker='.', c='red')
    plt.axis('off')
    plt.show()

    


class VideoCreator:
    '''
    Class to create a video from a series of image frames.

    Args:
        frame_dir (str): Directory containing the image frames.
        output_path (str): Path to save the output video. Default is "output/video_output.mp4".
        fps (int): Frames per second for the output video. Default is 30.

    Methods:
        _load_frames(): Loads image frames from the specified directory.
        _get_frame_size(): Gets the size of the first frame to set video dimensions.
        create_video(): Creates a video from the loaded frames and saves it to the specified output path.

    Returns:
        A video file created from the image frames is saved to the specified output path.
    '''

    def __init__(self, frame_dir, output_path="output/output_video.mp4", fps=30):
        self.frame_dir = frame_dir
        self.output_path = output_path
        self.fps = fps
        self.frames = self._load_frames()

        if not self.frames:
            raise ValueError(f"No frames found in directory: {frame_dir}")

        self.frame_size = self._get_frame_size(self.frames[0])
        self.fourcc = self._choose_fourcc()


    def _load_frames(self):
        valid_exts = ('.jpg', '.jpeg', '.png')
        return natsorted([f for f in os.listdir(self.frame_dir) if f.lower().endswith(valid_exts)])


    def _get_frame_size(self, sample_file):
        path = os.path.join(self.frame_dir, sample_file)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Can't read sample frame: {sample_file}")
        return (img.shape[1], img.shape[0])


    def _choose_fourcc(self):
        ext = os.path.splitext(self.output_path)[-1].lower()
        if ext == ".avi":
            return cv2.VideoWriter_fourcc(*"XVID")
        else:  # .mp4 or others
            return cv2.VideoWriter_fourcc(*"mp4v")


    def create_video(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, self.frame_size)

        for f in self.frames:
            img = cv2.imread(os.path.join(self.frame_dir, f))
            if img is None:
                print(f"Skipping unreadable frame: {f}")
                continue
            if (img.shape[1], img.shape[0]) != self.frame_size:
                img = cv2.resize(img, self.frame_size)
            writer.write(img)

        writer.release()
        return self.output_path




def convert_to_modern_mp4(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True)
    log_text('logs/log.txt',f"Video successfully saved to: {output_path}")
    os.remove(input_path)
    return input_path


def setup_logger(log_file_path):
    '''
    A helper function to setup the logger - allowing it to access files, read & write them.
    '''
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_text(log_file_path, text):
    '''
    A helper function to log the text to the file path specified.
    '''
    logger = setup_logger(log_file_path)
    logger.info(text)