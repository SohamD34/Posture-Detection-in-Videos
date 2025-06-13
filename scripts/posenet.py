import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pycocotools.coco import COCO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from models import PoseNet
from utils import CocoKeypoints, VideoCreator, convert_to_modern_mp4, detect_keypoints, log_text
from IPython.display import FileLink
import warnings
warnings.filterwarnings("ignore")
os.chdir('..')


def train(model, train_loader, epochs=10, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, heatmaps) in enumerate(train_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                log_text('logs/training_log.txt',f'Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/500:.4f}')
                running_loss = 0.0

        # Save the model after each epoch
        model_path = f'model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        log_text('logs/training_log.txt', f'Model saved: {model_path}')

        log_text('logs/training_log.txt', f'Epoch {epoch+1} completed')

    return model


coco_root = '/kaggle/input/coco-2017-dataset/coco2017'
train_img_dir = os.path.join(coco_root, 'train2017')
train_ann_file = os.path.join(coco_root, 'annotations/person_keypoints_train2017.json')


log_text('logs/training_log.txt', 'Custom dataset creation started')
dataset = CocoKeypoints(train_img_dir, train_ann_file)

log_text('logs/training_log.txt', 'Initializing DataLoader')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

log_text('logs/training_log.txt', 'Initializing PoseNet model')
model = PoseNet()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_text('logs/training_log.txt', f'Training started on device: {device}')
model = train(model, train_loader, epochs=10, device=device)
log_text('logs/training_log.txt', 'Training completed')