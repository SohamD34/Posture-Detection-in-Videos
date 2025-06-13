import os
import cv2
import numpy as np
import warnings
import subprocess
import random
import matplotlib.pyplot as plt
from natsort import natsorted
from utils import VideoCreator, convert_to_modern_mp4, log_text
from models import LucasKanadePositionDetector
warnings.filterwarnings("ignore")
os.chdir('..')


''' EXAMPLE 1 '''

video_name = "demo1"
video_path = f"data/{video_name}.mp4"
frame_dir = f"frames/lucas_kanade/{video_name}/"

log_text('logs/log.txt', f"Lucas-Kanade Flow Detection started for {video_name}")
lk_detector = LucasKanadePositionDetector(video_path, frame_dir)
lk_detector.detect()

log_text('logs/log.txt', f"Starting video creation...")
frame_dir = f"frames/lucas_kanade/{video_name}/"
vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output_lucas_kanade.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_lucas_kanade.mp4")

log_text('logs/log.txt', f"Output video saved at: {output_mp4_file}")




''' EXAMPLE 2 '''

video_name = "demo2"
video_path = f"data/{video_name}.mp4"
frame_dir = f"frames/lucas_kanade/{video_name}/"

log_text('logs/log.txt', f"Lucas-Kanade Flow Detection started for {video_name}")
lk_detector = LucasKanadePositionDetector(video_path, frame_dir)
lk_detector.detect()

log_text('logs/log.txt', f"Starting video creation...")
frame_dir = f"frames/lucas_kanade/{video_name}/"
vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output_lucas_kanade.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_lucas_kanade.mp4")

log_text('logs/log.txt', f"Output video saved at: {output_mp4_file}")