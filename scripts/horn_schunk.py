import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
import subprocess
import random
from natsort import natsorted
from utils import VideoCreator, convert_to_modern_mp4, log_text
from models import HornSchunckPositionDetector
warnings.filterwarnings("ignore")
os.chdir('..')



''' EXAMPLE 1 '''

file_name = 'demo1'

detector = HornSchunckPositionDetector(
    video_path=f"data/{file_name}.mp4",
    output_dir=f"frames/horn_schunck/{file_name}",
    alpha=1.0,
    num_iter=100
)
log_text('logs/log.txt', f"Horn-Shunck Detection started for {file_name}")
frame_dir = detector.detect()
log_text('logs/log.txt', f"Frames saved at: {frame_dir}")

log_text('logs/log.txt', f"Starting video creation...")
vc = VideoCreator(frame_dir, output_path=f"output/{file_name}_output_horn_schunck.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{file_name}_output_horn_schunck.mp4")

log_text('logs/log.txt',f"Output video saved at: {output_mp4_file}")



''' EXAMPLE 2 '''

file_name = 'demo2'

detector = HornSchunckPositionDetector(
    video_path=f"data/{file_name}.mp4",
    output_dir=f"frames/horn_schunck/{file_name}",
    alpha=1.0,
    num_iter=100
)
log_text('logs/log.txt', f"Horn-Shunck Detection started for {file_name}")
frame_dir = detector.detect()
log_text('logs/log.txt', f"Frames saved at: {frame_dir}")

log_text('logs/log.txt', f"Starting video creation...")
vc = VideoCreator(frame_dir, output_path=f"output/{file_name}_output_horn_schunck.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{file_name}_output_horn_schunck.mp4")

log_text('logs/log.txt',f"Output video saved at: {output_mp4_file}")

