import os
import cv2
import warnings
import subprocess
import random
import matplotlib.pyplot as plt
import pickle
from cvzone.PoseModule import PoseDetector
from natsort import natsorted
from models import PoseProcessor
from utils import VideoCreator, convert_to_modern_mp4, log_text
warnings.filterwarnings("ignore")

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.chdir('..')


''' EXAMPLE 1 '''

video_name = "demo1"

log_text('logs/log.txt', f'Pose processing started for {video_name}')
processor = PoseProcessor(video_name)
frame_dir = processor.process_video()
log_text('logs/log.txt', f'Pose processing completed for {video_name}')

log_text('logs/log.txt', f"Starting video creation...")
frame_dir = f"frames/posemodule/{video_name}/"
vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_posemodule.mp4")

log_text('logs/log.txt', f"Output video saved at: {output_mp4_file}")



''' EXAMPLE 2 '''

video_name = "demo2"

log_text('logs/log.txt', f'Pose processing started for {video_name}')
processor = PoseProcessor(video_name)
frame_dir = processor.process_video()
log_text('logs/log.txt', f'Pose processing completed for {video_name}')

log_text('logs/log.txt', f"Starting video creation...")
frame_dir = f"frames/posemodule/{video_name}/"
vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output.avi", fps=30)
output_mpeg4_file = vc.create_video()
output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_posemodule.mp4")

log_text('logs/log.txt', f"Output video saved at: {output_mp4_file}")