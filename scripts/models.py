import os
import cv2
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from cvzone.PoseModule import PoseDetector
from natsort import natsorted
import matplotlib.pyplot as plt
import subprocess
import random
warnings.filterwarnings("ignore")


class LucasKanadePositionDetector:
    '''
    Class to process a video and detect positions using Lucas-Kanade Optical Flow.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the processed frames.
        max_corners (int): Maximum number of corners to return. Default is 100.
        quality_level (float): Quality level for corner detection. Default is 0.3.
        min_distance (int): Minimum distance between detected corners. Default is 7.
        block_size (int): Size of the averaging block for corner detection. Default is 7.
        win_size (tuple): Size of the search window for optical flow. Default is (15, 15).
        max_level (int): Maximum number of pyramid levels for optical flow. Default is 2.
        criteria (tuple): Criteria for termination of the iterative search. Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03).

    Methods:
        detect(): Processes the video frame by frame, detecting positions and saving frames.
    
    Returns:
        Images with detected positions are saved in the specified output directory.
    '''

    def __init__(self, video_path, output_dir, max_corners=100, quality_level=0.3,
                 min_distance=7, block_size=7, win_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        
        self.video_path = video_path
        self.output_dir = output_dir# ➕ Draw lines between all detected new points

        self.feature_params = dict(maxCorners=max_corners, qualityLevel=quality_level,
                                   minDistance=min_distance, blockSize=block_size)
        self.lk_params = dict(winSize=win_size, maxLevel=max_level, criteria=criteria)

    def detect(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, old_frame = cap.read()

        if not ret:
            print(f"[Error] Could not read the video: {self.video_path}")
            cap.release()
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)            # Corner detection 

        frame_id = 0

        while True:
            frame_id += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if p0 is not None and len(p0) > 0:

                # Creates image pyramids (scaled down versions) - old frames are at top of pyramid, new frames at bottom of pyramid
                # Calculates the corresponding points in the next frame using Lucas-Kanade method using displacement vector (dx, dy)

                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)        # Optical flow calculation
                
                # Based on error value, status - tracked or untracked
                # If tracked - good point
                # If untracked - discard point

                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                        cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                    # ➕ Draw lines between all detected new points
                    # for i in range(len(good_new)):
                    #     for j in range(i + 1, len(good_new)):
                    #         pt1 = tuple(good_new[i].ravel().astype(int))
                    #         pt2 = tuple(good_new[j].ravel().astype(int))
                    #         cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)

            cv2.imwrite(os.path.join(self.output_dir, f'frame_{frame_id}.jpg'), frame)

            if cv2.waitKey(30) & 0xFF == 27:
                break

            old_gray = frame_gray.copy()

        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {self.output_dir}")




class HornSchunckPositionDetector:
    '''
    Horn-Schunck optical flow detector that marks bright red points at motion hotspots.
    '''

    def __init__(self, video_path, output_dir, alpha=1.0, num_iter=100):
        self.video_path = video_path
        self.output_dir = output_dir
        self.alpha = alpha
        self.num_iter = num_iter

        os.makedirs(self.output_dir, exist_ok=True)

    def _horn_schunck(self, im1, im2):
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        # Compute gradients using Sobel operator (both horizontal + vertical)
        # -1 0 1
        # -2 0 2
        # -1 0 1
        
        Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
        It = im2 - im1

        # Horizontal and vertical components of flow vectors
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)

        # Gaussian kernel for averaging/smoothing the flow
        kernel = np.array([[1/12, 1/6, 1/12],
                           [1/6,   0,  1/6],
                           [1/12, 1/6, 1/12]])

        for _ in range(self.num_iter):
            u_avg = cv2.filter2D(u, -1, kernel)
            v_avg = cv2.filter2D(v, -1, kernel)

            deriv = (Ix * u_avg + Iy * v_avg + It) / (self.alpha**2 + Ix**2 + Iy**2)
            u = u_avg - Ix * deriv
            v = v_avg - Iy * deriv

        return u, v

    def detect(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, old_frame = cap.read()
        if not ret:
            print(f"[Error] Could not read video: {self.video_path}")
            cap.release()
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_id = 0

        while True:
            frame_id += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get optical flow vectors using Horn-Schunck method
            u, v = self._horn_schunck(old_gray, frame_gray)

            # Calculate magnitude of flow vectors
            mag, _ = cv2.cartToPolar(u, v)
            threshold = 0.2  # Lower threshold to capture subtle motion

            # Copy original frame to mark red points
            output_frame = frame.copy()

            # Draw bright red circles at high-motion points
            y_indices, x_indices = np.where(mag > threshold)
            for y, x in zip(y_indices, x_indices):
                cv2.circle(output_frame, (x, y), 2, (0, 0, 255), -1)  # Bright red dot

            cv2.imwrite(os.path.join(self.output_dir, f'frame_{frame_id}.jpg'), output_frame)

            old_gray = frame_gray.copy()

        cap.release()
        cv2.destroyAllWindows()
        print(f"[Done] Processed video saved to: {self.output_dir}")
        return self.output_dir
    





class PoseProcessor:
    '''
    Class to process a video and detect poses using Mediapipe PoseDetector, OpenCV and cvzone.

    Args:
        video_name (str): Name of the video file (without extension) to process.
        output_dir (str): Directory to save the processed frames. Default is "frames/posemodule".
        target_angle (int): Target angle for the arm. Default is 50.
        offset (int): Offset for angle checking. Default is 10.

    Methods:
        _setup_dirs(): Creates the output directory if it doesn't exist.
        _setup_video(): Initializes the video capture object.
        _setup_pose_detector(): Initializes the PoseDetector object.
        process_video(): Processes the video frame by frame, detecting poses and saving frames.

    Returns:
        Images with detected poses and angles are saved in the specified output directory.
    '''

    def __init__(self, video_name, output_dir="frames/posemodule", target_angle=50, offset=10):
        
        self.video_name = video_name
        self.video_path = f"data/{video_name}.mp4"
        self.output_path = os.path.join(output_dir, video_name)
        self.frame_count = 0
        self.target_angle = target_angle
        self.offset = offset

        self._setup_dirs()
        self._setup_video()
        self._setup_pose_detector()


    def _setup_dirs(self):
        os.makedirs(self.output_path, exist_ok=True)


    def _setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.video_path}")


    def _setup_pose_detector(self):
        self.detector = PoseDetector(
            staticMode=False,
            modelComplexity=1,
            smoothLandmarks=True,                   
            enableSegmentation=False,               # Helps segment body from background, background removal
            smoothSegmentation=True,                # Smoothes the segmentation
            detectionCon=0.5,
            trackCon=0.5
        )

    def process_video(self):
        while True:
            success, img = self.cap.read()
            if not success or img is None:
                print("Video ended. Processing complete.")
                break
            
            # Detects landmarks - face, eyes, shoulders, elbows, etc.
            img = self.detector.findPose(img)

            # Detects bounding box
            # For this, it detects the extreme points of the body - left to right shoulder points
            # Top to bottom - head, feet, shoulder aligned hips
            # Then it creates a box using these extremety points
            lmList, bboxInfo = self.detector.findPosition(img, draw=True, bboxWithHands=False)


            if lmList:
                center = bboxInfo["center"]
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

                length, img, _ = self.detector.findDistance(
                    lmList[11][0:2],            # left shoulder
                    lmList[15][0:2],            # right shoulder
                    img=img,
                    color=(255, 0, 0),
                    scale=10
                )

                angle, img = self.detector.findAngle(
                    lmList[11][0:2],            # left shoulder
                    lmList[13][0:2],            # left elbow
                    lmList[15][0:2],            # right shoulder
                    img=img,
                    color=(0, 0, 255),
                    scale=10
                )

                isClose = self.detector.angleCheck(
                    myAngle=angle,              # angle of the arm
                    targetAngle=self.target_angle,
                    offset=self.offset
                )

                print(f"Frame {self.frame_count}: Angle ~{self.target_angle}°? {isClose}")

            out_file = os.path.join(self.output_path, f"frame_{self.frame_count:04}.jpg")
            cv2.imwrite(out_file, img)
            self.frame_count += 1

            cv2.waitKey(1)

        self.cap.release()
        cv2.destroyAllWindows()
        return self.output_path



class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 8x6 -> 16x12
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # 16x12 -> 32x24
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # 32x24 -> 64x48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_keypoints, kernel_size=3, padding=1)             # Final heatmap
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x