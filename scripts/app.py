import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from cvzone.PoseModule import PoseDetector
from natsort import natsorted
import subprocess

os.chdir('/home/soham/Desktop/GitHub/Posture-Detection-in-Videos/')


class LucasKanadePositionDetector:
    '''
    Class to process a video and detect positions using Lucas-Kanade Optical Flow.

    Args:
        video_name (str): Path to the input video file.
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

    def __init__(self, video_name, output_dir, max_corners=100, quality_level=0.3,
                 min_distance=7, block_size=7, win_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        
        self.video_name = video_name
        self.output_dir = output_dir# ➕ Draw lines between all detected new points

        self.feature_params = dict(maxCorners=max_corners, qualityLevel=quality_level,
                                   minDistance=min_distance, blockSize=block_size)
        self.lk_params = dict(winSize=win_size, maxLevel=max_level, criteria=criteria)

    def detect(self):
        cap = cv2.VideoCapture(self.video_name)
        ret, old_frame = cap.read()

        if not ret:
            print(f"[Error] Could not read the video: {self.video_name}")
            cap.release()
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        frame_id = 0

        while True:
            frame_id += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

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

    def __init__(self, video_name, output_dir, alpha=1.0, num_iter=100):
        self.video_name = video_name
        self.output_dir = output_dir
        self.alpha = alpha
        self.num_iter = num_iter

        os.makedirs(self.output_dir, exist_ok=True)

    def _horn_schunck(self, im1, im2):
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
        It = im2 - im1

        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)

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
        cap = cv2.VideoCapture(self.video_name)
        ret, old_frame = cap.read()
        if not ret:
            print(f"[Error] Could not read video: {self.video_name}")
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

            u, v = self._horn_schunck(old_gray, frame_gray)

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
        self.video_name = f"data/{video_name}.mp4"
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
        self.cap = cv2.VideoCapture(self.video_name)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.video_name}")


    def _setup_pose_detector(self):
        self.detector = PoseDetector(
            staticMode=False,
            modelComplexity=1,
            smoothLandmarks=True,
            enableSegmentation=False,
            smoothSegmentation=True,
            detectionCon=0.5,
            trackCon=0.5
        )

    def process_video(self):
        while True:
            success, img = self.cap.read()
            if not success or img is None:
                print("Video ended. Processing complete.")
                break

            img = self.detector.findPose(img)
            lmList, bboxInfo = self.detector.findPosition(img, draw=True, bboxWithHands=False)

            if lmList:
                center = bboxInfo["center"]
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

                length, img, _ = self.detector.findDistance(
                    lmList[11][0:2],
                    lmList[15][0:2],
                    img=img,
                    color=(255, 0, 0),
                    scale=10
                )

                angle, img = self.detector.findAngle(
                    lmList[11][0:2],
                    lmList[13][0:2],
                    lmList[15][0:2],
                    img=img,
                    color=(0, 0, 255),
                    scale=10
                )

                isClose = self.detector.angleCheck(
                    myAngle=angle,
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
    print(f"Video successfully saved to: {output_path}")
    os.remove(input_path)
    return input_path
    



def process_video(video_name, detector):

    video_name = video_name.split("/")[-1].split(".")[0]
    print("VIDEO NAME =", video_name)


    if detector == "horn_schunck":
        frame_dir = f"frames/horn_schunck/{video_name}"
        video_path = f"output/{video_name}_output_horn_schunck.avi"
        detector = HornSchunckPositionDetector(
                        video_name=video_name,
                        output_dir=frame_dir,
                        alpha=1.0,
                        num_iter=100
                    )

    elif detector == "lucas_kanade":
        frame_dir = f"frames/lucas_kanade/{video_name}"
        video_path = f"output/{video_name}_output_lucas_kanade.avi"
        lucas_kanade_detector = LucasKanadePositionDetector(video_name, frame_dir)
        lucas_kanade_detector.detect()


    elif detector == "posemodule":
        video_path = f"output/{video_name}_output.avi"
        pose_detector = PoseProcessor(video_name)
        frame_dir = pose_detector.process_video()

    else:
        st.error("Invalid detector selected.")
        return None
    
    vc = VideoCreator(frame_dir, output_path=video_path, fps=30)
    output_mpeg4_file = vc.create_video()

    output_path = f"output/{video_name}_output_horn_schunck.mp4"
    output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, output_path)

    return output_path




st.title("Posture Detection in Videos")

video_file_path = st.text_input("Enter the path to the video file")
detector_option = st.selectbox("Select a detector", ["horn_schunck", "lucas_kanade", "posemodule"])

if st.button("Process Video"):

    if video_file_path:
        if os.path.exists(video_file_path):
            st.success("Video file found.")
            output_path = process_video(video_file_path, detector_option)
            if output_path:
                st.success(f"Processed video saved to: {output_path}")
                st.video(output_path)
        else:
            st.error("Error: Video file not found.")
    else:
        st.error("Please enter a valid video file path.")