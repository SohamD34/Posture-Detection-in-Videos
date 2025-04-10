import os
import cv2
import numpy as np
import warnings
import subprocess
from natsort import natsorted
warnings.filterwarnings("ignore")

# class LucasKanadePositionDetector:
#     '''
#     Class to process a video and detect positions using Lucas-Kanade Optical Flow.

#     Args:
#         video_path (str): Path to the input video file.
#         output_dir (str): Directory to save the processed frames.
#         max_corners (int): Maximum number of corners to return. Default is 100.
#         quality_level (float): Quality level for corner detection. Default is 0.3.
#         min_distance (int): Minimum distance between detected corners. Default is 7.
#         block_size (int): Size of the averaging block for corner detection. Default is 7.
#         win_size (tuple): Size of the search window for optical flow. Default is (15, 15).
#         max_level (int): Maximum number of pyramid levels for optical flow. Default is 2.
#         criteria (tuple): Criteria for termination of the iterative search. Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03).

#     Methods:
#         detect(): Processes the video frame by frame, detecting positions and saving frames.
    
#     Returns:
#         Images with detected positions are saved in the specified output directory.
#     '''

#     def __init__(self, video_path, output_dir, max_corners=100, quality_level=0.3,
#                  min_distance=7, block_size=7, win_size=(15, 15), max_level=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        
#         self.video_path = video_path
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#         # Feature and flow parameters
#         self.feature_params = dict(maxCorners=max_corners, qualityLevel=quality_level,
#                                    minDistance=min_distance, blockSize=block_size)
#         self.lk_params = dict(winSize=win_size, maxLevel=max_level, criteria=criteria)



#     def detect(self):
#         cap = cv2.VideoCapture(self.video_path)
#         ret, old_frame = cap.read()

#         if not ret:
#             print(f"[Error] Could not read the video: {self.video_path}")
#             cap.release()
#             return

#         old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#         p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

#         frame_id = 0

#         while True:
#             frame_id += 1
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             if p0 is not None and len(p0) > 0:
#                 p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

#                 if p1 is not None and st is not None:
#                     good_new = p1[st == 1]
#                     good_old = p0[st == 1]

#                     for new, old in zip(good_new, good_old):
#                         a, b = new.ravel()
#                         c, d = old.ravel()
#                         cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                         cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

#                     p0 = good_new.reshape(-1, 1, 2)
#                 else:
#                     p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
#             else:
#                 p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)

#             cv2.imwrite(os.path.join(self.output_dir, f'frame_{frame_id}.jpg'), frame)

#             if cv2.waitKey(30) & 0xFF == 27:
#                 break

#             old_gray = frame_gray.copy()

#         cap.release()
#         cv2.destroyAllWindows()
#         print(f"Processed video saved to: {self.output_dir}")


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
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Feature and flow parameters
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
                    for i in range(len(good_new)):
                        for j in range(i + 1, len(good_new)):
                            pt1 = tuple(good_new[i].ravel().astype(int))
                            pt2 = tuple(good_new[j].ravel().astype(int))
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

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





if __name__ == "__main__":

    video_name = "demo2"
    video_path = f"data/{video_name}.mp4"
    frame_dir = f"frames/lucas_kanade/{video_name}/"

    lk_detector = LucasKanadePositionDetector(video_path, frame_dir)
    lk_detector.detect()


    frame_dir = f"frames/lucas_kanade/{video_name}/"
    vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output_lucas_kanade.avi", fps=30)
    output_mpeg4_file = vc.create_video()
    output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_lucas_kanade.mp4")
