import os
import cv2
import warnings
from cvzone.PoseModule import PoseDetector
from natsort import natsorted
import subprocess
warnings.filterwarnings("ignore")

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"



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

    video_name = "demo1"
    processor = PoseProcessor(video_name)
    frame_dir = processor.process_video()

    frame_dir = f"frames/posemodule/{video_name}/"
    vc = VideoCreator(frame_dir, output_path=f"output/{video_name}_output.avi", fps=30)
    output_mpeg4_file = vc.create_video()
    output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{video_name}_output_posemodule.mp4")

    print(f"Output video saved at: {output_mp4_file}")