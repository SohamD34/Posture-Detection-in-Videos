import os
import cv2
import numpy as np
import warnings
from natsort import natsorted
import subprocess
warnings.filterwarnings("ignore")

# class HornSchunckPositionDetector:
#     '''
#     Class to detect motion in a video using the Horn-Schunck optical flow method.

#     Args:
#         video_path (str): Path to the input video file.
#         output_dir (str): Directory to save the output frames. Default is "frames/horn_schunck".
#         alpha (float): Regularization parameter for the Horn-Schunck method. Default is 1.0.
#         num_iter (int): Number of iterations for the algorithm. Default is 100.

#     Methods:
#         _horn_schunck(im1, im2): Computes the optical flow between two images using the Horn-Schunck method.
#         detect(): Processes the video frame by frame, applying the Horn-Schunck method and saving the output frames.
    
#     Returns:
#         A directory containing the processed frames with detected motion.
#     '''

#     def __init__(self, video_path, output_dir, alpha=1.0, num_iter=100):
#         self.video_path = video_path
#         self.output_dir = output_dir
#         self.alpha = alpha
#         self.num_iter = num_iter

#         os.makedirs(self.output_dir, exist_ok=True)

#     def _horn_schunck(self, im1, im2):
#         im1 = im1.astype(float) / 255.
#         im2 = im2.astype(float) / 255.

#         Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
#         Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
#         It = im2 - im1

#         u = np.zeros(im1.shape)
#         v = np.zeros(im1.shape)

#         kernel = np.array([[1/12, 1/6, 1/12],
#                            [1/6,   0,  1/6],
#                            [1/12, 1/6, 1/12]])

#         for _ in range(self.num_iter):
#             u_avg = cv2.filter2D(u, -1, kernel)
#             v_avg = cv2.filter2D(v, -1, kernel)

#             deriv = (Ix * u_avg + Iy * v_avg + It) / (self.alpha**2 + Ix**2 + Iy**2)
#             u = u_avg - Ix * deriv
#             v = v_avg - Iy * deriv

#         return u, v


#     def detect(self):
#         cap = cv2.VideoCapture(self.video_path)
#         ret, old_frame = cap.read()
#         if not ret:
#             print(f"[Error] Could not read video: {self.video_path}")
#             cap.release()
#             return

#         old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#         frame_id = 0

#         while True:
#             frame_id += 1
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             u, v = self._horn_schunck(old_gray, frame_gray)

#             hsv = np.zeros_like(frame)
#             hsv[..., 1] = 255

#             mag, ang = cv2.cartToPolar(u, v)
#             hsv[..., 0] = ang * 180 / np.pi / 2
#             hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

#             flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#             combined = cv2.addWeighted(frame, 0.6, flow_rgb, 0.4, 0)

#             cv2.imwrite(os.path.join(self.output_dir, f'frame_{frame_id}.jpg'), combined)

#             if cv2.waitKey(30) & 0xFF == 27:
#                 break

#             old_gray = frame_gray.copy()

#         cap.release()
#         cv2.destroyAllWindows()
#         print(f"[Done] Processed video saved to: {self.output_dir}")
#         return self.output_dir



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
    return input_path



if __name__ == "__main__":

    file_name = 'demo1'

    detector = HornSchunckPositionDetector(
        video_path=f"data/{file_name}.mp4",
        output_dir=f"frames/horn_schunck/{file_name}",
        alpha=1.0,
        num_iter=100
    )
    frame_dir = detector.detect()


    vc = VideoCreator(frame_dir, output_path=f"output/{file_name}_output_horn_schunck.avi", fps=30)
    output_mpeg4_file = vc.create_video()
    output_mp4_file = convert_to_modern_mp4(output_mpeg4_file, f"output/{file_name}_output_horn_schunck.mp4")

    print(f"Output video saved at: {output_mp4_file}")
