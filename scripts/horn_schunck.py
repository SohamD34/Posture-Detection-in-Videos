import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Horn-Schunck implementation
def horn_schunck(im1, im2, alpha=1.0, num_iter=100):
    
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Compute gradients
    Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
    It = im2 - im1

    # Initialize flow vectors
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,   0,  1/6],
                       [1/12, 1/6, 1/12]])

    for _ in range(num_iter):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)

        deriv = (Ix * u_avg + Iy * v_avg + It) / (alpha**2 + Ix**2 + Iy**2)
        u = u_avg - Ix * deriv
        v = v_avg - Iy * deriv

    return u, v


# Setup
file_name = 'demo1'
video_path = f'data/{file_name}.mp4'
output_dir = f'frames/horn_schunck/{file_name}'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
frame_id = 0

while True:
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Horn-Schunck
    u, v = horn_schunck(old_gray, frame_gray, alpha=1.0, num_iter=100)

    # Visualization
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay flow on frame
    combined = cv2.addWeighted(frame, 0.6, flow_rgb, 0.4, 0)

    # Save frame
    cv2.imwrite(f'{output_dir}/frame_{frame_id}.jpg', combined)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
