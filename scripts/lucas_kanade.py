import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

file_name = 'demo1'
video_path = f'data/{file_name}.mp4'
output_dir = f'frames/lucas_kanade/{file_name}'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Shi-Tomasi params
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas Kanade params
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret:
    print("Error reading video.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

frame_id = 0

while True:
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:

        # Optical flow computation
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)


    cv2.imwrite(f'{output_dir}/frame_{frame_id}.jpg', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    old_gray = frame_gray.copy()


cap.release()
cv2.destroyAllWindows()
