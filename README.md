# Posture-Detection-in-Videos
Posture detection in videos using Optical Flow detection models like Lucas Kanade, Horn Schunk and deep learning-based approaches like PoseNet and Mediapipe PoseDetector.

Apporaches used:
- **Optical Flow**: 
  - Lucas-Kanade method
  - Horn-Schunck method
- **Deep Learning**:
    - PoseNet
    - Mediapipe PoseDetector
    - Vanilla CNN

## Repo usage
1. Clone the repository:
   ```bash
   > cd <your_preferred_directory>
   >  git clone https://github.com/SohamD34/Posture-Detection-in-Videos.git
   ```
2. Navigate to the cloned directory:
   ```bash
   > cd Posture-Detection-in-Videos
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   > python -m venv venv python=3.11
   > source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
    > pip install -r requirements.txt
    ```
5. Add the required video files to the `data` directory.
6. Run the Streamlit app script:
   ```bash
   > cd scripts
   > streamlit run app.py
   ```

Output can be observed in the Streamlit application itself or in the ```output``` directory.


