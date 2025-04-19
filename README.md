# Monocular Visual Odometry with Loop Detection

This repository implements a real‑time monocular visual odometry (VO) pipeline with lightweight loop detection, tested on the KITTI dataset. Due to overwhelming size of the dataset, only the left folder of the sequence `00` of the dataset is available in the repo.

## Prerequisites:
```bash
pip install -r requirements.txt
```

---

## Usage

Run the main pipeline with:

```bash
python main.py 
```

or

```bash
python main.py 
  --dataset_path 00 
  --path_folder image_0 
  --ground_truth 00.txt
  --left False
```

- `--dataset_path`: path to the KITTI sequence folder (e.g., `00`)
- `--path_folder`: image subfolder (e.g., `image_0` for left camera)
- `--ground_truth`: filename of the poses file (e.g., `00.txt`)
- `--left`: whether the images used are left or right foler (`True` for left images, `False` for right images)

---

## Outputs

When the script finishes, it will produce:

1. **`camera_trajectory.csv`**: top-down (X, Z) trajectory of every frame
2. **`camera_trajectory.png`**: plotted trajectory
3. **`frame_by_frame_errors.csv`**: per-frame Euclidean error vs ground truth
4. **`frame_by_frame_error.png`**: plotted error over time

Additionally, during execution you will see:

- A pop-up window showing real-time tracked keypoints
- A live plot of the camera trajectory
- A pose graph window showing keyframe nodes and loop edges

Press `q` in any OpenCV window to exit early.

---

## Acknowledgement

I would like to express the greatest gratification towards the amazing TAs and the professor teaching CMPUT 428 winter 2025.

---

## License

This code is released for academic use under the MIT License. Feel free to cite our work if you find it useful!