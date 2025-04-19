# Monocular Visual Odometry with Loop Detection

This repository implements a real‑time monocular visual odometry (VO) pipeline with lightweight loop detection, tested on the KITTI dataset. Due to overwhelming size of the dataset, only the left folder of the sequence of the dataset is available in the repo.

## Prerequisites:
```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline with:

```bash
python main.py 
```

or

```bash
python main.py \
  --dataset_path 00 \
  --path_folder image_0 \
  --ground_truth 00.txt
```

- `--dataset_path`: path to the KITTI sequence folder (e.g., `00`)
- `--path_folder`: image subfolder (e.g., `image_0` for left camera)
- `--ground_truth`: filename of the poses file (e.g., `00.txt`)

---

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


## Prerequisites

- Python 3.7 or higher
- [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (e.g., sequence `00`)

### Python dependencies
You can install the required Python packages via:

```bash
pip install opencv-python numpy networkx matplotlib scikit-learn
```

Alternatively, create a `requirements.txt` file:

```text
opencv-python
numpy
networkx
matplotlib
scikit-learn
```

and run:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
.
├── main.py                # Entry point for VO + loop detection
├── feature_extractor.py   # ORB detection and descriptor extraction
├── feature_matching.py    # Stereo matching & helper routines
├── pose_estimation.py     # Essential matrix computation & pose recovery
├── loop_closure.py        # BoW vocabulary + loop detection functions
├── bundle_adjustment.py   # (optional) pose-graph optimization stubs
├── helpers.py             # Utility functions (file I/O, calibration parsing)
├── display.py             # Visualization routines
└── README.md              # This file
```

---



### Optional flags

You can adjust keyframe interval and other parameters by editing the constants in `main.py`:

- **Keyframe interval**: change `keyframe_threshold` from `5` to another integer
- **Descriptor limit**: adjust `nfeatures` in `feature_extractor.py`
- **Loop-similarity threshold**: modify `similarity_threshold` in `loop_closure.py`


---

## Tips & Troubleshooting

- Ensure the KITTI folder structure remains intact (`00/image_0`, `00/calib.txt`, `00/00.txt`).
- If you encounter file-not-found errors, double-check the `--dataset_path` and `--path_folder` arguments.
- Increase the Python recursion limit or buffer sizes if processing very long sequences.

---

## License

This code is released for academic use under the MIT License. Feel free to cite our work if you find it useful!

