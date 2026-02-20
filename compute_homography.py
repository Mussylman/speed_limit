"""Compute homography matrix from user-clicked points and save to YAML + .npy."""

import numpy as np
import cv2
import yaml
import os

# --------------- Source points (pixels clicked by user) ---------------
src_pts = np.array([
    [702, 715],
    [480, 212],
    [621, 185],
    [1280, 505],
], dtype=np.float32)

# --------------- Real-world dimensions & scale ---------------
real_width_m = 3.5
real_length_m = 20.0
scale_px_per_m = 100  # pixels per meter in the BEV image

# --------------- Destination points (bird's-eye view) ---------------
dst_pts = np.array([
    [0, 0],
    [350, 0],        # real_width_m * scale
    [350, 2000],     # real_width_m * scale, real_length_m * scale
    [0, 2000],       # 0, real_length_m * scale
], dtype=np.float32)

# --------------- Compute homography ---------------
H, status = cv2.findHomography(src_pts, dst_pts)

print("Homography matrix:")
print(H)
print()
print("Inlier mask:", status.ravel().tolist())

# --------------- Prepare YAML data ---------------
config_data = {
    "homography": {
        "src_points": src_pts.astype(int).tolist(),
        "real_width_m": float(real_width_m),
        "real_length_m": float(real_length_m),
        "scale_px_per_m": int(scale_px_per_m),
        "matrix": [[float(H[i][j]) for j in range(3)] for i in range(3)],
        "image_size": [1920, 1080],
    }
}

# --------------- Save files ---------------
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
os.makedirs(config_dir, exist_ok=True)

yaml_path = os.path.join(config_dir, "homography_Camera_21.yaml")
npy_path = os.path.join(config_dir, "homography_Camera_21_matrix.npy")

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

np.save(npy_path, H)

print(f"\nSaved YAML config : {yaml_path}")
print(f"Saved matrix .npy  : {npy_path}")
