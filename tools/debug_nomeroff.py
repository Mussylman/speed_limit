"""Debug NomeroffNet pipeline output format."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# Load pipeline
print("Loading NomeroffNet...")
pipe = pipeline(
    "number_plate_detection_and_reading_runtime",
    off_number_plate_classification=True,
    default_label="kz",
    default_lines_count=1,
    path_to_model=os.path.join(os.path.dirname(os.path.dirname(__file__)), "nomeroff-net", "data", "models", "Detector", "yolov11x", "yolov11x-keypoints-2024-10-11.engine"),
)
print("Ready.\n")

# Test on a failed crop
crop_dir = r"C:\Users\gghuk\speed_limit\outputs\Camera_108_run_2026-02-12_11-17-27\failed\images"
test_images = [
    "7a193083_155SJNA02.jpg",   # real: 155 SJA 02
    "dd55af06_3B0BBE13.jpg",    # real: 380 BBQ 13
    "1732e06f_707TO5N7.jpg",    # real: 707 OT? 07
]

for img_name in test_images:
    img_path = os.path.join(crop_dir, img_name)
    if not os.path.exists(img_path):
        print(f"SKIP: {img_path}")
        continue

    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"=== {img_name} ({img.shape[1]}x{img.shape[0]}) ===")

    res = pipe([rgb])

    # Print raw structure
    print(f"  type(res) = {type(res)}")
    if isinstance(res, (tuple, list)):
        print(f"  len(res) = {len(res)}")
        for i, item in enumerate(res):
            t = type(item).__name__
            if isinstance(item, (list, tuple)):
                print(f"  res[{i}]: {t}, len={len(item)}")
                if len(item) > 0:
                    inner = item[0]
                    inner_t = type(inner).__name__
                    if isinstance(inner, np.ndarray):
                        print(f"    [0]: ndarray shape={inner.shape} dtype={inner.dtype}")
                    elif isinstance(inner, (list, tuple)):
                        print(f"    [0]: {inner_t} len={len(inner)}")
                        if len(inner) > 0:
                            print(f"      [0][0]: {type(inner[0]).__name__} = {inner[0]}")
                    else:
                        print(f"    [0]: {inner_t} = {inner}")
            elif isinstance(item, np.ndarray):
                print(f"  res[{i}]: ndarray shape={item.shape} dtype={item.dtype}")
            else:
                print(f"  res[{i}]: {t} = {item}")

    # Try to extract data the way plate_recognizer does
    print("\n  --- Extracting like plate_recognizer ---")
    if isinstance(res, tuple) and len(res) > 1:
        data = res[1][0] if res[1] else None
        print(f"  data from res[1][0]: type={type(data).__name__}")
        if data is not None:
            if isinstance(data, (list, tuple)):
                print(f"  len(data) = {len(data)}")
                for j in range(min(len(data), 10)):
                    v = data[j]
                    if isinstance(v, np.ndarray):
                        print(f"    data[{j}]: ndarray shape={v.shape}")
                    elif isinstance(v, (list, tuple)):
                        print(f"    data[{j}]: {type(v).__name__} = {v}")
                    else:
                        print(f"    data[{j}]: {type(v).__name__} = {v}")

    # Also try unzip approach
    print("\n  --- unzip approach ---")
    try:
        result = unzip(res)
        print(f"  type(unzip) = {type(result).__name__}")
        if isinstance(result, (list, tuple)):
            print(f"  len = {len(result)}")
            for j in range(min(len(result), 10)):
                v = result[j]
                if isinstance(v, np.ndarray):
                    print(f"    [{j}]: ndarray shape={v.shape}")
                elif isinstance(v, (list, tuple)):
                    print(f"    [{j}]: {type(v).__name__} len={len(v)}")
                    if len(v) > 0:
                        inner = v[0]
                        if isinstance(inner, np.ndarray):
                            print(f"      [0]: ndarray shape={inner.shape}")
                        else:
                            print(f"      [0]: {type(inner).__name__} = {inner}")
                else:
                    print(f"    [{j}]: {type(v).__name__} = {v}")
    except Exception as e:
        print(f"  unzip error: {e}")

    # Direct access: try all possible ways to get text and conf
    print("\n  --- Direct pipeline result ---")
    try:
        # Flat approach
        if isinstance(res, (tuple, list)):
            for i in range(len(res)):
                item = res[i]
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    for j in range(len(item)):
                        sub = item[j]
                        if isinstance(sub, str):
                            print(f"  STRING at res[{i}][{j}] = '{sub}'")
                        elif isinstance(sub, (list, tuple)):
                            for k in range(len(sub)):
                                if isinstance(sub[k], str):
                                    print(f"  STRING at res[{i}][{j}][{k}] = '{sub[k]}'")
                                elif isinstance(sub[k], (int, float)):
                                    print(f"  NUMBER at res[{i}][{j}][{k}] = {sub[k]}")
    except Exception as e:
        print(f"  Error: {e}")

    print()

# Also try with classification ON to compare
print("\n\n===== NOW WITH CLASSIFICATION ON =====\n")
pipe2 = pipeline(
    "number_plate_detection_and_reading_runtime",
    off_number_plate_classification=False,
    default_label="kz",
    default_lines_count=1,
    path_to_model=os.path.join(os.path.dirname(os.path.dirname(__file__)), "nomeroff-net", "data", "models", "Detector", "yolov11x", "yolov11x-keypoints-2024-10-11.engine"),
)
print("Pipeline with classification ON ready.\n")

for img_name in test_images:
    img_path = os.path.join(crop_dir, img_name)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"=== {img_name} (classification ON) ===")

    res = pipe2([rgb])
    if isinstance(res, tuple) and len(res) > 1:
        data = res[1][0] if res[1] else None
        if data and isinstance(data, (list, tuple)):
            print(f"  len(data) = {len(data)}")
            for j in range(len(data)):
                v = data[j]
                if isinstance(v, np.ndarray):
                    print(f"    data[{j}]: ndarray shape={v.shape}")
                elif isinstance(v, (list, tuple)):
                    print(f"    data[{j}]: {type(v).__name__} = {v}")
                else:
                    print(f"    data[{j}]: {type(v).__name__} = {v}")
    print()
