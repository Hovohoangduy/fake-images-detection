import cv2
import numpy as np
import os

def detect_black_border(img, border_thickness=20, ratio_thresh=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top = gray[0:border_thickness, :]
    bottom = gray[h-border_thickness:h, :]
    left = gray[:, 0:border_thickness]
    right = gray[:, w-border_thickness:w]

    ratios = {
        "top": dark_ratio(top),
        "bottom": dark_ratio(bottom),
        "left": dark_ratio(left),
        "right": dark_ratio(right)
    }
    
    border_detected = any(r > ratio_thresh for r in ratios.values())
    print("REGION DARK: ", ratios)
    if border_detected:
        print("FAKE IMAGES")
    else:
        print("REAL IMAGE")
    return border_detected, ratios

def dark_ratio(region, dark_thresh=40):
    dark_pixels = np.sum(region < dark_thresh)
    total_pixels = region.size
    return dark_pixels / total_pixels

def process_folder(input_folder, output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
    img_exts = (".jpg", ".jpeg", ".png")
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(img_exts)]
    for filename in img_files:
        img_pth = os.path.join(input_folder, filename)
        img = cv2.imread(img_pth)
        if img is None:
            print("[SKIP] Don't read image", filename)
            continue
        border_detected, ratios = detect_black_border(img)
        label = "FAKE" if border_detected else "REAL"
        color = (0, 0, 255) if border_detected else (0, 255, 0)
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)
        debug_text = f"T: {ratios['top']: .2f} B: {ratios['bottom']: .2f} L: {ratios['left']: .2f} R: {ratios['left']: .2f}"
        cv2.putText(img, debug_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        save_pth = os.path.join(output_folder, filename)
        cv2.imwrite(save_pth, img)

        print(f"[DONE] {filename} -> {label} | {debug_text}")

if __name__=="__main__":
    # img_pth = r"border/66b24977971f4_CAP8155653565902274793_compressed8226990011204787494.jpg"
    # img = cv2.imread(img_pth)
    # if img is None:
    #     raise ValueError("Don't read image")
    # detect_black_border(img)
    input_folder = "real"
    output_folder = "results_real"
    process_folder(input_folder, output_folder)