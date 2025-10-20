import cv2
import numpy as np
import os

def crop(img, edge_thickness=50):
    h, w = img.shape[:2]

    top = img[:edge_thickness, :]
    right = img[:, -edge_thickness:]
    bottom = img[-edge_thickness:, :]
    left = img[:, :edge_thickness]
    
    top_resized = cv2.resize(top, (w, edge_thickness))
    bottom_resized = cv2.resize(bottom, (w, edge_thickness))
    left_resized = cv2.resize(left, (w, edge_thickness))
    right_resized = cv2.resize(right, (w, edge_thickness))

    combined = np.hstack([top_resized, right_resized, bottom_resized, left_resized])
    return combined

def process_folder(input_folder, output_folder="output_edges"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_ext = (".jpg", ".jpeg", ".png")
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_ext):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{filename}")
        
        img = cv2.imread(input_path)
        if img is None:
            continue
        result = crop(img, edge_thickness=100)
        cv2.imwrite(output_path, result)

if __name__=="__main__":
    # img_pth = r"tmp\66c4202fb23b1_CAP_48BB1EC5-946D-406B-B1AD-663170AF6B86E9061F69-3FFA-4A55-84AA-D1DA6F9B2995_compressed.jpg"
    # img = cv2.imread(img_pth)
    # result = crop(img, edge_thickness=100)
    # cv2.imwrite("edges_crop.jpg", result)

    input_folder = "border"
    process_folder(input_folder, output_folder="output_edges")