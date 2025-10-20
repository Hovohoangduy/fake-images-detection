import os
import cv2
import numpy as np
import pywt

def wavelet_transform_image(img, wavelet="db2", level=1, threshold_ratio=0.2):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    coeffs2 = pywt.dwt2(gray, wavelet=wavelet)
    LL, (LH, HL, HH) = coeffs2

    high_freq_energy = np.abs(LH) + np.abs(HL) + np.abs(HH)
    high_freq_energy = high_freq_energy / np.max(high_freq_energy)
    high_freq_energy = (high_freq_energy * 255).astype(np.uint8)

    thresh_val = int(threshold_ratio * 255)
    _, binary_map = cv2.threshold(high_freq_energy, thresh_val, 255, cv2.THRESH_BINARY)

    return binary_map

def process_folder(input_folder, output_folder="output_wavelet"):
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

        wavelet_img = wavelet_transform_image(img, wavelet="haar", level=1)
        cv2.imwrite(output_path, wavelet_img)

if __name__=="__main__":
    input_folder = "real"
    process_folder(input_folder, output_folder="output_real")