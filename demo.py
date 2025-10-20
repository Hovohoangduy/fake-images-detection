import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob, os

def extract_features(img):
    features = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernels = {
        'h': np.array([[1, 0, -1]]),
        'v': np.array([[1], [0], [1]]),
        'd': np.array([[1, 0], [0, -1]]),
        'a': np.array([[0, 1], [-1, 0]])
    }
    
    for k in kernels.values():
        diff = cv2.filter2D(gray, -1, k)
        hist = cv2.calcHist([diff], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)
    features.extend(hist_lbp)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray.astype("float32") - blur.astype("float32")
    for stat in [np.mean, np.var, np.std, lambda x: np.mean(np.abs(x))]:
        features.append(stat(noise))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for c in cv2.split(img) + cv2.split(hsv):
        features.append(np.mean(c))
        features.append(np.std(c))
        features.append(np.mean((c - np.mean(c)) ** 3))
    Imax, Imin = np.max(gray), np.min(gray)
    contrast = (Imax - Imin) / (Imax + Imin + 1e-6)
    features.append(contrast)

    return np.array(features, dtype="float32")

def build_dataset(real_dir, recaptured_dir):
    X, y = [], []

    for path in glob.glob(os.path.join(real_dir, "*.jpg")):
        img = cv2.imread(path)
        X.append(extract_features(img))
        y.append(0)

    for path in glob.glob(os.path.join(recaptured_dir, "*.jpg")):
        img =  cv2.imread(path)
        X.append(extract_features(img))
        y.append(1)
    
    return np.array(X), np.array(y)

if __name__=="__main__":
    X, y = build_dataset(r"datasets\train\real", r"datasets\train\moire")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["real", "moire"]))