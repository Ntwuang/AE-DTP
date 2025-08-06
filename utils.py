import cv2
import numpy as np

def read_image(path):
    """Đọc ảnh từ đường dẫn"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {path}")
    return img

def save_image(image, path):
    """Lưu ảnh ra file"""
    cv2.imwrite(path, image)

def preprocess_image_for_ocr(image):
    """Tiền xử lý ảnh trước khi đưa vào OCR"""
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm sắc nét ảnh
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Cân bằng histogram
    equalized = cv2.equalizeHist(sharpened)
    
    return equalized