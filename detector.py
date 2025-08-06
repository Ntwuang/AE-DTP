import torch
import cv2
import numpy as np
from typing import List, Dict

class LicensePlateDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Đường dẫn đến thư mục yolov5
        yolov5_repo_path = '/Users/nhatquang/CPC1HN/pipeline/yolov5' 
        
        self.device = device
        self.model = torch.hub.load(yolov5_repo_path, 'custom', 
                                  path=model_path, source='local')
        self.model.to(self.device).eval()
        
        # Warmup model
        with torch.no_grad():
            self.model(torch.zeros(1, 3, 640, 640).to(self.device))
        
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Phát hiện biển số trên nhiều ảnh cùng lúc"""
        # Chuyển đổi sang tensor và đưa vào device
        results = self.model(images)
        
        batch_plates = []
        for img, result in zip(images, results.xyxy):
            plates = []
            for *box, conf, cls in result:
                if conf > 0.5:  # Ngưỡng tin cậy
                    x1, y1, x2, y2 = map(int, box)
                    plate_img = img[y1:y2, x1:x2]
                    plates.append({
                        'image': plate_img,
                        'box': (x1, y1, x2, y2),
                        'conf': float(conf)
                    })
            batch_plates.append(plates)
        
        return batch_plates
