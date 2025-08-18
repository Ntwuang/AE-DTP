import os
import cv2
import time
import csv
import psutil
import GPUtil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from detector import LicensePlateDetector
from recognizer import LicensePlateRecognizer

class LicensePlatePipeline:
    def __init__(self, detect_model_path: str, ocr_model_path: str, ocr_config_path: str = None, log_file: str = 'license_plate_logs.csv'):
        self.detector = LicensePlateDetector(detect_model_path)
        self.recognizer = LicensePlateRecognizer(ocr_model_path, ocr_config_path)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.log_file = log_file
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'image_path', 'plate', 'box', 'vehicle_type',
                    'confidence', 'timing_crop', 'timing_ocr', 
                    'overall_decode', 'overall_yolo', 'overall_total',
                    'cpu_usage', 'ram_usage', 'gpu_usage', 'gpu_memory'
                ])
                writer.writeheader()
    
    def _get_system_metrics(self) -> Dict:
        """Get current system metrics including CPU, RAM, and GPU usage"""
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'ram_usage': psutil.virtual_memory().percent,
            'gpu_usage': 0,
            'gpu_memory': 0
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_usage'] = gpus[0].load * 100
                metrics['gpu_memory'] = gpus[0].memoryUsed
        except Exception as e:
            print(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def _write_log_entry(self, log_data: Dict):
        """Write a log entry to the CSV file"""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'image_path', 'plate', 'box', 'vehicle_type',
                'confidence', 'timing_crop', 'timing_ocr', 
                'overall_decode', 'overall_yolo', 'overall_total',
                'cpu_usage', 'ram_usage', 'gpu_usage', 'gpu_memory'
            ])
            
            # Convert box to string if it's a list
            if 'box' in log_data and isinstance(log_data['box'], list):
                log_data['box'] = str(log_data['box'])
            
            writer.writerow(log_data)
    
    def process_image(self, image_path: str, output_dir: str = '/home/nhatquang/final/outputs') -> Dict:
        """Xử lý một ảnh duy nhất"""
        start_time = time.time()
        system_metrics_start = self._get_system_metrics()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        base_name = os.path.basename(image_path).split('.')[0]
        os.makedirs(os.path.join(output_dir, 'detected'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plates'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # Detect biển số
        detect_start = time.time()
        plates_list = self.detector.detect_batch([img])[0]
        detect_time = time.time() - detect_start
        
        results = []
        
        for i, plate in enumerate(plates_list):
            plate_start_time = time.time()
            
            # Lưu ảnh biển số
            plate_path = os.path.join(output_dir, 'plates', f'{base_name}_plate_{i}.jpg')
            cv2.imwrite(plate_path, plate['image'])
            
            # Nhận diện văn bản
            ocr_start = time.time()
            text = self.recognizer.recognize_batch([plate['image']])[0]
            ocr_time = time.time() - ocr_start
            h, w = plate['image'].shape[:2]
            vehicle_type = "car" if w/h > 2 else "motorbike"
            
            results.append({
                'plate_image': plate_path,
                'text': text,
                'confidence': plate['conf'],
                'bounding_box': plate['box'],
                'vehicle_type': vehicle_type
            })
            
            # Prepare log data
            log_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_path': image_path,
                'plate': text,
                'box': plate['box'],
                'vehicle_type': vehicle_type,
                'confidence': plate['conf'],
                'timing_crop': f"{plate_start_time - detect_start:.3f}",
                'timing_ocr': f"{ocr_time:.3f}",
                'overall_decode': f"{ocr_time:.3f}",
                'overall_yolo': f"{detect_time:.3f}",
                'overall_total': f"{time.time() - start_time:.3f}",
            }
            
            # Add system metrics
            system_metrics = self._get_system_metrics()
            log_data.update({
                'cpu_usage': system_metrics['cpu_usage'],
                'ram_usage': system_metrics['ram_usage'],
                'gpu_usage': system_metrics['gpu_usage'],
                'gpu_memory': system_metrics['gpu_memory']
            })
            
            # Write log entry
            self._write_log_entry(log_data)
            
            # Vẽ kết quả lên ảnh
            x1, y1, x2, y2 = plate['box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Lưu ảnh kết quả
        result_path = os.path.join(output_dir, 'results', f'{base_name}_result.jpg')
        cv2.imwrite(result_path, img)
        
        return {
            'image_path': image_path,
            'output_image': result_path,
            'plates': results
        }
    
    def process_batch(self, image_paths: List[str], output_dir: str = '/home/nhatquang/final/outputs') -> List[Dict]:
        """Xử lý hàng loạt ảnh song song"""
        start_time = time.time()
        
        # Xử lý song song
        futures = [self.executor.submit(self.process_image, path, output_dir) for path in image_paths]
        results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        print(f"\nĐã xử lý {len(image_paths)} ảnh trong {total_time:.2f} giây")
        print(f"Trung bình {avg_time:.2f} giây/ảnh")
        
        return results

from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil

# Khởi tạo app FastAPI
app = FastAPI()

# Khởi tạo pipeline 
pipeline = LicensePlatePipeline(
    detect_model_path='/home/nhatquang/final/models/best.pt',
    ocr_model_path='/home/nhatquang/final/models/ckpt-epoch_30-acc_0.962.onnx',
    ocr_config_path='/home/nhatquang/final/cct_xs_v1_global_plate_config.yaml',
    log_file='license_plate_logs.csv'
)

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    # Lưu file upload vào thư mục tạm
    input_dir = "/home/nhatquang/final/uploads"
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = pipeline.process_image(input_path)

    # Ketqua tra ve
    response = {
        "plates": [
            {
                "plate_text": p["text"],
                "coordinates": list(map(int, p["bounding_box"])),
                "vehicle_type": p["vehicle_type"]
            } for p in result["plates"]
        ],
        "output_image": result["output_image"]
    }
    return response
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dochinhanhinputs:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["/home/nhatquang/final"]
    )

