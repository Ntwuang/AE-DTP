import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from detector import LicensePlateDetector
from recognizer import LicensePlateRecognizer

class LicensePlatePipeline:
    def __init__(self, detect_model_path: str, ocr_model_path: str, ocr_config_path: str = None):
        self.detector = LicensePlateDetector(detect_model_path)
        self.recognizer = LicensePlateRecognizer(ocr_model_path, ocr_config_path)  # Thêm tham số config
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_image(self, image_path: str, output_dir: str = 'outputs') -> Dict:
        """Xử lý một ảnh duy nhất"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        base_name = os.path.basename(image_path).split('.')[0]
        os.makedirs(os.path.join(output_dir, 'detected'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plates'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # Detect biển số
        plates_list = self.detector.detect_batch([img])[0]
        results = []
        
        for i, plate in enumerate(plates_list):
            # Lưu ảnh biển số
            plate_path = os.path.join(output_dir, 'plates', f'{base_name}_plate_{i}.jpg')
            cv2.imwrite(plate_path, plate['image'])
            
            # Nhận diện văn bản
            text = self.recognizer.recognize_batch([plate['image']])[0]
            
            results.append({
                'plate_image': plate_path,
                'text': text,
                'confidence': plate['conf'],
                'bounding_box': plate['box']
            })
            
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
    
    def process_batch(self, image_paths: List[str], output_dir: str = 'outputs') -> List[Dict]:
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

if __name__ == '__main__':
    # Khởi tạo pipeline với cấu hình OCR
    pipeline = LicensePlatePipeline(
        detect_model_path='models/best.pt',
        ocr_model_path='models/ckpt-epoch_30-acc_0.962.onnx',
        ocr_config_path='cct_xs_v1_global_plate_config.yaml'  # Thêm đường dẫn config
    )
    
    # Lấy danh sách ảnh từ thư mục inputs
    input_dir = 'inputs'
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục inputs/")
    else:
        # Xử lý hàng loạt
        results = pipeline.process_batch(image_paths)
        
        # In kết quả
        for result in results:
            print(f"\nKết quả cho ảnh: {result['image_path']}")
            for i, plate in enumerate(result['plates']):
                print(f"  Biển số {i+1}: {plate['text']} (Độ tin cậy: {plate['confidence']:.2f})")
            print(f"  Ảnh kết quả: {result['output_image']}")