import onnxruntime as ort
import numpy as np
import cv2
import yaml
from typing import List, Optional, Dict, Any

class LicensePlateRecognizer:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Khởi tạo bộ nhận diện biển số
        
        Args:
            model_path: Đường dẫn đến file model ONNX
            config_path: Đường dẫn đến file cấu hình YAML (optional)
        """
        # Khởi tạo ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        self.input_details = self.session.get_inputs()[0]
        self.output_name = self.session.get_outputs()[0].name
        
        # Load cấu hình mặc định
        self.config = self._load_default_config()
        
        # Ghi đè bằng cấu hình từ file nếu có
        if config_path:
            self._load_config_from_file(config_path)
            
        # In thông tin cấu hình để debug
        self._print_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Tạo cấu hình mặc định"""
        return {
            'max_plate_slots': 9,
            'alphabet': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'pad_char': '_',
            'img_height': 64,
            'img_width': 128,
            'keep_aspect_ratio': False,
            'interpolation': 'linear',
            'image_color_mode': 'rgb',
            'min_plate_length': 7,
            'max_plate_length': 9
        }

    def _load_config_from_file(self, config_path: str):
        """Tải cấu hình từ file YAML"""
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                self.config.update(user_config)
                
        # Xử lý riêng danh sách ký tự
        if 'alphabet' in self.config:
            self.char_list = self.config['alphabet']
        else:
            self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def _print_config(self):
        """In thông tin cấu hình để debug"""
        print("\n=== OCR Configuration ===")
        print(f"Input size: {self.config['img_width']}x{self.config['img_height']}")
        print(f"Alphabet: {self.char_list}")
        print(f"Color mode: {self.config['image_color_mode'].upper()}")
        print(f"Plate length range: {self.config['min_plate_length']}-{self.config['max_plate_length']}\n")

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh theo cấu hình"""
        # Xử lý màu sắc - đảm bảo đầu ra là RGB
        if len(image.shape) == 2:  # Nếu ảnh grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    
        # Resize ảnh về đúng kích thước model yêu cầu (128x64)
        resized = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    
        # Chuẩn hóa kiểu dữ liệu
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
    
        # Thêm batch dimension (từ HWC thành NHWC)
        return np.expand_dims(resized, axis=0)  # Shape sẽ là [1, 64, 128, 3]
    
    def recognize_batch(self, plate_images: List[np.ndarray]) -> List[str]:
        """Nhận diện hàng loạt biển số"""
        processed = [self.preprocess_for_ocr(img) for img in plate_images]
        batch_input = np.concatenate(processed, axis=0)
        
        # Chạy inference
        outputs = self.session.run([self.output_name], 
                                 {self.input_details.name: batch_input})
        
        # Xử lý kết quả
        return [self._postprocess(self._decode(pred)) for pred in outputs[0]]

    def _decode(self, pred: np.ndarray) -> str:
        """Giải mã đầu ra model thành text"""
        pred_indices = np.argmax(pred, axis=1)
        text = ''
        
        for idx in pred_indices.flatten():
            if idx < len(self.char_list):
                text += self.char_list[idx]
                
        return text

    def _postprocess(self, text: str) -> str:
        """Hậu xử lý văn bản biển số"""
        # Lọc ký tự hợp lệ
        valid_chars = [c for c in text if c in self.char_list]
        cleaned = ''.join(valid_chars)[:self.config['max_plate_length']]
        
        # Kiểm tra cấu trúc biển số Việt Nam
        if len(cleaned) >= self.config['min_plate_length']:
            # Dạng 51A12345 hoặc 51A1234
            if (cleaned[:2].isdigit() and cleaned[2].isalpha() and 
                cleaned[3:].isdigit()):
                return cleaned
            
            # Dạng AB12345
            elif (cleaned[:2].isalpha() and cleaned[2:].isdigit()):
                return cleaned
        
        return cleaned

    def recognize_single(self, plate_image: np.ndarray) -> str:
        """Nhận diện từng ảnh biển số"""
        return self.recognize_batch([plate_image])[0]