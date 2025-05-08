import os
import cv2
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, yolo_model_path: str, output_dir: str = "output"):
        self.model = YOLO(yolo_model_path)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _crop_and_save(self, img, boxes, img_name_no_ext):
        cropped_images = []
        for i, box in enumerate(boxes.xywh):
            x_result, y_result, w, h = map(int, box)
            x1 = x_result - w // 2
            x2 = x_result + w // 2
            y1 = y_result - h // 2
            y2 = y_result + h // 2

            # 邊界檢查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            cropimg = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(self.output_dir, f"cropped_{img_name_no_ext}_{i+1}.jpg")
            cv2.imwrite(cropped_img_path, cropimg)
            cropped_images.append(cropped_img_path)

        return cropped_images

    def _annotate_and_save(self, img, boxes, img_name_no_ext):
        for i, box in enumerate(boxes.xywh):
            x_result, y_result, w, h = map(int, box)
            x1 = x_result - w // 2
            x2 = x_result + w // 2
            y1 = y_result - h // 2
            y2 = y_result + h // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Object {i+1}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        annotated_img_path = os.path.join(self.output_dir, f"annotated_{img_name_no_ext}.jpg")
        cv2.imwrite(annotated_img_path, img)

    def predict_result(self, img_path: str):
        img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        result = self.model.predict(source=img_path)

        # 提取框框結果
        boxes = result[0].boxes

        # 裁剪並儲存圖像
        cropped_images = self._crop_and_save(img, boxes, img_name_no_ext)

        # 儲存標註圖像
        self._annotate_and_save(img, boxes, img_name_no_ext)

        return cropped_images 