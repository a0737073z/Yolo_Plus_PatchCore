from yolo_test import ObjectDetection

def main():
    #Yolo
    yolo_model_path = r"C:\Users\user\Desktop\project\code\yolo\runs\train\metal_yolo\weights\best.pt"
    img_path = r"C:\Users\user\Desktop\0424\yolo\val\images\20.png"

    detector = ObjectDetection(yolo_model_path=yolo_model_path)

    cropped_images = detector.predict_result(img_path)

    print("裁剪圖像已儲存:")
    for cropped_img in cropped_images:
        print(cropped_img)

    #PatchCore
    

if __name__ == "__main__":
    main()
