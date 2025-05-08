from yolo_test import ObjectDetection
from patchcore_test_alldata import *


def main():
    #Yolo
    yolo_model_path = r"C:\Users\user\Desktop\project\code\yolo\runs\train\metal_yolo\weights\best.pt"
    result_path = r"C:\Users\user\Desktop\0424\yolo_plus_patchcore_result"
    img_path = r"C:\Users\user\Desktop\0424\yolo\val\images\20.png"

    detector = ObjectDetection(yolo_model_path=yolo_model_path,output_dir=result_path)

    cropped_images = detector.predict_result(img_path)

    print("裁剪圖像已儲存:")
    for cropped_img in cropped_images:
        print(cropped_img)

    #PatchCore
    dataset_path = result_path
    output_path = r'C:\Users\user\Desktop\0424\patchcore_result\12_1024_0.05'
    checkpoint_path = os.path.join(output_path, 'lightning_logs', 'version_0', 'checkpoints', 'epoch=0-step=12.ckpt')

    model = AnomalyModel(
        dataset_path=dataset_path,
        output_path=output_path,
        input_size=1024,
        n_neighbors=9,
        save_anomaly_map=True
    )
    model = AnomalyModel.load_from_checkpoint(checkpoint_path, dataset_path=dataset_path, output_path=output_path)

    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
    trainer.test(model)


if __name__ == "__main__":
    main()
