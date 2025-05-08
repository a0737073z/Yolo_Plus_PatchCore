import os
import pickle
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import csv
import time
from utils import *

# 影像補 padding
class ResizeWithPadding:
    def __init__(self, target_size, interpolation=InterpolationMode.LANCZOS):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation)
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top
        img = transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        return img

# 讀取指定資料夾下所有圖片
class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0), img_path, "unknown"

# 主模型
class AnomalyModel(pl.LightningModule):
    def __init__(self, dataset_path, output_path, input_size=512, n_neighbors=9, save_anomaly_map=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.input_size = input_size
        self.n_neighbors = n_neighbors
        self.save_anomaly_map = save_anomaly_map
        #set threshold
        self.manual_threshold = 2.4

        self.embedding_dir_path = os.path.join(self.output_path, "embeddings")
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        print("Loaded embedding from:", self.embedding_dir_path)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        def hook_t(module, input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.mean_train = [0.485, 0.456, 0.406]
        self.std_train = [0.229, 0.224, 0.225]
        self.data_transforms = transforms.Compose([
            ResizeWithPadding(target_size=self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_train, std=self.std_train)
        ])
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )

        self.init_features()
        self.anomaly_map_all = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.input_x_list = []

    def init_features(self):
        self.features = []

    def forward(self, x):
        self.init_features()
        _ = self.model(x)
        return self.features

    def test_dataloader(self):
        test_dataset = SimpleImageFolder(root=self.dataset_path, transform=self.data_transforms)
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def test_step(self, batch, batch_idx):
        start_time = time.time()  # 開始時間記錄

        x, _, file_name, _ = batch
        features = self(x)
        embeddings = [torch.nn.functional.avg_pool2d(f, 3, 1, 1) for f in features]
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=self.n_neighbors)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        h = w = int(np.sqrt(score_patches.shape[0]))
        anomaly_map = score_patches[:, 0].reshape((h, w))
        anomaly_map_resized = cv2.resize(anomaly_map, (self.input_size, self.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        score = np.max(score_patches[:, 0])
        pred_label = 1 if score > self.manual_threshold else 0

        self.anomaly_map_all.append(anomaly_map_resized_blur)
        self.pred_list_img_lvl.append((file_name[0], score, pred_label))

        x = self.inv_normalize(x)
        x = x.permute(0, 2, 3, 1).cpu().numpy()[0]
        x = np.uint8(min_max_norm(x) * 255)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        self.input_x_list.append(x)
        self.img_path_list.append(file_name[0])

        end_time = time.time()  # 結束時間記錄
        inference_time = end_time - start_time
        print(f"Inference time for {os.path.basename(file_name[0])}: {inference_time:.4f} seconds")

    def test_epoch_end(self, outputs):
        os.makedirs(self.output_path, exist_ok=True)
        
        result_csv_path = os.path.join(self.output_path, 'results.csv')
        with open(result_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'pred_score', 'pred_label'])
            for name, score, pred_label in self.pred_list_img_lvl:
                writer.writerow([os.path.basename(name), round(score, 4), pred_label])

        if self.save_anomaly_map:
            save_dir = os.path.join(self.output_path, 'anomaly_maps')
            os.makedirs(save_dir, exist_ok=True)

            maps = np.array(self.anomaly_map_all)
            max_val, min_val = maps.max(), maps.min()
            hi = (max_val - min_val) * 1.0 + min_val
            lo = (max_val - min_val) * 0.6 + min_val

            for i, (amap, orig, name) in enumerate(zip(self.anomaly_map_all, self.input_x_list, self.img_path_list)):
                amap_norm = select_min_max_norm(amap, hi, lo)
                amap_hm = cvt2heatmap(amap_norm * 255)
                amap_on_img = heatmap_on_image(amap_hm, orig)
                cv2.imwrite(os.path.join(save_dir, f'{i}_{os.path.basename(name)}.jpg'), orig)
                cv2.imwrite(os.path.join(save_dir, f'{i}_{os.path.basename(name)}_amap_on_img.jpg'), amap_on_img)

# 執行推論
if __name__ == '__main__':
    dataset_path = r'C:\Users\user\Desktop\0424\patchcore_13_V2\test\1_scratch' 
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

