import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch


class CUB2011Dataset(Dataset):
    def __init__(self, root_dir, train=True, image_size=224):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')

        # Load metadata
        images_txt = self._load_txt(os.path.join(root_dir, 'images.txt'))  # id, relative_path
        labels_txt = self._load_txt(os.path.join(root_dir, 'image_class_labels.txt'))  # id, class_label
        split_txt = self._load_txt(os.path.join(root_dir, 'train_test_split.txt'))  # id, is_train

        # id to image path
        id_to_img = {int(row[0]): row[1] for row in images_txt}
        id_to_label = {int(row[0]): int(row[1]) - 1 for row in labels_txt}  # 0-based label
        id_to_split = {int(row[0]): int(row[1]) for row in split_txt}

        self.samples = []
        for img_id in id_to_img:
            if (train and id_to_split[img_id] == 1) or (not train and id_to_split[img_id] == 0):
                path = os.path.join(self.image_dir, id_to_img[img_id])
                label = id_to_label[img_id]
                self.samples.append((path, label))

        # ViT-compatible transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])   # ImageNet std
        ])

    def _load_txt(self, filepath):
        with open(filepath, 'r') as f:
            return [line.strip().split() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

def show_image(img_tensor, title=None):
    # Unnormalize the image (reverse ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean  # unnormalize

    # Convert to numpy and transpose to HWC
    np_img = img.numpy().transpose(1, 2, 0)

    # Clip values to [0, 1] in case of minor float errors
    np_img = np.clip(np_img, 0, 1)

    # Show image
    plt.imshow(np_img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    root = 'raw_data/CUB_200_2011/CUB_200_2011'
    train_dataset = CUB2011Dataset(root_dir=root, train=True)
    test_dataset = CUB2011Dataset(root_dir=root, train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Sample check
    for images, labels in train_loader:
        print("Batch images shape:", images.shape)  # [B, 3, 224, 224]
        print("Batch labels shape:", labels.shape)  # [B]
        break