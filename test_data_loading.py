"""
Dataset Visualization Script - Thermal Image Classification
Author: Turgut Sofuyev
"""

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import torch

# ===============================
# 1. Dataset və Transform
# ===============================
data_path = "dataset/datasets/thermal_classification_cropped"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset-i oxu
dataset = datasets.ImageFolder(root=data_path, transform=transform)

print(f"Total number of images in the dataset: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# ===============================
# 2. Hər sinifdən 4 təsadüfi şəkil seç
# ===============================
class_names = dataset.classes
class_indices = {cls: [] for cls in class_names}

# Hər sinifin indekslərini topla
for idx, (_, label) in enumerate(dataset.samples):
    class_name = class_names[label]
    class_indices[class_name].append(idx)

# Hər sinifdən 4 şəkil təsadüfi seç
samples_to_show = []
for cls in class_names:
    samples_to_show.extend(random.sample(class_indices[cls], 4))

# ===============================
# 3. Şəkilləri göstər
# ===============================
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("Thermal Classification Dataset Overview", fontsize=16, weight='bold')

for i, idx in enumerate(samples_to_show):
    img, label = dataset[idx]
    ax = axes[i // 4, i % 4]
    ax.imshow(torch.permute(img, (1, 2, 0)))
    ax.set_title(class_names[label], fontsize=12)
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("dataset_preview_v2.png", dpi=150)
plt.show()
