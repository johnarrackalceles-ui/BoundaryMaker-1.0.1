import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import segmentation_models_pytorch as smp

# --- 1️⃣ Paths ---
TRAIN_IMAGES_DIR = "data/images/train" #IMPUT YOUR PATH TO FOLDERS [JUST FOR REFERENCE] 
TRAIN_MASKS_DIR  = "data/masks/train_bin"
PRETRAINED_MODEL_PATH = "unet_epoch50.pth"  # will not work in Mac. have to paste absolute path
FINE_TUNE_SAVE_PATH = "unet_finetuned.pth"

# --- 2️⃣ Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# --- 3️⃣ Data Augmentation & DataLoader ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
train_dataset = SegmentationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# --- 4️⃣ Load pretrained UNet ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained UNet from your previous training
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',  # loads ImageNet weights
    in_channels=3,
    classes=1
).to(device)

# Load your previously trained UNet weights
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device), strict=False)
print("Pretrained model loaded successfully.")

# Optionally freeze encoder for small dataset
for name, param in model.encoder.named_parameters():
    param.requires_grad = False

for name, param in model.encoder.layer4.named_parameters():
    param.requires_grad = True

# --- 5️⃣ Loss & optimizer ---
pos_weight = torch.tensor([20.0]).to(device)  # emphasizes sparse T-cells
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
print("Learning rate:", optimizer.param_groups[0]['lr'])

# --- 6️⃣ Fine-tuning Loop ---
num_epochs = 30  # small dataset, fewer epochs
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Print min/max of first batch
        outputs_sigmoid = torch.sigmoid(outputs)
        print(f"Batch output min: {outputs_sigmoid.min().item():.4f}, max: {outputs_sigmoid.max().item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# --- 7️⃣ Save fine-tuned model ---
torch.save(model.state_dict(), FINE_TUNE_SAVE_PATH)
print(f"Fine-tuning complete. Model saved to {FINE_TUNE_SAVE_PATH}")

# --- 8️⃣ Plot loss ---
plt.plot(range(1, num_epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss over Epochs")
plt.show()
