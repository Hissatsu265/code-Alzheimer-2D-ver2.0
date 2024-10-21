from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from src_v2.code_v2.model import MRIModel
from pytorch_lightning.loggers import WandbLogger

data_dir = '/home/jupyter-iec_iot13_toanlm/data/data6'

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
wandb_logger = WandbLogger(project="code 2D Alzheimer ver2.0")
# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)
# print(dataset.shape)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print("-------------------")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, verbose=True)

# Khởi tạo mô hình
model = MRIModel()

# trainer = Trainer(max_epochs=25, gpus=1 if torch.cuda.is_available() else 0, callbacks=[checkpoint_callback])
trainer = Trainer(
    logger=wandb_logger,

    max_epochs=25, 
    accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
    devices=1 if torch.cuda.is_available() else None, 
    callbacks=[checkpoint_callback]
)
# Bắt đầu huấn luyện
trainer.fit(model, train_loader, val_loader)

# Kiểm tra trên tập test
trainer.test(model, test_loader)
