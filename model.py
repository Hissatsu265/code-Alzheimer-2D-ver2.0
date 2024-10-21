import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import wandb


class MRIModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=0.001):
        super(MRIModel, self).__init__()
        self.save_hyperparameters()

        # Dùng ResNet18 pre-trained
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Xóa lớp fully connected cuối cùng

        # Thêm các lớp CNN khác để tăng độ phức tạp
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(128 * 7 * 7, 512),  
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes) 
        # )

        self.fc = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    # def forward(self, x):
    #     x = self.resnet(x)
    #     x = x.unsqueeze(2).unsqueeze(3)  # Thêm chiều để CNN xử lý
    #     x = self.cnn(x)
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = self.fc(x)
    #     return x
    def forward(self, x):
        print(f"Input size before ResNet: {x.size()}")
        x = self.resnet(x)
        print(f"Size after ResNet: {x.size()}")  \
        # x = self.cnn(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)        
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        # wandb.log({'train_loss': loss, 'train_acc': acc})
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        # wandb.log({'val_loss': loss, 'val_acc': acc})
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        # wandb.log({'test_loss': loss, 'test_acc': acc})
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]
