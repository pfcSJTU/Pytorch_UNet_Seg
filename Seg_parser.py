import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import argparse

SUPPORTED_IMAGE_FORMATS = {
    'jpg': 'JPEG',
    'png': 'PNG' ,
    'bmp': 'BMP'
}

# Define the UNet architecture
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


# Define a dataset class for loading images and masks
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = []
        for filename in os.listdir(image_dir):
            for suffix in list(SUPPORTED_IMAGE_FORMATS.keys()):
                if filename.endswith("."+str(suffix)):
                    self.image_filenames.append(filename)
        self.mask_filenames = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_filename = self.image_filenames[idx]
        mask_filename = self.mask_filenames[idx]
        image = Image.open(os.path.join(self.image_dir, image_filename))
        mask = Image.open(os.path.join(self.mask_dir, mask_filename))

        # Apply transforms, if any
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert to tensors
        # image = transforms.functional.to_tensor(image)
        # mask = transforms.functional.to_tensor(mask)

        return image, mask

def train(args):
    # Define the dataset and dataloader
    dataset = SegmentationDataset(image_dir=args.image_dir,
                                  mask_dir=args.mask_dir,
                                  transform=transforms.Compose([transforms.Resize((args.resize_height, args.resize_width)),
                                                                transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Instantiate a UNet model and move it to the device
    model = UNet().to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

    # Train the model for a specified number of epochs
    best_loss = float('inf')
    save_dir = args.save_dir
    best_model_path = None

    for epoch in range(args.num_epochs):
        running_loss = 0.0

        # Create a tqdm progress bar to show the progress of each epoch
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            start_time = time.time()
            for i, (inputs, masks) in enumerate(dataloader):
                # Move the inputs and masks to the device
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

                # Update the progress bar with the current progress of the epoch
                pbar.set_postfix({'loss': running_loss / (i+1)})
                pbar.update(1)

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(dataloader)}, Time: {epoch_time:.2f}s")

        # Save the model if the loss has improved
        average_loss = running_loss / len(dataloader)

        if average_loss < best_loss:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_loss = average_loss
            best_model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), best_model_path)

    print(f'Training complete. Best model saved at {best_model_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a UNet model for image segmentation')
    parser.add_argument('--image_dir', type=str, required=True, help='directory containing the images')
    parser.add_argument('--mask_dir', type=str, required=True, help='directory containing the masks')
    parser.add_argument('--resize_height', type=int, default=224, help='height to resize the images and masks to')
    parser.add_argument('--resize_width', type=int, default=224, help='width to resize the images and masks to')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs to train for')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the trained model')

    args = parser.parse_args()

    train(args)


