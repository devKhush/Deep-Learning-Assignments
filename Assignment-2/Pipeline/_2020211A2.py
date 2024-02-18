import math
import torch
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchaudio.datasets import SPEECHCOMMANDS
from torchvision import transforms

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
image_dataset_downloader = CIFAR10(root='./data', train=True, download=True, transform=None)

# Audio Downloader
audio_dataset_downloader = SPEECHCOMMANDS(root='./data', download=True)


class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.type = "image"
        if self.datasplit == "train" or self.datasplit == "val":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_ = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            seed = 42
            train_to_val_split = 0.85
            train_set_size = int(train_to_val_split * len(dataset_))
            val_set_size = len(dataset_) - train_set_size
            train_set, val_set = torch.utils.data.random_split(dataset_, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(seed))
            if self.datasplit == "train":
                self.dataset = train_set
            else:
                self.dataset = val_set
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index:int) -> tuple:
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        return image, label
        

class AudioDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        self.type = "audio"
        if self.datasplit == 'train':
            self.dataset = SPEECHCOMMANDS(root='./data', download=True, subset='training')
        elif self.datasplit == 'test':
            self.dataset = SPEECHCOMMANDS(root='./data', download=True, subset='testing')
        else:
            self.dataset = SPEECHCOMMANDS(root='./data', download=True, subset='validation')
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.dataset)))
        self.class_to_idx = {label: i for i, label in enumerate(self.labels)}
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index:int) -> tuple:
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        # pad the waveform to have the same size
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]), 'constant', value=0)
        return waveform, self.class_to_idx[label]



def get_resnet_image_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm2d(num_features=out_channel)
    )

def get_resnet_audio_block(prev_channel, channel):
    return nn.Sequential(
        nn.Conv1d(in_channels=prev_channel, out_channels=channel, kernel_size=3, padding='same'),
        nn.BatchNorm1d(num_features=channel),
        nn.ReLU(),
        nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=3, padding='same'),
        nn.BatchNorm1d(num_features=channel)
    )

class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # RESNET for Image Classification
        layers = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6]
        # layers = [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 10, 10, 10]
        self.im_block1 = get_resnet_image_block(layers[0], layers[1])
        self.im_relu1 = nn.ReLU()
        self.im_block2 = get_resnet_image_block(layers[1], layers[2])
        self.im_relu2 = nn.ReLU()
        self.im_block3 = get_resnet_image_block(layers[2], layers[3])
        self.im_relu3 = nn.ReLU()
        self.im_block4 = get_resnet_image_block(layers[3], layers[4])
        self.im_relu4 = nn.ReLU()
        self.im_block5 = get_resnet_image_block(layers[4], layers[5])
        self.im_relu5 = nn.ReLU()
        self.im_block6 = get_resnet_image_block(layers[5], layers[6])
        self.im_relu6 = nn.ReLU()
        self.im_block7 = get_resnet_image_block(layers[6], layers[7])
        self.im_relu7 = nn.ReLU()
        self.im_block8 = get_resnet_image_block(layers[7], layers[8])
        self.im_relu8 = nn.ReLU()
        self.im_block9 = get_resnet_image_block(layers[8], layers[9])
        self.im_relu9 = nn.ReLU()
        self.im_block10 = get_resnet_image_block(layers[9], layers[10])
        self.im_relu10 = nn.ReLU()
        self.im_block11 = get_resnet_image_block(layers[10], layers[11])
        self.im_relu11 = nn.ReLU()
        self.im_block12 = get_resnet_image_block(layers[11], layers[12])
        self.im_relu12 = nn.ReLU()
        self.im_block13 = get_resnet_image_block(layers[12], layers[13])
        self.im_relu13 = nn.ReLU()
        self.im_block14 = get_resnet_image_block(layers[13], layers[14])
        self.im_relu14 = nn.ReLU()
        self.im_block15 = get_resnet_image_block(layers[14], layers[15])
        self.im_relu15 = nn.ReLU()
        self.im_block16 = get_resnet_image_block(layers[15], layers[16])
        self.im_relu16 = nn.ReLU()
        self.im_block17 = get_resnet_image_block(layers[16], layers[17])
        self.im_relu17 = nn.ReLU()
        self.im_block18 = get_resnet_image_block(layers[17], layers[18])
        self.im_relu18 = nn.ReLU()
        self.im_flatten = nn.Flatten()
        self.im_dropout1 = nn.Dropout(0.35)
        self.im_fc_layer1 = nn.Linear(layers[-1]*32*32, 2048)
        self.im_relu1 = nn.ReLU()
        self.im_dropout2 = nn.Dropout(0.2)
        self.im_fc_layer2 = nn.Linear(2048, 512)
        self.im_relu2 = nn.ReLU()
        self.im_fc_layer3 = nn.Linear(512, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # RESNET for Audio Classification
        layers = [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        self.au_block1 = get_resnet_audio_block(layers[0], layers[1])
        self.au_relu1 = nn.ReLU()
        self.au_block2 = get_resnet_audio_block(layers[1], layers[2])
        self.au_relu2 = nn.ReLU()
        self.au_block3 = get_resnet_audio_block(layers[2], layers[3])
        self.au_relu3 = nn.ReLU()
        self.au_block4 = get_resnet_audio_block(layers[3], layers[4])        
        self.au_relu4 = nn.ReLU()
        self.au_block5 = get_resnet_audio_block(layers[4], layers[5])
        self.au_relu5 = nn.ReLU()
        self.au_block6 = get_resnet_audio_block(layers[5], layers[6])
        self.au_relu6 = nn.ReLU()
        self.au_block7 = get_resnet_audio_block(layers[6], layers[7])
        self.au_relu7 = nn.ReLU()
        self.au_block8 = get_resnet_audio_block(layers[7], layers[8])
        self.au_relu8 = nn.ReLU()
        self.au_block9 = get_resnet_audio_block(layers[8], layers[9])
        self.au_relu9 = nn.ReLU()
        self.au_block10 = get_resnet_audio_block(layers[9], layers[10])
        self.au_relu10 = nn.ReLU()
        self.au_block11 = get_resnet_audio_block(layers[10], layers[11])
        self.au_relu11 = nn.ReLU()
        self.au_block12 = get_resnet_audio_block(layers[11], layers[12])
        self.au_relu12 = nn.ReLU()
        self.au_block13 = get_resnet_audio_block(layers[12], layers[13])
        self.au_relu13 = nn.ReLU()
        self.au_block14 = get_resnet_audio_block(layers[13], layers[14])
        self.au_relu14 = nn.ReLU()
        self.au_block15 = get_resnet_audio_block(layers[14], layers[15])
        self.au_relu15 = nn.ReLU()
        self.au_block16 = get_resnet_audio_block(layers[15], layers[16])
        self.au_relu16 = nn.ReLU()
        self.au_block17 = get_resnet_audio_block(layers[16], layers[17])
        self.au_relu17 = nn.ReLU()
        self.au_block18 = get_resnet_audio_block(layers[17], layers[18])
        self.au_relu18 = nn.ReLU()
        self.au_flatten = nn.Flatten()
        self.au_dropout1 = nn.Dropout(0.4)
        self.au_fc_layer1 = nn.Linear(layers[-1]*16000, 1024)
        self.au_relu1 = nn.ReLU()
        self.au_dropout2 = nn.Dropout(0.3)
        self.au_fc_layer2 = nn.Linear(1024, 35)
        self.au_softmax = nn.Softmax(dim=1)
    
    def resnet_block_out(self, block, relu, x):
        block_out = block(x)
        signal_out = torch.cat([x for _ in range(block_out.shape[1]//x.shape[1] + 1)], dim=1)
        if len(x.shape) == 4:
            signal_out = signal_out[:, :block_out.shape[1], :, :]
        else: # shape is 3 for audio
            signal_out = signal_out[:, :block_out.shape[1], :]
        out = block_out + signal_out
        return relu(out)
        
    def forward_image(self, x):
        x = self.resnet_block_out(self.im_block1, self.im_relu1, x)
        x = self.resnet_block_out(self.im_block2, self.im_relu2, x)
        x = self.resnet_block_out(self.im_block3, self.im_relu3, x)
        x = self.resnet_block_out(self.im_block4, self.im_relu4, x)
        x = self.resnet_block_out(self.im_block5, self.im_relu5, x)
        x = self.resnet_block_out(self.im_block6, self.im_relu6, x)
        x = self.resnet_block_out(self.im_block7, self.im_relu7, x)
        x = self.resnet_block_out(self.im_block8, self.im_relu8, x)
        x = self.resnet_block_out(self.im_block9, self.im_relu9, x)
        x = self.resnet_block_out(self.im_block10, self.im_relu10, x)
        x = self.resnet_block_out(self.im_block11, self.im_relu11, x)
        x = self.resnet_block_out(self.im_block12, self.im_relu12, x)
        x = self.resnet_block_out(self.im_block13, self.im_relu13, x)
        x = self.resnet_block_out(self.im_block14, self.im_relu14, x)
        x = self.resnet_block_out(self.im_block15, self.im_relu15, x)
        x = self.resnet_block_out(self.im_block16, self.im_relu16, x)
        x = self.resnet_block_out(self.im_block17, self.im_relu17, x)
        x = self.resnet_block_out(self.im_block18, self.im_relu18, x)
        x = self.im_flatten(x)
        x = self.im_dropout1(x)
        x = self.im_fc_layer1(x)
        x = self.im_relu1(x)
        x = self.im_dropout2(x)
        x = self.im_fc_layer2(x)
        # x = self.im_softmax(x)
        return x
        
    def forward_audio(self, x):
        x = self.resnet_block_out(self.au_block1, self.au_relu1, x)
        x = self.resnet_block_out(self.au_block2, self.au_relu2, x)
        x = self.resnet_block_out(self.au_block3, self.au_relu3, x)
        x = self.resnet_block_out(self.au_block4, self.au_relu4, x)
        x = self.resnet_block_out(self.au_block5, self.au_relu5, x)
        x = self.resnet_block_out(self.au_block6, self.au_relu6, x)
        x = self.resnet_block_out(self.au_block7, self.au_relu7, x)
        x = self.resnet_block_out(self.au_block8, self.au_relu8, x)
        x = self.resnet_block_out(self.au_block9, self.au_relu9, x)
        x = self.resnet_block_out(self.au_block10, self.au_relu10, x)
        x = self.resnet_block_out(self.au_block11, self.au_relu11, x)
        x = self.resnet_block_out(self.au_block12, self.au_relu12, x)
        x = self.resnet_block_out(self.au_block13, self.au_relu13, x)
        x = self.resnet_block_out(self.au_block14, self.au_relu14, x)
        x = self.resnet_block_out(self.au_block15, self.au_relu15, x)
        x = self.resnet_block_out(self.au_block16, self.au_relu16, x)
        x = self.resnet_block_out(self.au_block17, self.au_relu17, x)
        x = self.resnet_block_out(self.au_block18, self.au_relu18, x)
        x = self.au_flatten(x)
        x = self.au_dropout1(x)
        x = self.au_fc_layer1(x)
        x = self.au_relu1(x)
        x = self.au_dropout2(x)
        x = self.au_fc_layer2(x)
        # x = self.au_softmax(x)
        return x
        


def get_vgg_image_block(prev_channel, channel, kernel_size, padding, num_conv):
    if num_conv == 2:
        return nn.Sequential(
            nn.Conv2d(in_channels=prev_channel, out_channels=channel, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(kernel_size=2)
        )
    elif num_conv == 3:
        return nn.Sequential(
            nn.Conv2d(in_channels=prev_channel, out_channels=channel, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(kernel_size=2)
        )
    return None

def get_vgg_audio_block(prev_channel, channel, kernel_size, num_conv):
    if num_conv == 2:
        return nn.Sequential(
            nn.Conv1d(in_channels=prev_channel, out_channels=channel, kernel_size=kernel_size),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2)
        )
    elif num_conv == 3:
        return nn.Sequential(
            nn.Conv1d(in_channels=prev_channel, out_channels=channel, kernel_size=kernel_size),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2)
        )
    return None
     
        
class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # VGG for Image Classification
        channel, kernel_size, padding = [3, 128], [2], [2]
        for i in range(4):
            channel.append(math.ceil(channel[-1] - 0.35*channel[-1]))
            kernel_size.append(math.ceil(kernel_size[-1] + 0.25*kernel_size[-1]))
            padding.append(math.ceil(padding[-1] + 0.5*padding[-1]))
        self.im_block1 = get_vgg_image_block(channel[0], channel[1], padding[0], kernel_size[0], 2)
        self.im_block2 = get_vgg_image_block(channel[1], channel[2], padding[1], kernel_size[1], 2)
        self.im_block3 = get_vgg_image_block(channel[2], channel[3], padding[2], kernel_size[2], 3)
        self.im_block4 = get_vgg_image_block(channel[3], channel[4], padding[3], kernel_size[3], 3)
        self.im_block5 = get_vgg_image_block(channel[4], channel[5], padding[4], kernel_size[4], 3)
        self.im_flatten = nn.Flatten()
        self.im_fc_layer1 = nn.Linear(channel[-1]*9*9, 128)
        self.im_relu1 = nn.ReLU()
        self.im_fc_layer2 = nn.Linear(128, 10)
        self.im_softmax = nn.Softmax(dim=1)
        print(channel, kernel_size, padding)
        
        # VGG for Audio Classification
        channel, kernel_size = [1, 128], [2]
        for i in range(4):
            channel.append(math.ceil(channel[-1] - 0.35*channel[-1]))
            kernel_size.append(math.ceil(kernel_size[-1] + 0.25*kernel_size[-1]))
        self.au_block1 = get_vgg_audio_block(channel[0], channel[1], kernel_size[0], 2)
        self.au_block2 = get_vgg_audio_block(channel[1], channel[2], kernel_size[1], 2)
        self.au_block3 = get_vgg_audio_block(channel[2], channel[3], kernel_size[2], 3)
        self.au_block4 = get_vgg_audio_block(channel[3], channel[4], kernel_size[3], 3)
        self.au_block5 = get_vgg_audio_block(channel[4], channel[5], kernel_size[4], 3)
        self.au_flatten = nn.Flatten()
        self.au_fc_layer1 = nn.Linear(channel[-1]*486, 2048)
        self.au_relu1 = nn.ReLU()
        self.au_fc_layer2 = nn.Linear(2048, 512)
        self.au_relu2 = nn.ReLU()
        self.au_fc_layer3 = nn.Linear(512, 35)
        self.au_softmax = nn.Softmax(dim=1)
        print(channel, kernel_size)
        
    
    def forward_image(self, x):
        x = self.im_block1(x)
        x = self.im_block2(x)
        x = self.im_block3(x)
        x = self.im_block4(x)
        x = self.im_block5(x)
        x = self.im_flatten(x)
        x = self.im_fc_layer1(x)
        x = self.im_relu1(x)
        x = self.im_fc_layer2(x)
        # x = self.im_softmax(x)
        return x
    
    def forward_audio(self, x):
        x = self.au_block1(x)
        x = self.au_block2(x)
        x = self.au_block3(x)
        x = self.au_block4(x)
        x = self.au_block5(x)
        x = self.au_flatten(x)
        x = self.au_fc_layer1(x)
        x = self.au_relu1(x)
        x = self.au_fc_layer2(x)
        x = self.au_relu2(x)
        x = self.au_fc_layer3(x)
        # x = self.au_softmax(x)
        return x
    
        

def get_inception_image_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(),
    ), nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same'),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU()
    ), nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same'),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU()
    ), nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    )
    
def get_inception_audio_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),
        nn.BatchNorm1d(num_features=out_channel),
        nn.ReLU(),
    ), nn.Sequential(
        nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm1d(num_features=out_channel),
        nn.ReLU(),
        nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same'),
        nn.BatchNorm1d(num_features=out_channel),
        nn.ReLU()
    ), nn.Sequential(
        nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
        nn.BatchNorm1d(num_features=out_channel),
        nn.ReLU(),
        nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same'),
        nn.BatchNorm1d(num_features=out_channel),
        nn.ReLU()
    ), nn.Sequential(
        nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
    )
    
    
class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Inception for Image Classification
        channel = [3, 8, 16, 32, 64]
        self.im_b1_path1, self.im_b1_path2, self.im_b1_path3, self.im_b1_path4 = get_inception_image_block(channel[0], channel[1])
        self.im_b2_path1, self.im_b2_path2, self.im_b2_path3, self.im_b2_path4 = get_inception_image_block(channel[1], channel[2])
        self.im_b3_path1, self.im_b3_path2, self.im_b3_path3, self.im_b3_path4 = get_inception_image_block(channel[2], channel[3])
        self.im_b4_path1, self.im_b4_path2, self.im_b4_path3, self.im_b4_path4 = get_inception_image_block(channel[3], channel[4])
        self.im_flatten = nn.Flatten()
        self.im_fc_layer1 = nn.Linear(channel[-1]*32*32, 1024)
        self.im_relu1 = nn.ReLU()
        self.im_fc_layer2 = nn.Linear(1024, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # Inception for Audio Classification
        channel = [1, 3, 5, 7, 9]
        self.au_b1_path1, self.au_b1_path2, self.au_b1_path3, self.au_b1_path4 = get_inception_audio_block(channel[0], channel[1])
        self.au_b2_path1, self.au_b2_path2, self.au_b2_path3, self.au_b2_path4 = get_inception_audio_block(channel[1], channel[2])
        self.au_b3_path1, self.au_b3_path2, self.au_b3_path3, self.au_b3_path4 = get_inception_audio_block(channel[2], channel[3])
        self.au_b4_path1, self.au_b4_path2, self.au_b4_path3, self.au_b4_path4 = get_inception_audio_block(channel[3], channel[4])
        self.au_flatten = nn.Flatten()
        self.au_dropout1 = nn.Dropout(0.35)
        self.au_fc_layer1 = nn.Linear(channel[-1]*16000, 1024)
        self.au_relu1 = nn.ReLU()
        self.au_dropout2 = nn.Dropout(0.2)
        self.au_fc_layer2 = nn.Linear(1024, 35)
        self.au_softmax = nn.Softmax(dim=1)
        
    def inception_block_out(self, p1, p2, p3, p4, x):
        p1_out, p2_out, p3_out, p4_out = p1(x), p2(x), p3(x), p4(x)
        p4_out = torch.cat([p4_out for _ in range(p1_out.shape[1]//p4_out.shape[1] + 1)], dim=1)
        if len(p4_out.shape) == 4:
            p4_out = p4_out[:, :p1_out.shape[1], :, :]
        else:
            p4_out = p4_out[:, :p1_out.shape[1], :]
        return p1_out + p2_out + p3_out + p4_out
    
    def forward_image(self, x):
        x = self.inception_block_out(self.im_b1_path1, self.im_b1_path2, self.im_b1_path3, self.im_b1_path4, x)
        x = self.inception_block_out(self.im_b2_path1, self.im_b2_path2, self.im_b2_path3, self.im_b2_path4, x)
        x = self.inception_block_out(self.im_b3_path1, self.im_b3_path2, self.im_b3_path3, self.im_b3_path4, x)
        x = self.inception_block_out(self.im_b4_path1, self.im_b4_path2, self.im_b4_path3, self.im_b4_path4, x)
        x = self.im_flatten(x)
        x = self.im_fc_layer1(x)
        x = self.im_relu1(x)
        x = self.im_fc_layer2(x)
        # x = self.im_softmax(x)
        return x
    
    def forward_audio(self, x):
        x = self.inception_block_out(self.au_b1_path1, self.au_b1_path2, self.au_b1_path3, self.au_b1_path4, x)
        x = self.inception_block_out(self.au_b2_path1, self.au_b2_path2, self.au_b2_path3, self.au_b2_path4, x)
        x = self.inception_block_out(self.au_b3_path1, self.au_b3_path2, self.au_b3_path3, self.au_b3_path4, x)
        x = self.inception_block_out(self.au_b4_path1, self.au_b4_path2, self.au_b4_path3, self.au_b4_path4, x)
        x = self.au_flatten(x)
        x = self.au_dropout1(x)
        x = self.au_fc_layer1(x)
        x = self.au_relu1(x)
        x = self.au_dropout2(x)
        x = self.au_fc_layer2(x)
        # x = self.au_softmax(x)
        return x
        

class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """


def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    network.train()
    for epoch in range(EPOCH):
        train_loss = 0.0
        correct = 0
        total = 0
        # batch = 0
        # print('Batch', end=" ")
        for data, labels in dataloader:
            # print(batch, end=" ")
            # batch += 1
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            if dataloader.dataset.type == "image":
                outputs = network.forward_image(data)
            else:
                outputs = network.forward_audio(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        train_loss = train_loss / len(dataloader) 
        # print()
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch, train_loss, accuracy))
        
    check_point = {
        "epoch": epoch,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "accuracy": accuracy
    }
    torch.save(check_point, f"checkpoint_{dataloader.dataset.type}.pth")
    return



def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    network.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            if dataloader.dataset.type == "image":
                outputs = network.forward_image(data)
            else:
                outputs = network.forward_audio(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    val_loss = val_loss / len(dataloader)
    print("Validation: [Loss: {}, Accuracy: {}]".format(val_loss, accuracy))
    return



def evaluator(dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):    
    check_point = torch.load(f"checkpoint_{dataloader.dataset.type}.pth")
    network.load_state_dict(check_point["model_state_dict"])
    device = torch.device("cuda:0") if  torch.cuda.is_available() else torch.device("cpu")
    network = network.to(device)
    network.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            if dataloader.dataset.type == "image":
                outputs = network.forward_image(data)
            else:
                outputs = network.forward_audio(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss = test_loss / len(dataloader)
    print("Testing: [Loss: {}, Accuracy: {}]".format(test_loss, accuracy))
    return
