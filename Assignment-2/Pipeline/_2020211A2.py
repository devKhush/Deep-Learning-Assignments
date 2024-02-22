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


'''
Image Dataset Class
'''
class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.type = "image"
        if self.datasplit == "train" or self.datasplit == "val":
            self.transform = transforms.Compose([transforms.ToTensor()])
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
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index:int) -> tuple:
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        return image, label
        

'''
Audio Dataset Class
'''
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



'''
Resnet Block Architecture for Image and Audio Classification
'''
def get_resnet_image_block(in_channel, out_channel, keep_size_same=True):
    if not keep_size_same:
        resnet_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channel)
        )
        skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2),
        )
    else:
        resnet_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channel)
        )
        skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding='same')
        )
    return resnet_block, skip_connection


def get_resnet_audio_block(in_channel, out_channel, keep_size_same=True):
    if not keep_size_same:
        resnet_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channel)
        )
        skip_connection = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=2))
        return resnet_block, skip_connection
    else:
        resnet_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=out_channel)
        )
        skip_connection = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1, padding='same'))
        return resnet_block, skip_connection
    

class Resnet_Q1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # RESNET for Image Classification
        # layers = [3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]
        layers = [3, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64]
        self.im_block1, self.im_skip1 = get_resnet_image_block(layers[0], layers[1])
        self.im_relu1 = nn.ReLU()
        self.im_block2, self.im_skip2 = get_resnet_image_block(layers[1], layers[2])
        self.im_relu2 = nn.ReLU()
        self.im_block3, self.im_skip3 = get_resnet_image_block(layers[2], layers[3])
        self.im_relu3 = nn.ReLU()
        self.im_block4, self.im_skip4 = get_resnet_image_block(layers[3], layers[4], keep_size_same=False)
        self.im_relu4 = nn.ReLU()
        self.im_block5, self.im_skip5 = get_resnet_image_block(layers[4], layers[5])
        self.im_relu5 = nn.ReLU()
        self.im_block6, self.im_skip6 = get_resnet_image_block(layers[5], layers[6])
        self.im_relu6 = nn.ReLU()
        self.im_block7, self.im_skip7 = get_resnet_image_block(layers[6], layers[7])
        self.im_relu7 = nn.ReLU()
        self.im_block8, self.im_skip8 = get_resnet_image_block(layers[7], layers[8])
        self.im_relu8 = nn.ReLU()
        self.im_block9, self.im_skip9 = get_resnet_image_block(layers[8], layers[9], keep_size_same=False)
        self.im_relu9 = nn.ReLU()
        self.im_block10, self.im_skip10 = get_resnet_image_block(layers[9], layers[10])
        self.im_relu10 = nn.ReLU()
        self.im_block11, self.im_skip11 = get_resnet_image_block(layers[10], layers[11])
        self.im_relu11 = nn.ReLU()
        self.im_block12, self.im_skip12 = get_resnet_image_block(layers[11], layers[12])
        self.im_relu12 = nn.ReLU()
        self.im_block13, self.im_skip13 = get_resnet_image_block(layers[12], layers[13])
        self.im_relu13 = nn.ReLU()
        self.im_block14, self.im_skip14 = get_resnet_image_block(layers[13], layers[14], keep_size_same=False)
        self.im_relu14 = nn.ReLU()
        self.im_block15, self.im_skip15 = get_resnet_image_block(layers[14], layers[15])
        self.im_relu15 = nn.ReLU()
        self.im_block16, self.im_skip16 = get_resnet_image_block(layers[15], layers[16])
        self.im_relu16 = nn.ReLU()
        self.im_block17, self.im_skip17 = get_resnet_image_block(layers[16], layers[17])
        self.im_relu17 = nn.ReLU()
        self.im_block18, self.im_skip18 = get_resnet_image_block(layers[17], layers[18])
        self.im_relu18 = nn.ReLU()
        self.im_flatten = nn.Flatten()
        self.im_fc_layer1 = nn.Linear(layers[-1]*4*4, 10)
        # self.im_fc_relu1 = nn.ReLU()
        # self.im_fc_layer2 = nn.Linear(32, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # RESNET for Audio Classification
        # layers = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        # layers = [1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16]
        layers = [1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        self.au_block1, self.au_skip1 = get_resnet_audio_block(layers[0], layers[1], keep_size_same=False)
        self.au_relu1 = nn.ReLU()
        self.au_block2, self.au_skip2 = get_resnet_audio_block(layers[1], layers[2], keep_size_same=False)
        self.au_relu2 = nn.ReLU()
        self.au_block3, self.au_skip3 = get_resnet_audio_block(layers[2], layers[3], keep_size_same=False)
        self.au_relu3 = nn.ReLU()
        self.au_block4, self.au_skip4 = get_resnet_audio_block(layers[3], layers[4], keep_size_same=False)
        self.au_relu4 = nn.ReLU()
        self.au_block5, self.au_skip5 = get_resnet_audio_block(layers[4], layers[5])
        self.au_relu5 = nn.ReLU()
        self.au_block6, self.au_skip6 = get_resnet_audio_block(layers[5], layers[6], keep_size_same=False)
        self.au_relu6 = nn.ReLU()
        self.au_block7, self.au_skip7 = get_resnet_audio_block(layers[6], layers[7])
        self.au_relu7 = nn.ReLU()
        self.au_block8, self.au_skip8 = get_resnet_audio_block(layers[7], layers[8], keep_size_same=False)
        self.au_relu8 = nn.ReLU()
        self.au_block9, self.au_skip9 = get_resnet_audio_block(layers[8], layers[9])
        self.au_relu9 = nn.ReLU()
        self.au_block10, self.au_skip10 = get_resnet_audio_block(layers[9], layers[10])
        self.au_relu10 = nn.ReLU()
        self.au_block11, self.au_skip11 = get_resnet_audio_block(layers[10], layers[11], keep_size_same=False)
        self.au_relu11 = nn.ReLU()
        self.au_block12, self.au_skip12 = get_resnet_audio_block(layers[11], layers[12])
        self.au_relu12 = nn.ReLU()
        self.au_block13, self.au_skip13 = get_resnet_audio_block(layers[12], layers[13])
        self.au_relu13 = nn.ReLU()
        self.au_block14, self.au_skip14 = get_resnet_audio_block(layers[13], layers[14], keep_size_same=False)
        self.au_relu14 = nn.ReLU()
        self.au_block15, self.au_skip15 = get_resnet_audio_block(layers[14], layers[15])
        self.au_relu15 = nn.ReLU()
        self.au_block16, self.au_skip16 = get_resnet_audio_block(layers[15], layers[16], keep_size_same=False)
        self.au_relu16 = nn.ReLU()
        self.au_block17, self.au_skip17 = get_resnet_audio_block(layers[16], layers[17])
        self.au_relu17 = nn.ReLU()
        self.au_block18, self.au_skip18 = get_resnet_audio_block(layers[17], layers[18], keep_size_same=False)
        self.au_relu18 = nn.ReLU()
        self.au_flatten = nn.Flatten()
        self.au_dropout1 = nn.Dropout(0.2) # TODO: Adjust dropout
        self.au_fc_layer1 = nn.Linear(layers[-1]*16, 128)
        self.au_fc_relu1 = nn.ReLU()
        self.au_fc_layer2 = nn.Linear(128, 35)
        self.au_softmax = nn.Softmax(dim=1)
        
    def forward_image(self, x):
        x = self.im_relu1(self.im_block1(x) + self.im_skip1(x))
        x = self.im_relu2(self.im_block2(x) + self.im_skip2(x))
        x = self.im_relu3(self.im_block3(x) + self.im_skip3(x))
        x = self.im_relu4(self.im_block4(x) + self.im_skip4(x))
        x = self.im_relu5(self.im_block5(x) + self.im_skip5(x))
        x = self.im_relu6(self.im_block6(x) + self.im_skip6(x))
        x = self.im_relu7(self.im_block7(x) + self.im_skip7(x))
        x = self.im_relu8(self.im_block8(x) + self.im_skip8(x))
        x = self.im_relu9(self.im_block9(x) + self.im_skip9(x))
        x = self.im_relu10(self.im_block10(x) + self.im_skip10(x))
        x = self.im_relu11(self.im_block11(x) + self.im_skip11(x))
        x = self.im_relu12(self.im_block12(x) + self.im_skip12(x))
        x = self.im_relu13(self.im_block13(x) + self.im_skip13(x))
        x = self.im_relu14(self.im_block14(x) + self.im_skip14(x))
        x = self.im_relu15(self.im_block15(x) + self.im_skip15(x))
        x = self.im_relu16(self.im_block16(x) + self.im_skip16(x))
        x = self.im_relu17(self.im_block17(x) + self.im_skip17(x))
        x = self.im_relu18(self.im_block18(x) + self.im_skip18(x))
        x = self.im_flatten(x)
        x = self.im_fc_layer1(x)
        # x = self.im_fc_relu1(x)
        # x = self.im_fc_layer2(x)
        # x = self.im_softmax(x)
        return x
        
    def forward_audio(self, x):
        x = self.au_relu1(self.au_block1(x) + self.au_skip1(x))
        x = self.au_relu2(self.au_block2(x) + self.au_skip2(x))
        x = self.au_relu3(self.au_block3(x) + self.au_skip3(x))
        x = self.au_relu4(self.au_block4(x) + self.au_skip4(x))
        x = self.au_relu5(self.au_block5(x) + self.au_skip5(x))
        x = self.au_relu6(self.au_block6(x) + self.au_skip6(x))
        x = self.au_relu7(self.au_block7(x) + self.au_skip7(x))
        x = self.au_relu8(self.au_block8(x) + self.au_skip8(x))
        x = self.au_relu9(self.au_block9(x) + self.au_skip9(x))
        x = self.au_relu10(self.au_block10(x) + self.au_skip10(x))
        x = self.au_relu11(self.au_block11(x) + self.au_skip11(x))
        x = self.au_relu12(self.au_block12(x) + self.au_skip12(x))
        x = self.au_relu13(self.au_block13(x) + self.au_skip13(x))
        x = self.au_relu14(self.au_block14(x) + self.au_skip14(x))
        x = self.au_relu15(self.au_block15(x) + self.au_skip15(x))
        x = self.au_relu16(self.au_block16(x) + self.au_skip16(x))
        x = self.au_relu17(self.au_block17(x) + self.au_skip17(x))
        x = self.au_relu18(self.au_block18(x) + self.au_skip18(x))
        x = self.au_flatten(x)
        x = self.au_dropout1(x)
        x = self.au_fc_layer1(x)
        x = self.au_fc_relu1(x)
        x = self.au_fc_layer2(x)
        # x = self.au_softmax(x)
        return x        
    



def get_vgg_image_block(prev_channel, channel, kernel_size, num_conv, padding=0):
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
            nn.MaxPool1d(kernel_size=3)
        )
    elif num_conv == 3:
        return nn.Sequential(
            nn.Conv1d(in_channels=prev_channel, out_channels=channel, kernel_size=kernel_size),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=3)
        )
    return None
     
        
class VGG_Q2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # VGG for Image Classification
        channel, kernel_size = [3, 14], [1]
        for i in range(4):
            channel.append(math.ceil(channel[-1] - 0.35*channel[-1]))
            new_kernel = math.ceil(kernel_size[-1] + 0.25*kernel_size[-1])
            kernel_size.append(new_kernel if new_kernel % 2 == 1 else new_kernel + 1)
        self.im_block1 = get_vgg_image_block(channel[0], channel[1], kernel_size[0], num_conv=2, padding=2)
        self.im_block2 = get_vgg_image_block(channel[1], channel[2], kernel_size[1], num_conv=2, padding=3)
        self.im_block3 = get_vgg_image_block(channel[2], channel[3], kernel_size[2], num_conv=3, padding=3)
        self.im_block4 = get_vgg_image_block(channel[3], channel[4], kernel_size[3], num_conv=3, padding=4)
        self.im_block5 = get_vgg_image_block(channel[4], channel[5], kernel_size[4], num_conv=3, padding=4)
        self.im_flatten = nn.Flatten()
        self.im_fc_layer1 = nn.Linear(channel[-1]*4*4, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # VGG for Audio Classification
        channel, kernel_size = [1, 32], [3] #TODO: [choose channel as 32 OR 64]
        for i in range(4):
            channel.append(math.ceil(channel[-1] - 0.35*channel[-1]))
            kernel_size.append(math.ceil(kernel_size[-1] + 0.25*kernel_size[-1]))
        self.au_block1 = get_vgg_audio_block(channel[0], channel[1], kernel_size[0], 2)
        self.au_block2 = get_vgg_audio_block(channel[1], channel[2], kernel_size[1], 2)
        self.au_block3 = get_vgg_audio_block(channel[2], channel[3], kernel_size[2], 3)
        self.au_block4 = get_vgg_audio_block(channel[3], channel[4], kernel_size[3], 3)
        self.au_block5 = get_vgg_audio_block(channel[4], channel[5], kernel_size[4], 3)
        self.au_flatten = nn.Flatten()
        self.au_fc_layer1 = nn.Linear(channel[-1]*55, 128)
        self.au_relu1 = nn.ReLU()
        self.au_fc_layer2 = nn.Linear(128, 35)
        self.au_softmax = nn.Softmax(dim=1)
        
    
    def forward_image(self, x):
        x = self.im_block1(x)
        x = self.im_block2(x)
        x = self.im_block3(x)
        x = self.im_block4(x)
        x = self.im_block5(x)
        x = self.im_flatten(x)
        # x = self.im_dropout1(x)
        x = self.im_fc_layer1(x)
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
        # x = self.au_softmax(x)
        return x
    


def get_inception_image_block(in_channel, out_channel, keep_size_same=True):
    if keep_size_same:
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
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
        ), nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU()
        ), nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU()
        ), nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        )
    
def get_inception_audio_block(in_channel, out_channel, keep_size_same=True):
    if keep_size_same:
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
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
        ), nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU()
        ), nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU()
        ), nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        )
    
    
class Inception_Q3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Inception for Image Classification
        channel = [3, 8, 16, 16, 32]
        self.im_b1_path1, self.im_b1_path2, self.im_b1_path3, self.im_b1_path4 = get_inception_image_block(channel[0], channel[1], keep_size_same=False)
        self.im_b2_path1, self.im_b2_path2, self.im_b2_path3, self.im_b2_path4 = get_inception_image_block(channel[1], channel[2], keep_size_same=False)
        self.im_b3_path1, self.im_b3_path2, self.im_b3_path3, self.im_b3_path4 = get_inception_image_block(channel[2], channel[3], keep_size_same=False)
        self.im_b4_path1, self.im_b4_path2, self.im_b4_path3, self.im_b4_path4 = get_inception_image_block(channel[3], channel[4], keep_size_same=False)
        self.im_flatten = nn.Flatten()
        self.im_dropout1 = nn.Dropout(0.3)
        self.im_fc_layer1 = nn.Linear(channel[-1]*2*2, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # Inception for Audio Classification
        channel = [1, 2, 4, 4, 8]
        self.au_b1_path1, self.au_b1_path2, self.au_b1_path3, self.au_b1_path4 = get_inception_audio_block(channel[0], channel[1], keep_size_same=False)
        self.au_b2_path1, self.au_b2_path2, self.au_b2_path3, self.au_b2_path4 = get_inception_audio_block(channel[1], channel[2], keep_size_same=False)
        self.au_b3_path1, self.au_b3_path2, self.au_b3_path3, self.au_b3_path4 = get_inception_audio_block(channel[2], channel[3], keep_size_same=False)
        self.au_b4_path1, self.au_b4_path2, self.au_b4_path3, self.au_b4_path4 = get_inception_audio_block(channel[3], channel[4], keep_size_same=False)
        self.au_flatten = nn.Flatten()
        self.au_dropout1 = nn.Dropout(0.1)
        self.au_fc_layer1 = nn.Linear(channel[-1]*1000, 256)
        self.au_relu1 = nn.ReLU()
        self.au_fc_layer2 = nn.Linear(256, 35)
        self.au_softmax = nn.Softmax(dim=1)
    
    def forward_image(self, x):
        x = self.im_b1_path1(x) + self.im_b1_path2(x) + self.im_b1_path3(x) + self.im_b1_path4(x)
        x = self.im_b2_path1(x) + self.im_b2_path2(x) + self.im_b2_path3(x) + self.im_b2_path4(x)
        x = self.im_b3_path1(x) + self.im_b3_path2(x) + self.im_b3_path3(x) + self.im_b3_path4(x)
        x = self.im_b4_path1(x) + self.im_b4_path2(x) + self.im_b4_path3(x) + self.im_b4_path4(x)
        x = self.im_flatten(x)
        x = self.im_dropout1(x)
        x = self.im_fc_layer1(x)
        # x = self.im_softmax(x)
        return x
    
    def forward_audio(self, x):
        x = self.au_b1_path1(x) + self.au_b1_path2(x) + self.au_b1_path3(x) + self.au_b1_path4(x)
        x = self.au_b2_path1(x) + self.au_b2_path2(x) + self.au_b2_path3(x) + self.au_b2_path4(x)
        x = self.au_b3_path1(x) + self.au_b3_path2(x) + self.au_b3_path3(x) + self.au_b3_path4(x)
        x = self.au_b4_path1(x) + self.au_b4_path2(x) + self.au_b4_path3(x) + self.au_b4_path4(x)
        x = self.au_flatten(x)
        x = self.au_dropout1(x)
        x = self.au_fc_layer1(x)
        x = self.au_relu1(x)
        x = self.au_fc_layer2(x)
        # x = self.au_softmax(x)
        return x

        

class CustomNetwork_Q4(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Custom Network for Image Classification
        channels = [3, 128]
        for i in range(9):
            channels.append(math.ceil(channels[-1] - 0.35*channels[-1]))
        self.im_resnet_block1, self.im_resnet_skip1 = get_resnet_image_block(channels[0], channels[1])
        self.im_relu1 = nn.ReLU()
        self.im_resnet_block2, self.im_resnet_skip2 = get_resnet_image_block(channels[1], channels[2], keep_size_same=False)
        self.im_relu2 = nn.ReLU()
        self.im_inception_block1_path1, self.im_inception_block1_path2, self.im_inception_block1_path3, self.im_inception_block1_path4 = get_inception_image_block(channels[2], channels[3])
        self.im_inception_block2_path1, self.im_inception_block2_path2, self.im_inception_block2_path3, self.im_inception_block2_path4 = get_inception_image_block(channels[3], channels[4], keep_size_same=False)
        self.im_resnet_block3, self.im_resnet_skip3 = get_resnet_image_block(channels[4], channels[5])
        self.im_relu3 = nn.ReLU()
        self.im_inception_block3_path1, self.im_inception_block3_path2, self.im_inception_block3_path3, self.im_inception_block3_path4 = get_inception_image_block(channels[5], channels[6])
        self.im_resnet_block4, self.im_resnet_skip4 = get_resnet_image_block(channels[6], channels[7])
        self.im_relu4 = nn.ReLU()
        self.im_inception_block4_path1, self.im_inception_block4_path2, self.im_inception_block4_path3, self.im_inception_block4_path4 = get_inception_image_block(channels[7], channels[8], keep_size_same=False)
        self.im_resnet_block5, self.im_resnet_skip5 = get_resnet_image_block(channels[8], channels[9])
        self.im_relu5 = nn.ReLU()
        self.im_inception_block5_path1, self.im_inception_block5_path2, self.im_inception_block5_path3, self.im_inception_block5_path4 = get_inception_image_block(channels[9], channels[10])
        self.im_flatten = nn.Flatten()
        self.im_fc_layer1 = nn.Linear(channels[-1]*4*4, 10)
        self.im_softmax = nn.Softmax(dim=1)
        
        # Custom Network for Audio Classification
        channels = [1, 16] # Increase channels if low accuracy is observed (64)
        for i in range(9):
            channels.append(math.ceil(channels[-1] - 0.35*channels[-1]))
        self.au_resnet_block1, self.au_resnet_skip1 = get_resnet_audio_block(channels[0], channels[1], keep_size_same=False)
        self.au_relu1 = nn.ReLU()
        self.au_resnet_block2, self.au_resnet_skip2 = get_resnet_audio_block(channels[1], channels[2], keep_size_same=False)
        self.au_relu2 = nn.ReLU()
        self.au_inception_block1_path1, self.au_inception_block1_path2, self.au_inception_block1_path3, self.au_inception_block1_path4 = get_inception_audio_block(channels[2], channels[3], keep_size_same=False)
        self.au_inception_block2_path1, self.au_inception_block2_path2, self.au_inception_block2_path3, self.au_inception_block2_path4 = get_inception_audio_block(channels[3], channels[4], keep_size_same=False)
        self.au_resnet_block3, self.au_resnet_skip3 = get_resnet_audio_block(channels[4], channels[5], keep_size_same=False)
        self.au_relu3 = nn.ReLU()
        self.au_inception_block3_path1, self.au_inception_block3_path2, self.au_inception_block3_path3, self.au_inception_block3_path4 = get_inception_audio_block(channels[5], channels[6], keep_size_same=False)
        self.au_resnet_block4, self.au_resnet_skip4 = get_resnet_audio_block(channels[6], channels[7], keep_size_same=False)
        self.au_relu4 = nn.ReLU()
        self.au_inception_block4_path1, self.au_inception_block4_path2, self.au_inception_block4_path3, self.au_inception_block4_path4 = get_inception_audio_block(channels[7], channels[8], keep_size_same=False)
        self.au_resnet_block5, self.au_resnet_skip5 = get_resnet_audio_block(channels[8], channels[9], keep_size_same=False)
        self.au_relu5 = nn.ReLU()
        self.au_inception_block5_path1, self.au_inception_block5_path2, self.au_inception_block5_path3, self.au_inception_block5_path4 = get_inception_audio_block(channels[9], channels[10])
        self.au_flatten = nn.Flatten()
        self.au_dropout1 = nn.Dropout(0.1) 
        self.au_fc_layer1 = nn.Linear(channels[-1]*32, 64)
        self.au_fc_relu1 = nn.ReLU()
        self.au_fc_layer2 = nn.Linear(64, 35)
        self.au_softmax = nn.Softmax(dim=1)
    
    def forward_image(self, x):
        x = self.im_relu1(self.im_resnet_block1(x) + self.im_resnet_skip1(x))
        x = self.im_relu2(self.im_resnet_block2(x) + self.im_resnet_skip2(x))
        x = self.im_inception_block1_path1(x) + self.im_inception_block1_path2(x) + self.im_inception_block1_path3(x) + self.im_inception_block1_path4(x)
        x = self.im_inception_block2_path1(x) + self.im_inception_block2_path2(x) + self.im_inception_block2_path3(x) + self.im_inception_block2_path4(x)
        x = self.im_relu3(self.im_resnet_block3(x) + self.im_resnet_skip3(x))
        x = self.im_inception_block3_path1(x) + self.im_inception_block3_path2(x) + self.im_inception_block3_path3(x) + self.im_inception_block3_path4(x)
        x = self.im_relu4(self.im_resnet_block4(x) + self.im_resnet_skip4(x))
        x = self.im_inception_block4_path1(x) + self.im_inception_block4_path2(x) + self.im_inception_block4_path3(x) + self.im_inception_block4_path4(x)
        x = self.im_relu5(self.im_resnet_block5(x) + self.im_resnet_skip5(x))
        x = self.im_inception_block5_path1(x) + self.im_inception_block5_path2(x) + self.im_inception_block5_path3(x) + self.im_inception_block5_path4(x)
        x = self.im_flatten(x)
        x = self.im_fc_layer1(x)
        # x = self.im_softmax(x)
        return x
    
    def forward_audio(self, x):
        x = self.au_relu1(self.au_resnet_block1(x) + self.au_resnet_skip1(x))
        x = self.au_relu2(self.au_resnet_block2(x) + self.au_resnet_skip2(x))
        x = self.au_inception_block1_path1(x) + self.au_inception_block1_path2(x) + self.au_inception_block1_path3(x) + self.au_inception_block1_path4(x)
        x = self.au_inception_block2_path1(x) + self.au_inception_block2_path2(x) + self.au_inception_block2_path3(x) + self.au_inception_block2_path4(x)
        x = self.au_relu3(self.au_resnet_block3(x) + self.au_resnet_skip3(x))
        x = self.au_inception_block3_path1(x) + self.au_inception_block3_path2(x) + self.au_inception_block3_path3(x) + self.au_inception_block3_path4(x)
        x = self.au_relu4(self.au_resnet_block4(x) + self.au_resnet_skip4(x))
        x = self.au_inception_block4_path1(x) + self.au_inception_block4_path2(x) + self.au_inception_block4_path3(x) + self.au_inception_block4_path4(x)
        x = self.au_relu5(self.au_resnet_block5(x) + self.au_resnet_skip5(x))
        x = self.au_inception_block5_path1(x) + self.au_inception_block5_path2(x) + self.au_inception_block5_path3(x) + self.au_inception_block5_path4(x)
        x = self.au_dropout1(x)
        x = self.au_flatten(x)
        x = self.au_fc_layer1(x)
        x = self.au_fc_relu1(x)
        x = self.au_fc_layer2(x)
        # x = self.au_softmax(x)
        return x


def trainer(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    network.train()
    best_accuracy = 0
    
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            check_point = {
                "epoch": epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "accuracy": accuracy
            }
            torch.save(check_point, f"checkpoint_{dataloader.dataset.type}.pth")
        if accuracy > 60:
            break        
    return



def validator(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    # Fine Tuning on the validation set
    check_point = torch.load(f"checkpoint_{dataloader.dataset.type}.pth")
    network.load_state_dict(check_point["model_state_dict"])
    optimizer.load_state_dict(check_point["optimizer_state_dict"])
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    network.train()
    best_accuracy = 0
    for epoch in range(EPOCH):
        val_loss = 0.0
        correct = 0
        total = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            if dataloader.dataset.type == "image":
                outputs = network.forward_image(data)
            else:
                outputs = network.forward_audio(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        val_loss = val_loss / len(dataloader)
        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch, val_loss, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            check_point = {
                "epoch": epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "accuracy": accuracy
            }
            torch.save(check_point, f"checkpoint_{dataloader.dataset.type}.pth")
        if accuracy > 55:
            break
    return
    



def evaluator(dataloader=None, network=None, criterion=None, optimizer=None):    
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
