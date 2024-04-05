import os
import torch
import random
import argparse
from EncDec import *
from EncDec._2020211A3 import *
from torch.utils.data import DataLoader

P = argparse.ArgumentParser()
P.add_argument("gpu", type=str)
A = P.parse_args()

Data = DataLoader(dataset=AlteredMNIST(train_test=True, train=True),
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=2,
                      drop_last=True,
                      pin_memory=True)

testData = DataLoader(dataset=AlteredMNIST(train_test=True, train=False),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=2,
                        drop_last=True,
                        pin_memory=True)

E = Encoder()
D = Decoder()
L = [AELossFn()]
O = torch.optim.Adam(ParameterSelector(E, D), lr=LEARNING_RATE)
print("Training Encoder: {}, Decoder: {} on Modified MNIST dataset in AE training paradigm".format(
        E.__class__.__name__,
        D.__class__.__name__,
    ))

AETrainer(Data, E, D, L[0], O, A.gpu)

AETester(testData, E, D, L[0], A.gpu)

AE_Pipeline = AE_TRAINED(gpu=False)