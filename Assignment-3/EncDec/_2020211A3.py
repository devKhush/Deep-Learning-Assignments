import os
import torch
import random 
from EncDec import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity


class AlteredMNIST(ImageFolder):
    def __init__(self, train_test=False, train=True) -> None:
        # Transformations for Augmented Images
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self.transform2 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        super().__init__(root='./Data', transform=self.transform2)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Separating augmented and clean images
        self.aug_images = []
        self.clean_images = []
        for i in range(len(self.samples)):
            img, label = self.samples[i]
            if self.idx_to_class[label] == 'aug':
                self.aug_images.append(img)
            else:
                self.clean_images.append(img)
              
        # Prepare many to many samples mapping for augmented to clean images  
        self.label_wise_clean_images = {i:[] for i in range(10)}
        for image_path in self.clean_images:
            label = int(image_path[-5])
            self.label_wise_clean_images[label].append(image_path)

        # Choose 40 clean images for each augmented image
        self.samples = []
        for image_path in self.aug_images:
            label = int(image_path[-5])
            self.samples.append((image_path, random.sample(self.label_wise_clean_images[label], 40), label))
        random.shuffle(self.samples)
        
        # Train-Test split for testing purpose
        self.train_test = train_test
        self.train = train
        if train_test:
            # Train-Test split: 85-15 
            self.test_samples = self.samples[-int(0.15*len(self.samples)):]
            self.samples = self.samples[:-int(0.15*len(self.samples))]
        
    def __len__(self):
        if self.train_test:
            if self.train == True:
                return len(self.samples)
            elif self.train == False:
                return len(self.test_samples)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.train_test:
            if self.train == True:
                noise_image_path, clean_image_paths, label = self.samples[idx]
            elif self.train == False:
                noise_image_path, clean_image_paths, label = self.test_samples[idx]
        else:
            noise_image_path, clean_image_paths, label = self.samples[idx]
        
        # Load images and apply transformations      
        # Pick the image with least difference from noise image as clean image      
        noise_image = self.transform(self.loader(noise_image_path))
        clean_images = [self.transform2(self.loader(image_path)) for image_path in clean_image_paths]
        if (not self.train_test) or (self.train_test and self.train):
            clean_to_noise_diff = [torch.sum((noise_image - clean_image)**2) for clean_image in clean_images]
            min_index = clean_to_noise_diff.index(min(clean_to_noise_diff))
            clean_image = clean_images[min_index]
        else:
            # clean_to_noise_diff = [torch.sum((noise_image - clean_image)**2) for clean_image in clean_images]
            # min_index = clean_to_noise_diff.index(min(clean_to_noise_diff))
            # clean_image = clean_images[min_index]
            clean_image = random.choice(clean_images)
        return noise_image, clean_image, label



'''
EncoderBlock for AutoEncoder and Variational AutoEncoder
'''
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s1, p1, s2, p2, sskip, pskip):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s1, padding=p1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=s2, padding=p2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=sskip, padding=pskip),
            nn.BatchNorm2d(out_channels)
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


'''
DecoderBlock for AutoEncoder and Variational AutoEncoder
'''
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s1, p1, s2, p2, sskip, pskip, k1, k2, kskip):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k1, stride=s1, padding=p1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=k2, stride=s2, padding=p2)
        self.residual_connection = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kskip, stride=sskip, padding=pskip),
        )
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += residual
        out = self.relu2(out)
        return out


class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.ae_block1 = EncoderBlock(1, 16, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        self.ae_block2 = EncoderBlock(16, 32, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        self.ae_block3 = EncoderBlock(32, 64, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        self.ae_flatten = nn.Flatten()
        self.ae_linear = nn.Linear(64*4*4, 10*1*1)
        self.ae_sigmoid = nn.Sigmoid()
        self.ae_unflatten = nn.Unflatten(1, (10, 1, 1))
        
        # self.vae_block1 = EncoderBlock(1, 16, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        # self.vae_block2 = EncoderBlock(16, 32, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        # self.vae_block3 = EncoderBlock(32, 64, s1=1, p1=1, s2=2, p2=1, sskip=2, pskip=0)
        self.vae_block1 = EncoderBlock(1, 8, s1=1, p1=4, s2=2, p2=4, sskip=2, pskip=6)
        self.vae_block2 = EncoderBlock(8, 16, s1=1, p1=3, s2=2, p2=3, sskip=2, pskip=4)
        self.vae_block3 = EncoderBlock(16, 24, s1=1, p1=2, s2=2, p2=3, sskip=2, pskip=3)
        self.vae_block4 = EncoderBlock(24, 32, s1=1, p1=2, s2=2, p2=2, sskip=2, pskip=2)
        self.vae_block5 = EncoderBlock(32, 64, s1=2, p1=2, s2=2, p2=2, sskip=2, pskip=0)
        self.vae_flatten = nn.Flatten()
        self.vae_mean_layer = nn.Linear(64*4*4, 256)
        self.vae_logvar_layer = nn.Linear(64*4*4, 256)

    # Forward pass of AutoEncoder
    def forward_ae(self, x):
        x = self.ae_block1(x)
        x = self.ae_block2(x)
        x = self.ae_block3(x)
        x = self.ae_flatten(x)
        x = self.ae_linear(x)
        x = self.ae_sigmoid(x)
        x = self.ae_unflatten(x)
        return x
    
    def encode_vae(self, x):
        x = self.vae_block1(x)
        x = self.vae_block2(x)
        x = self.vae_block3(x)
        x = self.vae_block4(x)
        x = self.vae_block5(x)
        return x

    # Reparameterization trick to sample N(mu, sigma) from N(0, 1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # Forward pass of Variational AutoEncoder
    def forward_vae(self, x):
        x = self.encode_vae(x)
        x = self.vae_flatten(x)
        mu = self.vae_mean_layer(x)
        logvar = self.vae_logvar_layer(x)
        latent = self.reparameterize(mu, logvar)
        return mu, logvar, latent
        

class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.ae_flatten = nn.Flatten()
        self.ae_linear = nn.Linear(10*1*1, 64*4*4)
        self.ae_sigmoid = nn.Sigmoid()
        self.ae_unflatten = nn.Unflatten(1, (64, 4, 4))
        self.ae_block1 = DecoderBlock(64, 32, s1=2, p1=1, s2=1, p2=1, sskip=2, pskip=0, k1=3, k2=3, kskip=1)
        self.ae_block2 = DecoderBlock(32, 16, s1=1, p1=1, s2=2, p2=2, sskip=2, pskip=1, k1=3, k2=3, kskip=1)
        self.ae_block3 = DecoderBlock(16, 1, s1=2, p1=4, s2=2, p2=3, sskip=3, pskip=2, k1=4, k2=4, kskip=2)
       
        self.vae_linear = nn.Linear(256, 64*4*4)
        self.vae_activation = nn.Sigmoid()
        self.vae_unflatten = nn.Unflatten(1, (64, 4, 4))
        # self.vae_block1 = DecoderBlock(64, 32, s1=2, p1=1, s2=1, p2=1, sskip=2, pskip=0, k1=3, k2=3, kskip=1)
        # self.vae_block2 = DecoderBlock(32, 16, s1=1, p1=1, s2=2, p2=2, sskip=2, pskip=1, k1=3, k2=3, kskip=1)
        # self.vae_block3 = DecoderBlock(16, 1, s1=2, p1=4, s2=2, p2=3, sskip=3, pskip=2, k1=4, k2=4, kskip=2)
        self.vae_block1 = DecoderBlock(64, 32, s1=1, p1=0, s2=1, p2=1, sskip=3, pskip=2, k1=3, k2=3, kskip=1)
        self.vae_block2 = DecoderBlock(32, 24, s1=1, p1=1, s2=2, p2=2, sskip=2, pskip=1, k1=3, k2=3, kskip=1)
        self.vae_block3 = DecoderBlock(24, 16, s1=1, p1=0, s2=1, p2=0, sskip=3, pskip=6, k1=3, k2=3, kskip=1)
        self.vae_block4 = DecoderBlock(16, 8, s1=2, p1=2, s2=1, p2=2, sskip=2, pskip=2, k1=3, k2=3, kskip=1)
        self.vae_block5 = DecoderBlock(8, 1, s1=1, p1=4, s2=2, p2=3, sskip=2, pskip=7, k1=4, k2=4, kskip=2)

    # Decoder forward pass for AutoEncoder
    def forward_ae(self, x):
        x = self.ae_flatten(x)
        x = self.ae_linear(x)
        x = self.ae_sigmoid(x)
        x = self.ae_unflatten(x)
        x = self.ae_block1(x)
        x = self.ae_block2(x)
        x = self.ae_block3(x)
        return x
    
    # Decoder forward pass for Variational AutoEncoder
    def forward_vae(self, x):
        x = self.vae_linear(x)
        x = self.vae_activation(x)
        x = self.vae_unflatten(x)
        x = self.vae_block1(x)
        x = self.vae_block2(x)
        x = self.vae_block3(x)
        x = self.vae_block4(x)
        x = self.vae_block5(x)
        return x


class AELossFn(nn.Module):
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(AELossFn, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, y):
        return self.mse_loss(x, y)
    

class VAELossFn(nn.Module):
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(VAELossFn, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.mse_loss(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence


def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train 
    """
    return list(E.parameters()) + list(D.parameters())


class AETrainer(): 
    """
    Write code for training AutoEncoder here.
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu) -> None:
        self.dataloader = dataloader
        self.encoder:Encoder = encoder
        self.decoder:Decoder = decoder
        self.loss_fn:AELossFn = loss_fn
        self.optimizer = optimizer
        self.gpu_is_available = gpu == 'True' or gpu == 'T' or gpu == True
        self.train()
    
    def train(self):
        device = torch.device('cuda') if self.gpu_is_available else torch.device('cpu')
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        
        for epoch in range(1, EPOCH+1):
            total_loss = 0
            total_similarity = 0
            # Logits and Labels for t-SNE plot
            if epoch % 10 == 0:
                logits_list = []
                labels_list = []
            
            for minibatch, (noise_image, clean_image, label) in enumerate(self.dataloader):
                self.optimizer.zero_grad() 
                noise_image = noise_image.to(device)
                clean_image = clean_image.to(device)
                logits = self.encoder.forward_ae(noise_image)
                if epoch % 10 == 0:
                    logits_list.append(logits)
                    labels_list.append(label)
                output = self.decoder.forward_ae(logits)
                loss = self.loss_fn(output, clean_image)
                similarity = structure_similarity_index(output.to('cpu'), clean_image.to('cpu'))
                total_loss += loss
                total_similarity += similarity
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch+1, loss, similarity))
                loss.backward()
                self.optimizer.step()
            total_loss /= len(self.dataloader)
            total_similarity /= len(self.dataloader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss, total_similarity))

            # t-SNE plot code
            if epoch % 10 == 0:
                logits = torch.cat(logits_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                logits_reshaped = logits.view(logits.size(0), -1)
                tsne = TSNE(n_components=2, random_state=42)
                embeddings = tsne.fit_transform(logits_reshaped.to('cpu').detach().numpy())
                plt.figure(figsize=(10, 8))
                plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels.to('cpu'), cmap='tab10', alpha=0.7)
                plt.colorbar()
                plt.title('t-SNE Plot of Logits')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.savefig(f'AE_epoch_{epoch}.png')
                plt.close()
        
        # Save model checkpoints
        self.encoder = self.encoder.to('cpu')
        self.decoder = self.decoder.to('cpu')
        checkpoint_encoder = {
            'model_state_dict': self.encoder.state_dict(),
            'loss': total_loss,
            'similarity': total_similarity
        }
        checkpoint_decoder = {
            'model_state_dict': self.decoder.state_dict(),
            'loss': total_loss,
            'similarity': total_similarity
        }
        torch.save(checkpoint_encoder, os.path.join(SAVEPATH, 'AE_encoder_checkpoint.pth'))
        torch.save(checkpoint_decoder, os.path.join(SAVEPATH, 'AE_decoder_checkpoint.pth'))


'''
Custom function to test AutoEncoder
Not for TA evaluation
'''
def AETester(dataloader, encoder, decoder, loss_fn, gpu):
    """
    Write code for testing AutoEncoder here.
    """
    device = torch.device('cuda') if (gpu == 'True' or gpu == 'T' or gpu == True) else torch.device('cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    total_similarity = 0
    total_loss = 0
    for minibatch, (noise_image, clean_image, label) in enumerate(dataloader):
        noise_image = noise_image.to(device)
        clean_image = clean_image.to(device)
        logits = encoder.forward_ae(noise_image)
        output = decoder.forward_ae(logits)
        loss = loss_fn.mse_loss(output, clean_image)
        similarity = structure_similarity_index(output.to('cpu'), clean_image.to('cpu'))
        total_loss += loss
        total_similarity += similarity
    total_loss /= len(dataloader)
    total_similarity /= len(dataloader)
    print("----- Testing Loss:{}, Similarity:{}".format(total_loss, total_similarity))

            
class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu) -> None:
        self.dataloader = dataloader
        self.encoder:Encoder = encoder
        self.decoder:Decoder = decoder
        self.loss_fn:VAELossFn = loss_fn
        self.optimizer = optimizer
        self.gpu_is_available = gpu == 'True' or gpu == 'T' or gpu == True
        self.train()
    
    def train(self):
        device = torch.device('cuda') if self.gpu_is_available else torch.device('cpu')
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        
        for epoch in range(1, EPOCH+1):
            total_loss = 0
            total_similarity = 0
            # Logits and Labels for t-SNE plot
            if epoch % 10 == 0:
                logits_list = []
                labels_list = []
            for minibatch, (noise_image, clean_image, label) in enumerate(self.dataloader):
                self.optimizer.zero_grad() 
                noise_image = noise_image.to(device)
                clean_image = clean_image.to(device)
                mu, logvar, latent = self.encoder.forward_vae(noise_image)
                if epoch % 10 == 0:
                    logits_list.append(latent)
                    labels_list.append(label)
                output = self.decoder.forward_vae(latent)
                loss = self.loss_fn(noise_image, output, mu, logvar)
                similarity = structure_similarity_index(output.to('cpu'), clean_image.to('cpu'))
                total_loss += loss
                total_similarity += similarity
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch+1, loss, similarity))
                loss.backward()
                self.optimizer.step()
            total_loss /= len(self.dataloader)
            total_similarity /= len(self.dataloader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss, total_similarity))

            # t-SNE plot code
            if epoch % 10 == 0:
                logits = torch.cat(logits_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                logits_reshaped = logits.view(logits.size(0), -1)
                tsne = TSNE(n_components=2, random_state=42)
                embeddings = tsne.fit_transform(logits_reshaped.to('cpu').detach().numpy())
                plt.figure(figsize=(10, 8))
                plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels.to('cpu'), cmap='tab10', alpha=0.7)
                plt.colorbar()
                plt.title('t-SNE Plot of Logits')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.savefig(f'VAE_epoch_{epoch}.png')
                plt.close()
        
        # Save model checkpoints
        self.encoder = self.encoder.to('cpu')
        self.decoder = self.decoder.to('cpu')
        checkpoint_encoder = {
            'model_state_dict': self.encoder.state_dict(),
            'loss': total_loss,
            'similarity': total_similarity
        }
        checkpoint_decoder = {
            'model_state_dict': self.decoder.state_dict(),
            'loss': total_loss,
            'similarity': total_similarity
        }
        torch.save(checkpoint_encoder, os.path.join(SAVEPATH, 'VAE_encoder_checkpoint.pth'))
        torch.save(checkpoint_decoder, os.path.join(SAVEPATH, 'VAE_decoder_checkpoint.pth'))

'''
Custom function to test Variational AutoEncoder
Not for TA evaluation
'''
def VAETester(dataloader, encoder, decoder, loss_fn, gpu):
    """
    Write code for testing Variational AutoEncoder here.
    """
    device = torch.device('cuda') if (gpu == 'True' or gpu == 'T' or gpu == True) else torch.device('cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    total_similarity = 0
    total_loss = 0
    for minibatch, (noise_image, clean_image, label) in enumerate(dataloader):
        noise_image = noise_image.to(device)
        clean_image = clean_image.to(device)
        mu, logvar, latent = encoder.forward_vae(noise_image)
        output = decoder.forward_vae(latent)
        loss = loss_fn.mse_loss(output, clean_image)
        similarity = structure_similarity_index(output.to('cpu'), clean_image.to('cpu'))
        total_loss += loss
        total_similarity += similarity
    total_loss /= len(dataloader)
    total_similarity /= len(dataloader)
    print("----- Testing Loss:{}, Similarity:{}".format(total_loss, total_similarity))


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu):
        self.device = torch.device('cuda') if (gpu == 'True' or gpu == 'T' or gpu == True) else torch.device('cpu')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.data = AlteredMNIST()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(SAVEPATH, 'AE_encoder_checkpoint.pth'))['model_state_dict'])
        self.decoder.load_state_dict(torch.load(os.path.join(SAVEPATH, 'AE_decoder_checkpoint.pth'))['model_state_dict'])
        
    def from_path(self, sample, original, metric_type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample = self.data.transform2(self.data.loader(sample)).unsqueeze(0)
        original = self.data.transform2(self.data.loader(original)).unsqueeze(0)
        x = sample.to(self.device)
        latent = self.encoder.forward_ae(x)
        x_hat = self.decoder.forward_ae(latent)
        if metric_type == 'SSIM':
            return structure_similarity_index(x_hat, original)
        elif metric_type == 'PSNR':
            return peak_signal_to_noise_ratio(x_hat, original)

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu):
        self.device = torch.device('cuda') if (gpu == 'True' or gpu == 'T' or gpu == True) else torch.device('cpu')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.data = AlteredMNIST()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(SAVEPATH, 'VAE_encoder_checkpoint.pth'))['model_state_dict'])
        self.decoder.load_state_dict(torch.load(os.path.join(SAVEPATH, 'VAE_decoder_checkpoint.pth'))['model_state_dict'])
        
    def from_path(self, sample, original, metric_type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        sample = self.data.transform2(self.data.loader(sample)).unsqueeze(0)
        original = self.data.transform2(self.data.loader(original)).unsqueeze(0)
        x = sample.to(self.device)
        mu, logvar, latent = self.encoder.forward_vae(x)
        x_hat = self.decoder.forward_vae(latent)
        if metric_type == 'SSIM':
            return structure_similarity_index(x_hat, original)
        elif metric_type == 'PSNR':
            return peak_signal_to_noise_ratio(x_hat, original)


class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    def save_image(self, digit, save_path):
        
        pass


def peak_signal_to_noise_ratio(sample, original):
    """
    Write code to calculate PSNR. Return in float
    """
    sample, original = sample.to(torch.float64), original.to(torch.float64)
    mse = sample.sub(original).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()


def structure_similarity_index(sample, original):
    """
    Write code to calculate SSIM. Return in float
    """
    sample = sample.squeeze().cpu().detach().numpy()
    original = original.squeeze().cpu().detach().numpy()
    return structural_similarity(sample, original, data_range=1)
