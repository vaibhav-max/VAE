#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[12]:


import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, extra_param=None):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, latent_dim * 2)  # Two times latent_dim for mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for image data in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        enc_output = self.encoder(x)
        mu, logvar = enc_output[:, :latent_dim], enc_output[:, latent_dim:]

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        dec_output = self.decoder(z)

        return dec_output, mu, logvar


# In[3]:


class AnimalfaceDataset(Dataset):
    def __init__(self, transform, type='train', label_dict={"dog": 0, "cat": 1, "wild": 2}, img_width=128) -> None:
        self.transform = transform
        # self.root_dir specifies whether you are at afhq/train or afhq/val directory
        self.label_dict = label_dict
        self.root_dir = os.path.join(DATA_PATH, type)
        assert os.path.exists(self.root_dir), "Check for the dataset, it is not where it should be. If not present, you can download it by clicking above DATA_URL"
        subdir = os.listdir(self.root_dir)
        self.image_names = []

        for category in subdir:
            subdir_path = os.path.join(self.root_dir, category)
            self.image_names += os.listdir(subdir_path)

        self.img_arr = torch.zeros((len(self.image_names), 3, img_width, img_width))
        self.labels = torch.zeros(len(self.image_names))

        for i, img_name in enumerate(tqdm(self.image_names)):
            # if i > 10 :
            #     break
            label = self.label_dict[img_name.split("_")[1]]
            img_path = os.path.join(self.root_dir, img_name.split("_")[1], img_name)
            # Load image and convert it to RGB
            img = Image.open(img_path).convert('RGB')
            # Apply transformations to the image
            img = self.transform(img)
            self.img_arr[i] = img
            self.labels[i] = label

    def __getitem__(self, idx):
        return self.img_arr[idx], self.labels[idx]

    def __len__(self):
        return len(self.image_names)


# In[13]:


# Define the dataset and data loader
batch_size = 1024
label_dict = {"dog": 0, "cat": 1, "wild": 2}

DATA_PATH = "/data/home/vvaibhav/AI/VAE/afhq"
width = 128
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((width, width))])

# Assuming you have already defined the label_dict and other parameters for the dataset
train_dataset = AnimalfaceDataset(transform=train_transform, type='train', label_dict=label_dict, img_width=width)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# In[14]:


# Initialize the VAE
input_dim = 128 * 128 * 3  # Assuming images are RGB with size 128x128
hidden_dim = 256
latent_dim = 32
input_channels = 3
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 3, 128, 128), reduction='sum')

    #BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 128 * 128 * 3), reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)



# In[15]:
# Assuming you have a separate validation dataset
val_dataset = AnimalfaceDataset(transform=train_transform, type='val', label_dict=label_dict, img_width=width)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    vae.train()  # Set the mode of the VAE model to training mode
    total_train_loss = 0

    for batch in tqdm(train_dataloader):
        data, _ = batch
        data = data.to(device)

        optimizer.zero_grad()

        recon_data, mu, logvar = vae(data)
        loss = loss_function(recon_data, data, mu, logvar)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}")

    # Validation loop
    vae.eval()  # Set the mode to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_data, _ = val_batch
            val_data = val_data.to(device)

            recon_val_data, mu_val, logvar_val = vae(val_data)
            val_loss = loss_function(recon_val_data, val_data, mu_val, logvar_val)

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss}")

# Save the model after training
save_path = "/data/home/vvaibhav/AI/VAE/vae_model.pth"
torch.save(vae, save_path)


# Training loop
# num_epochs = 10

# for epoch in range(num_epochs):
#     vae.train() #is setting the mode of the VAE model to training mode
#     total_loss = 0

#     for batch in tqdm(train_dataloader):
#         data, _ = batch
#         data = data.to(device)

#         optimizer.zero_grad()

#         recon_data, mu, logvar = vae(data)
#         loss = loss_function(recon_data, data, mu, logvar)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_dataloader.dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")


# save_path = "/data/home/vvaibhav/AI/VAE/vae_model.pth"
# torch.save(vae, save_path)



# In[8]:


# Generate new images
vae.eval()
with torch.no_grad():
    # Generate random latent vectors
    random_latent = torch.randn(16, latent_dim).to(device)
    generated_images = vae.decoder(random_latent).view(-1, 3, 128, 128).cpu().numpy()



# In[16]:


print(len(generated_images))


# In[11]:


# Display or save the generated images as needed
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()

for i in range(16):
    image = generated_images[i].transpose(1, 2, 0)  # Assuming channels are the last dimension
    axes[i].imshow(image)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Save the generated images
save_path = "/data/home/vvaibhav/AI/VAE/afhq/generated"
plt.savefig(save_path)


# In[ ]:




