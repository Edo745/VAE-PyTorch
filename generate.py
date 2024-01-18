import torch
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import argparse
from models import linear_vae
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 

# Data
class BinaryMNIST(Dataset):
    def __init__(self, i, l):
        self.images , self.labels = i, l

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image[image > 0] = 1
        return image[None,:].float(), label

training_data = datasets.MNIST(
    root="VAE-PyTorch/data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="VAE-PyTorch/data",
    train=False,
    download=True,
    transform=ToTensor()
)
binary_train = BinaryMNIST(training_data.data, training_data.targets)
binary_test = BinaryMNIST(test_data.data, test_data.targets)
train_dataloader = DataLoader(binary_train, batch_size=256, shuffle=True)
test_dataloader = DataLoader(binary_test, batch_size=256)
 
# Parser
parser = argparse.ArgumentParser(description='Generate images using the trained VAE model.')
parser.add_argument('--latent_dim', type=int, default=32, help='Latent size')
parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--output_path', type=str, default='VAE-PyTorch/results', help='Path to save generated images')
parser.add_argument('--digit_to_generate', type=int, choices=list(range(10)), help='Digit to generate')
parser.add_argument('--from_digit', type=int, choices=list(range(10)), help='Digit to transition from')
parser.add_argument('--to_digit', type=int, choices=list(range(10)), help='Digit to transition to')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.checkpoint_path is None:
    print("No checkpoint path provided. Please provide a checkpoint path using the --checkpoint_path argument.")
    exit()
if not os.path.exists(args.checkpoint_path):
    print(f"The checkpoint path '{args.checkpoint_path}' does not exist. Please provide a valid checkpoint path.")
    exit()
    
model = linear_vae.LinearVariationalAutoencoder(args.latent_dim).to(device)

if os.path.exists(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
       
# Generate a specific digit
if args.digit_to_generate is not None:
    means=[]
    variances=[]
    labels = []

    for x,y in train_dataloader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        means.append(mu)
        variances.append(logvar.exp())
        labels.append(y.squeeze())

    mu = torch.cat(means)
    logvar = torch.cat(variances)
    l = torch.cat(labels)
    
    # mean and variance of "digit_to_generate"
    m = mu[l == args.digit_to_generate, :].mean(0)
    v = logvar[l == args.digit_to_generate, :].var(0)
    
    z = torch.randn(1, model.latent_dim).to(device)
    z = z.view(1, -1)

    num_samples = args.num_samples

    # Create a single subplot
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 2))

    for idx in range(num_samples):
        # Generate a random sample 'z' using the mean and variance of the class 'i'
        
        z = torch.randn(1, model.latent_dim) * v.sqrt() + m

        # Decode the sample
        i = model.decoder(z.to(device))

        # Reshape and convert to NumPy array for visualization
        i_r = i.detach().cpu().view(-1, 28, 28).squeeze().numpy()

        # Overlay the image in the corresponding subplot
        axs[idx].imshow(i_r, cmap="gray")
        axs[idx].axis('off')

    plt.savefig(f"{args.output_path}/{args.digit_to_generate}_generate")
    plt.show()
    
    
# From a digit to another
# z∗=zj6+δ⋅v0−>6
if args.from_digit is not None and args.to_digit is not None:
    from_digit = args.from_digit
    to_digit = args.to_digit

    # Means and variances for digits from_digit and to_digit
    mu_from = mu[l == from_digit, :].mean(0)
    logvar_from = logvar[l == from_digit, :].mean(0)

    mu_to = mu[l == to_digit, :].mean(0)
    logvar_to = logvar[l == to_digit, :].mean(0)

    # Interpolation vector from to_digit to from_digit
    delta_to_from = mu_to - mu_from

    # Choose a specific latent code z_j_from for digit from_digit
    j_from = 10  # You can change this index based on your preference
    z_j_from = mu[l == from_digit][j_from]

    # Choose a specific latent code z_j_to for digit to_digit
    j_to = 0  # You can change this index based on your preference
    z_j_to = mu[l == to_digit][j_to]

    # Difference vector between latent codes z_j_from and z_j_to
    d = z_j_to - z_j_from

    # Number of samples
    num_samples = len(np.arange(start=0, stop=1, step=0.1))

    fig, axs = plt.subplots(1, num_samples, figsize=(15, 2))

    for i, a in enumerate(np.arange(start=0, stop=1, step=0.1)):
        # Interpolate between z_j_from and z_j_to
        z_from_to = z_j_from + a * d

        # Decode the interpolated sample
        decoded_sample = model.decoder(z_from_to.to(device))

        # Reshape and convert to NumPy array for visualization
        img = decoded_sample.detach().cpu().view(-1, 28, 28).squeeze().numpy()

        # Overlay the image in the corresponding subplot
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')

    plt.savefig(f"{args.output_path}/from_{args.from_digit}_to_{to_digit}")
    plt.show()
