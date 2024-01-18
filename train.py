import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
from models.linear_vae import LinearVariationalAutoencoder 

parser = argparse.ArgumentParser(description='Training script for the VAE model.')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for DataLoader')
parser.add_argument('--beta', type=float, default=0.005, help='KL parameter')
parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving checkpoints')

# Parse the command-line arguments
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download the data
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

model = LinearVariationalAutoencoder()  

optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

checkpoint_path = 'VAE-PyTorch/checkpoints/'
os.makedirs('VAE-PyTorch/checkpoints/', exist_ok=True)
save_interval = 5

size = len(train_dataloader.dataset)
all_losses = [[],[],[]]

# VAE loss function
def vae_loss_function(x_rec, x, mu, log_var, beta):
  """
  Computes the VAE loss function.
  KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
  """
  kld_weight = beta
  # kld_weight = 1*(1./(28*28)) # original value: 0.005
  recons_loss = torch.nn.functional.binary_cross_entropy(x_rec, x) #p(x|z)
  sum_terms = 1 + log_var - mu ** 2 - log_var.exp() #BATCH_SIZE x LATENT_DIM
  kl_batch = -0.5 * torch.sum(sum_terms, dim = 1)
  kld_loss = torch.mean(kl_batch, dim = 0)
  loss = recons_loss + kld_weight * kld_loss

  return loss, recons_loss, kld_weight*kld_loss

# Training
for epoch in range(args.epochs):
    for batch, (X, y) in enumerate(train_dataloader):
      # Compute prediction and loss
        X = X.to(device)
        pred, mu, log_var = model(X)

        loss, rec_loss, kld = vae_loss_function(pred, X, mu, log_var, args.beta)

        all_losses[0].append(loss.cpu().detach().numpy())
        all_losses[1].append(rec_loss.cpu().detach().numpy())
        all_losses[2].append(kld.cpu().detach().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Epoch [{epoch+1}/{args.epochs}] loss: {loss:>7f}, rec_loss: {rec_loss:>7f}, kld: {kld:>7f} | [{current:>5d}/{size:>5d}]")

    if (epoch + 1) % save_interval == 0:
        checkpoint_dir = 'VAE-PyTorch/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'{epoch + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved weights at epoch {epoch + 1} to {checkpoint_path}")

