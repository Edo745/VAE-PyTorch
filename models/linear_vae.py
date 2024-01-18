import torch

class LinearVariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=64) -> None:
        super(LinearVariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28,4096),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4096,128)
        )

        self.fc_mu = torch.nn.Linear(128,self.latent_dim) #encoder gets me mu
        
        self.fc_logvar = torch.nn.Linear(128,self.latent_dim) #encoder gets me log(sigma)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim,128),
            torch.nn.Linear(128,4096),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4096,28*28),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        f = self.encoder(x)
        return self.fc_mu(f), self.fc_logvar(f)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #print (f'std: {std.shape}, logvar: {logvar.shape}, eps: {eps.shape}' )
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) #reparameterize and sample
        x_rec = self.decoder(z)
        return  x_rec.view(-1,1,28,28), mu, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        samples = self.decoder(z.to(device))
        return samples


