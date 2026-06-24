import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterise(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim, latent_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.conv = nn.Sequential(
            nn.Conv1d(vocab_size, 128, kernel_size=9, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.Tanh(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, vocab_size, seq_len)
            flat_size = self.conv(dummy).reshape(1, -1).shape[1]

        self.fc = nn.Linear(flat_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        embedded = F.one_hot(x, num_classes=self.vocab_size).float()
        x_conv = embedded.permute(0, 2, 1)
        h = self.conv(x_conv)
        h = h.reshape(h.size(0), -1)
        h = F.relu(self.fc(h))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, latent_dim, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.fc_z = nn.Linear(latent_dim, hidden_dim * n_layers)
        self.gru = nn.GRU(
            vocab_size + latent_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, x):
        hidden = self.fc_z(z)
        hidden = (
            hidden.reshape(-1, self.n_layers, self.hidden_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

        seq_len = x.size(1) - 1
        x_onehot = F.one_hot(
            x[:, :-1], num_classes=self.vocab_size
        ).float()
        z_repeated = z.unsqueeze(1).expand(-1, seq_len, -1)
        gru_input = torch.cat([x_onehot, z_repeated], dim=-1)
        output, _ = self.gru(gru_input, hidden)
        return self.fc_out(output)


class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_size=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, mu):
        return self.net(mu).squeeze(-1)


class VAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        hidden_dim,
        latent_dim,
        n_layers,
        dropout=0.5,
        prop_hidden_size=64,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, seq_len, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, hidden_dim, latent_dim, n_layers)

        self.cyp2d6_predictor = PropertyPredictor(
            latent_dim,
            hidden_size=prop_hidden_size,
            dropout=dropout,
        )
        self.cyp2c19_predictor = PropertyPredictor(
            latent_dim,
            hidden_size=prop_hidden_size,
            dropout=dropout,
        )

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterise(mu, log_var)
        logits = self.decoder(z, x)
        cyp2d6_logit = self.cyp2d6_predictor(mu)
        cyp2c19_logit = self.cyp2c19_predictor(mu)

        return logits, mu, log_var, cyp2d6_logit, cyp2c19_logit