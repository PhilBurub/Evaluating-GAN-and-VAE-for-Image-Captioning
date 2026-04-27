import torch
from torch import nn

class VAEnc(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        input_tokens: int = 100,
        kernel_size = 5,
        kernel_stride = 5,
        layers_num: int = 3,
    ) -> None:

        super().__init__()
        
        nn_layers = []
        self.input_tokens = input_tokens
        output_shape = input_tokens
        for i in range(layers_num):
            nn_layers.extend([
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    kernel_stride
                ),
                nn.ReLU()
            ])
            output_shape = int((output_shape - kernel_size) / kernel_stride) + 1

        nn_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*nn_layers)
        
        self.fc_mu = nn.Linear(output_shape * hidden_dim, output_dim)
        self.fc_var = nn.Linear(output_shape * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
