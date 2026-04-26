from torch import nn
import torch

class ImageConvAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        dropout_rate: float = 0.2,
        kernel_size = 5,
        kernel_stride = 5,
        layers_num: int = 3,
        **kwargs
    ):
        super().__init__()

        nn_layers = []

        for i in range(layers_num - 1):
            nn_layers.extend([
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    kernel_stride
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        nn_layers.append(
            nn.Conv1d(hidden_dim, output_dim, kernel_size, kernel_stride)
        )

        self.adapter = nn.Sequential(*nn_layers)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        return self.adapter(x)
