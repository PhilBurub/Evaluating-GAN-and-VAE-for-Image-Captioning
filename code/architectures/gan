import torch
from torch import nn

class Descriminator(nn.Module):
    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        input_tokens: int = 100,
        kernel_size = 5,
        kernel_stride = 5,
        layers_num: int = 3,
    ) -> None:

        super().__init__()
        
        nn_layers = []
        self.output_shape = self.input_tokens = input_tokens
        
        self.img_encoder = nn.Conv1d(
            img_dim,
            hidden_dim,
            1
        )
        
        self.text_encoder = nn.Conv1d(
            text_dim,
            hidden_dim,
            1
        )
        
        for i in range(layers_num):
            nn_layers.extend([
                nn.ReLU(),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    kernel_stride
                )
            ])
            self.output_shape = int((self.output_shape - kernel_size) / kernel_stride) + 1

        nn_layers.extend([
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.output_shape * hidden_dim, 1),
            nn.Flatten(start_dim=0)
        ])
        self.descriminator = nn.Sequential(*nn_layers)

    def forward(self, c_img: torch.Tensor, x: torch.Tensor):
        inp = torch.concat(
            [
                self.img_encoder(c_img),
                self.text_encoder(x)
            ],
            dim=1
        )
        return self.descriminator(inp)
