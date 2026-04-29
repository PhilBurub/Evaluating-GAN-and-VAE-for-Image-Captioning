import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        text_tokens: int = 100,
        image_tokens: int = 100,
        kernel_size = 5,
        kernel_stride = 5,
        layers_num: int = 3
    ) -> None:

        super().__init__()
        
        image_layers = []
        text_layers = []
        
        output_text = self.text_tokens = text_tokens
        output_image = self.image_tokens = image_tokens
        
        for i in range(layers_num):
            image_layers.extend([
                nn.Conv1d(
                    img_dim if i==0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    kernel_stride
                ),
                nn.ReLU()
            ])
            output_image = int((output_image - kernel_size) / kernel_stride) + 1
            
            text_layers.extend([
                nn.Conv1d(
                    text_dim if i==0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    kernel_stride
                ),
                nn.ReLU()
            ])
            output_text = int((output_text - kernel_size) / kernel_stride) + 1

        image_layers.append(nn.Flatten())
        text_layers.append(nn.Flatten())
        
        self.img_encoder = nn.Sequential(*image_layers)
        self.text_encoder = nn.Sequential(*text_layers)
        
        self.descriminator = nn.Linear((output_image + output_text) * hidden_dim, 1)

    def forward(self, c_img: torch.Tensor, x: torch.Tensor):
        inp = torch.concat(
            [
                self.img_encoder(c_img),
                self.text_encoder(x)
            ],
            dim=1
        )
        return self.descriminator(inp).flatten()
