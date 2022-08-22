import math

import torch
import torchvision.transforms as T
import webdataset as wds
from torch import nn
from torch.utils import checkpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import timm
from timm.models.swin_transformer_v2 import SwinTransformerV2


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, inject_time_embeds=False, time_dims=None):
        super().__init__()
        self.t_embed_proj = nn.Linear(time_dims, in_channels) if inject_time_embeds else None
        self.resnet = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for i in range(depth):
            self.resnet.append(nn.Sequential(
                nn.GroupNorm(in_channels, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1)),
                nn.GroupNorm(in_channels, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1))
            ))

        if in_channels != out_channels:
            self.final_conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1))
        else:
            self.final_conv = None

    def forward(self, x, t_embed):
        if self.t_embed_proj is not None:
            t = self.t_embed_proj(t_embed)
            x = x + torch.unsqueeze(torch.unsqueeze(t, -1), -1)
        for i in range(len(self.resnet)):
            x = x + self.resnet[i](x)
        if self.final_conv is not None:
            x = self.final_conv(x)
        return x


#
# Training settings for image segmentation from Swin V2 paper:
# AdamW optimizer with an initial learning rate of 6x10e-5, a weight decay of 0.05,
# a linear decayed learning rate scheduler with 375 iteration linear warm-up. Batch size 64 for 40k iterations.
#
class SwinGenerator(nn.Module):
    def __init__(self, base_channels, decoder_depth=3, backbone_dims=48, window_size=16, time_dims=64):
        super().__init__()

        self._base_channels = base_channels

        self.backbone = SwinTransformerV2(64, 1, embed_dim=backbone_dims, window_size=window_size)
        self.backbone_feature_proj = nn.ModuleList()
        for i in range(4):
            in_channels = backbone_dims * 2 ** i
            out_channels = base_channels * 2 ** i
            self.backbone_feature_proj.append(nn.Conv2d(in_channels, out_channels, (1, 1)))

        self.t_embedder = SinusoidalPositionEmbeddings(time_dims)
        self.t_embed_projections = nn.ModuleDict({
            str(i): nn.Linear(time_dims, l.dim) for i, l in enumerate(self.backbone.layers)
        })

        self.decoder = nn.ModuleList()
        for i in range(3):
            channels = base_channels * 2 ** (3 - i)
            self.decoder.append(UpBlock(channels, channels * 2, decoder_depth, inject_time_embeds=True, time_dims=time_dims))
        self.decoder.append(UpBlock(base_channels, 3, decoder_depth, inject_time_embeds=True, time_dims=time_dims))

        self.pixel_shuffle = nn.PixelShuffle(2)

    def extract_features(self, swin, x, t_embed):
        x = swin.patch_embed(x)
        if swin.absolute_pos_embed is not None:
            x = x + swin.absolute_pos_embed
        x = swin.pos_drop(x)

        out = []
        for step, layer in enumerate(swin.layers):
            x = x + torch.unsqueeze(self.t_embed_projections[str(step)](t_embed), 1)

            for blk in layer.blocks:
                if layer.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            last_x = x
            x = layer.downsample(x)

            if last_x.shape[1] != x.shape[1]:
                out.append(last_x)

        x = swin.norm(x)
        out.append(x)

        for i in range(len(out)):
            res = round(math.sqrt(out[i].shape[1]))
            out[i] = torch.moveaxis(torch.reshape(out[i], (-1, res, res, out[i].shape[-1])), -1, 1)

        return out

    def forward(self, x, t):
        t_embed = self.t_embedder(t)
        # 64x64, 32x32, 16x16, 8x8
        features = self.extract_features(self.backbone, x, t_embed)

        x = features[-1]
        for i, layer in enumerate(self.decoder):
            if i != 0:
                residual = self.backbone_feature_proj[-i-1](features[-i-1])
                x = x + residual
            else:
                x = self.backbone_feature_proj[-1](features[-1])
            x = layer(x, t_embed)
            if i != len(self.decoder) - 1:
                x = self.pixel_shuffle(x)

        return x


if __name__ == '__main__':
    pos_embed = SinusoidalPositionEmbeddings(24)
    t = torch.arange(1000)
    t_embed = pos_embed(t)

    plt.plot(t.numpy(), t_embed.numpy()[:, 18:])


    def preprocess(x):
        x, = x
        x = T.Compose([
            T.ToTensor(),
            T.Resize(64),
            T.CenterCrop(64)
        ])(x)
        return x * 2 - 1


    lsun_bedroom = wds.WebDataset('../bedroom.tar').decode("rgb").to_tuple("input.jpg").map(preprocess)
    dl = DataLoader(lsun_bedroom, batch_size=1)

    m = SwinGenerator(32)
    for x in dl:
        o = m(x, torch.zeros(1))
        pass
