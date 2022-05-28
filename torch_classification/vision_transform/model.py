import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

class PatchEmbed(nn.Module):
    """
    2D Image to patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim,     # 224*224*3 -> 14*14*768
                            kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

if __name__ == "__main__":
    pass