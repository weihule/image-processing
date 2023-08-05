import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3,
                 embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        self.img_size = img_size
        patch_size = (patch_size, patch_size)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // img_size[1])
        self.num_patches = grid_size[0] * grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size,
                              stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        # [b, 3, 224, 224] -> [b, 768, 14, 14]
        x = self.proj(x)

        # [b, 768, 14, 14] -> [b, 768, 14*14]
        x = torch.flatten(x, start_dim=2)

        # [b, 768, 14*12] -> [b, 14*14, 768]
        x = torch.transpose(x, dim0=1, dim1=2)

        x = self.norm(x)

        return x


class VisionTransform(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransform, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU()

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_c=in_c,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch)


def test():
    i = torch.randn(4, 3, 224, 224)
    pe = PatchEmbed()
    o = pe(i)
    print(o.shape)


def test_kwargs(first, *args, **kwargs):
    print('Required argument: ', first)
    print(type(kwargs))
    for v in args:
        print('Optional argument (args): ', v)
    for k, v in kwargs.items():
        print(f'Optional argument {k} (kwargs): {v}')


def mod(n, m):
    return n % m


if __name__ == "__main__":
    # test()

    mod_by_15 = partial(mod, 15)
    print(mod(15, 7))
    print(mod_by_15(7))
