import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
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

        # [b, 768, 14*14] -> [b, 14*14, 768]
        x = torch.transpose(x, dim0=1, dim1=2)

        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,           # 输入token的dim (768)
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches+1, total_embed_dim] if input_size=224, num_patches=196, total_embed_dim=768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches+1, 3*total_embed_dim]
        # reshape(): -> [batch_size, num_patches+1, 3, num_heads, total_embed_dim//num_heads]
        # permute(): -> [3, batch_size, num_heads, num_patches+1, total_embed_dim//num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (Q * K_-1) / (d_k ** 0.5)
        # transpose(): -> [batch_size, num_heads, embed_dim_per_head, num_patches+1]
        # @: -> [batch_size, num_heads, num_patches+1, num_patches+1]
        attn = (q @ k.transpose(2, 3)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        # transpose(): -> [batch_size, num_patches+1, num_heads, embed_dim_per_head], 上面把3调到前面了，这里需要还原
        # reshape: -> [batch_size, num_patches+1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

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
        # if input size is 224*224, num_patches = (224*224)/(16*16) = 196
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # if input size is 224*224, pos_embed: [1, 196+1, 768]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # stochastic depth decay rule 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]


def test():
    i = torch.randn(4, 3, 224, 224)
    pe = PatchEmbed()
    o = pe(i)

    l1 = nn.Linear(768, 768*3, bias=False)
    attn = Attention(dim=768, attn_drop_ratio=0.2, proj_drop_ratio=0.2)
    i1 = torch.randn(4, 196+1, 768)
    o1 = attn(i1)
    print(o1.shape)


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
    test()
