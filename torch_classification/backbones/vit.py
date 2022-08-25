import torch
import torch.nn as nn

__all__ = [
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224'
]


class PatchEmbeddingBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(PatchEmbeddingBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inplanes,
                      out_channels=planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential())

    def forward(self, x):
        # if inputs shape is [224, 224, 3]
        # -> [14, 14, planes(768)]
        x = self.layer(x)

        # [B, 768, 14, 14] -> [B, 14*14, 768]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1).contiguous()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, inplanes, head_nums=8):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_nums
        self.scale = (inplanes // head_nums) ** (-0.5)

        self.head_linear = nn.Linear(inplanes, inplanes * 3)
        self.proj_linear = nn.Linear(inplanes, inplanes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        print(x.shape)
        b, n, c = x.shape

        # [b, n, c] -> [b, n, 3, head_num, c//head_num] -> [3, b, head_num, n, c//head_num]
        x = self.head_linear(x)

        # [3, b, head_num, n, c//head_num] -> 3个 [b, head_num, n, c//head_num]


class LayerNorm(nn.Module):
    def __init__(self, inplanes, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.a_2 =

class TransformerEncoderLayer(nn.Module):
    def __init__(self, inplanes, head_nums, feedforward_ratio=4):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(inplanes=inplanes,
                                            head_nums=head_nums)

    def forward(self, x):
        x = x + self.attention()


class ViT(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio=4,
                 num_classes=1000):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.num_classes = num_classes

        # if inputs is [224, 224, 3]
        # patch_size is [16, 16], patch_nums = (224 / 16)**2 = 196
        # kernel_size=16, stride=16, (224-16+2*0)/16+1 = 14
        self.patch_embedding = PatchEmbeddingBlock(inplanes=3,
                                                   planes=self.embedding_planes,
                                                   kernel_size=self.patch_size,
                                                   stride=self.patch_size,
                                                   padding=0,
                                                   groups=1,
                                                   has_bn=False,
                                                   has_act=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_planes))

        # (self.image_size//self.patch_size)**2+1 = (224//16)**2+1 = 197
        self.position_encoding = nn.Parameter(
            torch.ones(1, (self.image_size // self.patch_size)**2+1, self.embedding_planes)
        )

        blocks = []
        for _ in range(self.block_nums):
            blocks.append(TransformerEncoderLayer(inplanes=self.embedding_planes,
                                                  head_nums=self.head_nums,
                                                  feedforward_ratio=self.feedforward_ratio))

    def forward(self, x):
        # if base_vit and image_size=[224, 224]
        # x.shape is [B, 196, 768]
        x = self.patch_embedding(x)

        # [B, 197, 768]
        x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1)

        # self.position_encoding [1, 197, 768]
        x = x + self.position_encoding


def _vit(image_size, patch_size, embedding_planes, blocks_nums, head_nums,
         **kwargs):
    model = ViT(image_size, patch_size, embedding_planes, blocks_nums, head_nums,
                **kwargs)
    return model


def vit_tiny_patch16_224(**kwargs):
    return _vit(224, 16, embedding_planes=192,
                blocks_nums=12,
                head_nums=3,
                **kwargs)


def vit_small_patch16_224(**kwargs):
    # [224, 224, 3] -> [14, 14, 384]
    return _vit(224, 16, embedding_planes=384,
                blocks_nums=12,
                head_nums=6,
                **kwargs)


def vit_base_patch16_224(**kwargs):
    # [224, 224, 3] -> [14, 14, 768]
    return _vit(224, 16, embedding_planes=768,
                blocks_nums=12,
                head_nums=12,
                **kwargs)


if __name__ == "__main__":
    inplanes = 10

