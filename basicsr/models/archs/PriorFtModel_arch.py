import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base3
import loralib as lora


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,r=4,num_experts=3,top_k=1):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = lora.Conv2dMix(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.conv2 = lora.Conv2dMix(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.conv3 = lora.Conv2dMix(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            lora.Conv2dMix(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = lora.Conv2dMix(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.conv5 = lora.Conv2dMix(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, probs=None):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x,probs)
        x = self.conv2(x,probs)
        x = self.sg(x)
        x = x * (self.sca[1](self.sca[0](x),probs))
        x = self.conv3(x,probs)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y),probs)
        x = self.sg(x)
        x = self.conv5(x,probs)

        x = self.dropout2(x)

        return y + x * self.gamma

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class CustomSeqEnc(nn.Module):
    def __init__(self, chan, num_blocks,r=4,num_experts=3,top_k=1):
        super(CustomSeqEnc, self).__init__()
        self.blocks = nn.ModuleList(
            [NAFBlock(chan,r=r,num_experts=num_experts,top_k=top_k) for _ in range(num_blocks)]
        )
        self.conv1 = lora.Conv2dMix(in_channels=384,out_channels=128, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.act1 = nn.GELU()
        self.conv2 = lora.Conv2dMix(in_channels=128,out_channels=chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.conv3 = lora.Conv2dMix(in_channels=chan,out_channels=chan//16, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.act2 = nn.ReLU(True)
        self.conv4 = lora.Conv2dMix(in_channels=chan//16,out_channels=chan, kernel_size=1, padding=0, stride=1, groups=1, bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.act3 = nn.Sigmoid()
        self.norm = LayerNorm2d(chan)
        self.beta = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)

    def forward(self, x, de_prior, probs=None):
        inp = x
        x = self.norm(x)
        y = self.conv2(self.act1(self.conv1(de_prior,probs)),probs)
        y = self.act3(self.conv4(self.act2(self.conv3(y,probs)),probs))
        x = x*y
        x = inp + self.beta*x
        for block in self.blocks:
            x = block(x, probs)
        return x

class CustomSeq(nn.Module):
    def __init__(self, chan, num_blocks,r=4,num_experts=3,top_k=1):
        super(CustomSeq, self).__init__()
        self.blocks = nn.ModuleList(
            [NAFBlock(chan,r=r,num_experts=num_experts,top_k=top_k) for _ in range(num_blocks)]
        )
    def forward(self, x, probs):
        for block in self.blocks:
            x = block(x,probs)
        return x

class PriorFtModel(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],r=4,num_experts=3,top_k=1):
        super().__init__()

        self.intro = lora.Conv2dMix(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
        self.ending = lora.Conv2dMix(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
    
        self.intro_de = nn.Sequential(lora.Conv2dMix(in_channels=384, out_channels=384, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k), nn.GELU(),lora.Conv2dMix(in_channels=384, out_channels=384, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k), nn.GELU())

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                # nn.Sequential(
                #     *[NAFBlock(chan) for _ in range(num)]
                # )
                CustomSeqEnc(chan, num,r=r,num_experts=num_experts,top_k=top_k)
            )
            self.downs.append(
                lora.Conv2dMix(in_channels=chan, out_channels=chan * 2, kernel_size=2,stride=2,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k)
            )
            chan = chan * 2

        self.middle_blks = \
            CustomSeqEnc(chan, middle_blk_num,r=r,num_experts=num_experts,top_k=top_k)
            # nn.Sequential(
            #     *[NAFBlock(chan) for _ in range(middle_blk_num)]
            # )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    lora.Conv2dMix(in_channels=chan, out_channels=chan * 2, kernel_size=1, bias=False,r=r,lora_alpha=2*r,num_experts=num_experts,top_k=top_k),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                # nn.Sequential(
                #     *[NAFBlock(chan) for _ in range(num)]
                # )
                CustomSeq(chan, num, r=r,num_experts=num_experts,top_k=top_k)
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, de_prior, probs=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp,probs)

        de_prior = de_prior.unsqueeze(-1).unsqueeze(-1)
        de_prior = self.intro_de[3](self.intro_de[2](self.intro_de[1](self.intro_de[0](de_prior,probs)),probs))

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x,de_prior,probs)
            encs.append(x)
            x = down(x,probs)

        x = self.middle_blks(x, de_prior,probs)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up[1](up[0](x,probs))
            x = x + enc_skip
            x = decoder(x,probs)

        x = self.ending(x,probs)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class PriorFtModelL(Local_Base3, PriorFtModel):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base3.__init__(self)
        PriorFtModel.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
