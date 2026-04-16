from .imports import torch

class Common_Module(torch.nn.Module):
    def __init__(self, IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.pipe = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IN_CHANNEL,
                            out_channels=OUT_CHANNEL,
                            kernel_size=KERNEL_SIZE,
                            padding=PADDING),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=OUT_CHANNEL,
                            out_channels=OUT_CHANNEL,
                            kernel_size=KERNEL_SIZE,
                            padding=PADDING),
            torch.nn.ReLU(True)
        )
    def forward(self, input):
        return self.pipe(input)
    
class U_NET_VANILLA_ENCODER(torch.nn.Module):
    def __init__(self, IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.downsample = Common_Module(IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2) # kernel size 2 cuz halfing
        
    def forward(self, input):
        skip = self.downsample(input)
        out = self.pool(skip)
        return out, skip
class U_NET_VANILLA_DECODER(torch.nn.Module):
    def __init__(self, IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_channels=IN_CHANNEL, out_channels=OUT_CHANNEL, kernel_size=2, stride=2, padding=0)
        self.pipe = Common_Module(2*OUT_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1) # 2* to accomodate skip connecetion
    def forward(self, input, skip):
        up = self.upsample(input)
        cat = torch.concat([up, skip], dim=1)
        out = self.pipe(cat)
        return out


class Transformer_Encoder(torch.nn.Module):
    def __init__(self, EMBED_DIM = 512, num_heads=8):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=EMBED_DIM,
                                        num_heads=num_heads)
        self.norm_1 = torch.nn.LayerNorm(EMBED_DIM)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(in_features=EMBED_DIM,
                            out_features=2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=2048,
                            out_features=EMBED_DIM)
        )
        self.norm_2 = torch.nn.LayerNorm(EMBED_DIM)

    def forward(self, input):
        multi_head_out, _ = self.attention(input, input, input)
        add_1 = torch.add(input, multi_head_out)
        norm_1 = self.norm_1(add_1)
        ff_out = self.feedforward(norm_1)
        add_2 = torch.add(ff_out, norm_1)
        norm_2 = self.norm_2(add_2)
        return norm_2
    
class Trans_U_Net(torch.nn.Module):
    def __init__(self, FILTER_LIST = [64, 128, 256, 512], IN_CHANNEL = 3, OUT_CHANNEL = 1):
        super().__init__()
        self.encoder_list = [IN_CHANNEL] + FILTER_LIST
        self.bottleneck = FILTER_LIST[-1]
        self.bottleneck_depth = 12
        self.decoder_list = FILTER_LIST + [self.bottleneck]
        self.layers = torch.nn.ModuleDict()
        self.depth = len(FILTER_LIST)

        for i in range(self.depth): # note that self.encnoder/decoder_list both are len(self.depth) + 1 so no need of overflow cuz i + 1
            self.layers[f'encoder_{i}'] = U_NET_VANILLA_ENCODER(self.encoder_list[i], self.encoder_list[i+1])
            self.layers[f'decoder_{i}'] = U_NET_VANILLA_DECODER(self.decoder_list[i+1], self.decoder_list[i])
        
        for i in range(self.bottleneck_depth):
            self.layers[f'bottleneck_{i}'] = Transformer_Encoder()
        self.layers[f'output'] = torch.nn.Conv2d(in_channels=self.decoder_list[0],
                                                 out_channels=OUT_CHANNEL,
                                                 kernel_size=1)
        
    def forward(self, input):
        skips = []
        for i in range(self.depth):
            input, skip = self.layers[f'encoder_{i}'](input)
            skips.append(skip)

        # transformer need 1D version of image(s)
        B, C, H, W = input.shape
        input = input.flatten(2).permute(0, 2, 1) # to do H*W and order them by B HW C

        for i in range(self.bottleneck_depth):
            input = self.layers[f'bottleneck_{i}'](input)
        
        # reverse the 1D to 2D
        input = input.permute(0, 2, 1).view(B, C, H, W) 

        for i in reversed(range(self.depth)):
            input = self.layers[f'decoder_{i}'](input, skips[i])
        output = self.layers['output'](input)
        return output
    