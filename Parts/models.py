from .imports import torch, torchvision, OrderedDict
from dataclasses import dataclass
from .blocks import *

__all__ = ['U_NET_VANILLA', 'U_NET_RESNET', 'U_NET_RESNET_ATTENTION', 'U_NET_PLUS_PLUS', 'TRANS_U_NET']

class U_NET_VANILLA(torch.nn.Module):
    def __init__(self, FILTER_LIST = [64, 128, 256, 512], IN_CHANNELS = 3, OUT_CHANNEL = 1, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.depth = len(FILTER_LIST)
        self.layers = torch.nn.ModuleDict() # will be used to register all layers
        self.encoder_list = [IN_CHANNELS] + FILTER_LIST
        self.bottleneck = FILTER_LIST[-1] * 2
        self.decoder_list = FILTER_LIST + [self.bottleneck]

        for i in range(self.depth):
            self.layers[f'encoder_{i}'] = U_NET_VANILLA_ENCODER(self.encoder_list[i], self.encoder_list[i + 1], KERNEL_SIZE, PADDING)
            self.layers[f'decoder_{i}'] = U_NET_VANILLA_DECODER(self.decoder_list[i + 1], self.decoder_list[i], KERNEL_SIZE, PADDING)
        self.layers[f'bottleneck'] = Common_Module(self.encoder_list[-1], self.bottleneck)
        self.layers[f'output'] = torch.nn.Conv2d(self.decoder_list[0], OUT_CHANNEL, kernel_size = 1)

    def forward(self, input):
            skips = []
            for i in range(self.depth):
                input, skip = self.layers[f'encoder_{i}'](input)
                skips.append(skip)
            
            input = self.layers['bottleneck'](input)
            
            skips = skips[::-1]
            for i in reversed(range(self.depth)):
                input = self.layers[f'decoder_{i}'](input, skips[(self.depth - 1) - i])
            input = self.layers[f'output'](input)        
            return input
            

class U_NET_RESNET(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.base_layers = list(self.backbone.children())

        # first three layers are input
        self.input = torch.nn.Sequential(*self.base_layers[:3])
        self.max_pool = self.base_layers[3] # cuz we need ouput before max pooling
        self.down_1 = self.base_layers[4]
        self.down_2 = self.base_layers[5]
        self.down_3 = self.base_layers[6]
        self.down_4 = self.base_layers[7]
        # this (7) is the bottleneck layer
    
        self.up_1 = U_NET_VANILLA_DECODER(512, 256)
        self.up_2 = U_NET_VANILLA_DECODER(256, 128)
        self.up_3 = U_NET_VANILLA_DECODER(128, 64)
        self.up_4 = U_NET_VANILLA_DECODER(64, 64)

        # well the resnet first block does downsample AND stride so need to updsample one extra time
        self.output_doubled = torch.nn.ConvTranspose2d(64, 64, 
                                                       kernel_size=2,
                                                       stride=2)
        self.output = torch.nn.Conv2d(64, 1, 1)

    def forward(self, input) -> torch.Tensor :
        skip_0 = self.input(input)
        temp = self.max_pool(skip_0)
        skip_1 = self.down_1(temp)
        skip_2 = self.down_2(skip_1)
        skip_3 = self.down_3(skip_2)
        skip_4 = self.down_4(skip_3)
        
        out = self.up_1(skip_4, skip_3)
        out = self.up_2(out, skip_2)
        out = self.up_3(out, skip_1)
        out = self.up_4(out, skip_0)
        out = self.output_doubled(out)
        out = self.output(out)
        return out

class U_NET_RESNET_ATTENTION(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.base_layers = list(self.backbone.children())

        # first three layers are input
        self.input = torch.nn.Sequential(*self.base_layers[:3])
        self.max_pool = self.base_layers[3] # cuz we need ouput before max pooling
        self.down_1 = self.base_layers[4]
        self.down_2 = self.base_layers[5]
        self.down_3 = self.base_layers[6]
        self.down_4 = self.base_layers[7]
        # this (7) is the bottleneck layer
    
        self.up_1 = Attention_Decoder(512, 256)
        self.up_2 = Attention_Decoder(256, 128)
        self.up_3 = Attention_Decoder(128, 64)
        self.up_4 = Attention_Decoder(64, 64)

        # well the resnet first block does downsample AND stride so need to updsample one extra time
        self.output_doubled = torch.nn.ConvTranspose2d(64, 64, 
                                                       kernel_size=2,
                                                       stride=2)
        self.output = torch.nn.Conv2d(64, 1, 1)

    def forward(self, input) -> torch.Tensor :
        skip_0 = self.input(input)
        temp = self.max_pool(skip_0)
        skip_1 = self.down_1(temp)
        skip_2 = self.down_2(skip_1)
        skip_3 = self.down_3(skip_2)
        skip_4 = self.down_4(skip_3)
        
        out = self.up_1(skip_4, skip_3)
        out = self.up_2(out, skip_2)
        out = self.up_3(out, skip_1)
        out = self.up_4(out, skip_0)
        out = self.output_doubled(out)
        out = self.output(out)
        return out

class U_NET_PLUS_PLUS(torch.nn.Module):
    def __init__(self, IN_CHANNEL = 3, OUT_CHANNEL = 1, FILTER_LIST=[64, 128, 256, 512]):
        super().__init__()
        self.filters = FILTER_LIST
        self.depth = len(FILTER_LIST)
        self.pool = torch.nn.MaxPool2d(kernel_size=2,
                                       stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.blocks = torch.nn.ModuleDict()

        for j in range(self.depth):
            for i in range(self.depth - j):
                # collectively, both loop work to parse the UNet++ architecture horizontally
                if j == 0:
                    in_channel = IN_CHANNEL if i == 0 else FILTER_LIST[i-1]
                    self.blocks[f'{i}_{j}'] = Common_Module_BN(in_channel, FILTER_LIST[i]) # self.blocks names a layers as well as 
                    # registers it, other way, like using python dict, will not register the layer with pytorch
                else:
                    in_channel = FILTER_LIST[i] * j  + FILTER_LIST[i+1] # filters[i] * j skip connections, filters[i+1] input from lower layer
                    self.blocks[f'{i}_{j}'] = Common_Module_BN(in_channel, FILTER_LIST[i])
        
        self.final = torch.nn.Conv2d(in_channels=FILTER_LIST[0],
                                     out_channels=OUT_CHANNEL,
                                     kernel_size=1)
        # self.output = torch.nn.Conv2d(in_channels=OUT_CHANNEL,
        #                               out_channels=1,
        #                               kernel_size=1)

    def forward(self, input):
        intermediate_outputs = {}
        
        # Encoder and Nested Skip Connections
        for j in range(self.depth):
            for i in range(self.depth - j):
                if j == 0: # that means it is encoder part
                    input_tensor = input if i == 0 else self.pool(intermediate_outputs[f'{i-1}_{0}'])# else last layers' tensor
                    intermediate_outputs[f'{i}_{j}'] = self.blocks[f'{i}_{j}'](input_tensor)
                else:
                    # else I have j number of skip connections to handle
                    skip_connections = [intermediate_outputs[f'{i}_{k}'] for k in range(j)]
                    upsampled_g = self.upsample(intermediate_outputs[f'{i+1}_{j-1}'])
                    concatenated = torch.concat(skip_connections + [upsampled_g], dim=1)
                    intermediate_outputs[f'{i}_{j}'] = self.blocks[f'{i}_{j}'](concatenated)

        # Deep Supervision : Map all top-row nodes to the output dimension
        # 'masks' is a list of Tensors from all intermediate outputs
        masks = [self.final(intermediate_outputs[f'0_{j}']) for j in range(1, self.depth)]

        if self.training:
            return masks # Training loop handles the list/averaging
        
        return masks[-1] # Inference uses only the highest-resolution result
    
class TRANS_U_NET(torch.nn.Module):
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