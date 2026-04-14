from .imports import torch, torchvision, OrderedDict
from dataclasses import dataclass

__all__ = ['Common_Module', 'U_NET_VANILLA_ENCODER', 'U_NET_VANILLA_DECODER', 'U_NET_VANILLA', 'U_NET_RESNET', 'U_NET_RESNET_ATTENTION', 'U_NET_PLUS_PLUS']

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

class AttentionGate(torch.nn.Module):
    def __init__(self, IN_g, IN_x, OUT_gx):
        '''
        _g is the gating signal, recieved from lower layers hence, half in size
        _x is the skip connection input to attention gate
        _gx is output dim for above two cuz we need to do inplace addition hence need to make em same dim
        '''
        super().__init__()
        self.in_g = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IN_g,
                            out_channels=OUT_gx,
                            kernel_size=1),
            torch.nn.BatchNorm2d(num_features=OUT_gx)            
        )
        self.in_x = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IN_x,
                            out_channels=OUT_gx,
                            kernel_size=1),
            torch.nn.BatchNorm2d(num_features=OUT_gx)
        )
        # self.addition = torch.add() done in foreward 
        self.pipe = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=OUT_gx,
                            out_channels=1,
                            kernel_size=1),# one channel output because we want a (1 H W) map to overlay to each pixel of image
            torch.nn.Sigmoid(),
        )
    def forward(self, input, skip):
        out_g = self.in_g(input)
        out_x = self.in_x(skip)
        # well input is from lower layer so its half the size, so we need to interpolate it
        out_g = torch.nn.functional.interpolate(input=out_g, size=out_x.shape[2:], mode='bilinear')
        out = torch.add(out_g, out_x)
        out = self.pipe(out)
        return skip * out # <--- applied attention to skip connection to focus on relevant parts of image(s)
        
class Attention_Decoder(torch.nn.Module):
    """
    __IN__ in_channel, out_channel
    __OUT__ out (1 pass through decodre block)
    """
    def __init__(self, IN_CHANNEL, OUT_CHANNEL):
        # now we have a SKIP_CHANNEL, cuz backbone has weird default channel size which might not match out own
        # hence we need to match them manually
        super().__init__()
        self.attention = AttentionGate(IN_CHANNEL, OUT_CHANNEL, OUT_CHANNEL)
        self.upsample = torch.nn.ConvTranspose2d(in_channels=IN_CHANNEL,
                                                 out_channels=OUT_CHANNEL,
                                                 kernel_size=2,
                                                 stride=2)
        self.pipe = torch.nn.Sequential(
            OrderedDict(
                {
                    'up_1' : torch.nn.Conv2d(in_channels=OUT_CHANNEL*2,
                                                out_channels=OUT_CHANNEL,
                                                kernel_size=3,
                                                padding=1),
                    'batch_norm_1' : torch.nn.BatchNorm2d(num_features=OUT_CHANNEL),
                    'activation_1' : torch.nn.ReLU(),
                    'up_2' : torch.nn.Conv2d(in_channels=OUT_CHANNEL,
                                                out_channels=OUT_CHANNEL,
                                                kernel_size=3,
                                                padding=1),
                    'batch_norm_2' : torch.nn.BatchNorm2d(num_features=OUT_CHANNEL),
                    'activation_2' : torch.nn.ReLU(),
                }
            )
        )
        
    def forward(self, input, skip):
        skip = self.attention(input, skip)
        out = self.upsample(input)
        out = torch.concat([skip, out], dim=1) # dim = 1 for along channels
        out = self.pipe(out)
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

class Common_Module_BN(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

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
        self.output = torch.nn.Conv2d(in_channels=OUT_CHANNEL,
                                      out_channels=1,
                                      kernel_size=1)

    def forward(self, input):
        intermediate_outputs = {}
        for j in range(self.depth):
            for i in range(self.depth - j):
                if j == 0: # that means it is encoder part
                    input_tensor = input if i == 0 else self.pool(intermediate_outputs[f'{i-1}_{j}']) # else last layers' tensor
                    intermediate_outputs[f'{i}_{j}'] = self.blocks[f'{i}_{j}'](input_tensor)

                else:
                    # else I have j number of skip connections to handle
                    skip_connections = [intermediate_outputs[f'{i}_{k}'] for k in range(j)]
                    upsampled_g = self.upsample(intermediate_outputs[f'{i+1}_{j-1}'])

                    concatenated = torch.concat(skip_connections + [upsampled_g], dim=1)
                    intermediate_outputs[f'{i}_{j}'] = self.blocks[f'{i}_{j}'](concatenated)
                    # final = self.final(intermediate_outputs[f'0_{self.depth}'])
        final_outputs = []
        for j in range(1, self.depth):
            # Pass each top-row node through the final 1x1 conv
            mask = self.final(intermediate_outputs[f'0_{j}'])
            final_outputs.append(mask)
            
        return final_outputs 
    