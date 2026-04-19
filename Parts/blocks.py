from .imports import torch, OrderedDict

__all__ = ['Common_Module', 'Common_Module_BN', 'U_NET_VANILLA_ENCODER', 'U_NET_VANILLA_DECODER', 'AttentionGate', 'Attention_Decoder', 'Transformer_Encoder']

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

class U_NET_VANILLA_ENCODER(torch.nn.Module):
    def __init__(self, IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.downsample = Common_Module_BN(IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE, PADDING)
        self.pool = torch.nn.MaxPool2d(kernel_size=2) # kernel size 2 cuz halfing
        
    def forward(self, input):
        skip = self.downsample(input)
        out = self.pool(skip)
        return out, skip
    
class U_NET_VANILLA_DECODER(torch.nn.Module):
    def __init__(self, IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE = 3, PADDING = 1):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_channels=IN_CHANNEL, out_channels=OUT_CHANNEL, kernel_size=2, stride=2, padding=0)
        self.pipe = Common_Module_BN(2*OUT_CHANNEL, OUT_CHANNEL, KERNEL_SIZE, PADDING) # 2* to accomodate skip connecetion
    def forward(self, input, skip):
        up = self.upsample(input)
        cat = torch.concat([up, skip], dim=1)
        out = self.pipe(cat)
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
            torch.nn.ReLU(True),
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
                    'activation_1' : torch.nn.ReLU(True),
                    'up_2' : torch.nn.Conv2d(in_channels=OUT_CHANNEL,
                                                out_channels=OUT_CHANNEL,
                                                kernel_size=3,
                                                padding=1),
                    'batch_norm_2' : torch.nn.BatchNorm2d(num_features=OUT_CHANNEL),
                    'activation_2' : torch.nn.ReLU(True),
                }
            )
        )
        
    def forward(self, input, skip):
        skip = self.attention(input, skip)
        out = self.upsample(input)
        out = torch.concat([skip, out], dim=1) # dim = 1 for along channels
        out = self.pipe(out)
        return out
    
class Transformer_Encoder(torch.nn.Module):
    def __init__(self, EMBED_DIM = 512, num_heads=8):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=EMBED_DIM,
                                        num_heads=num_heads, batch_first=True)
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
    