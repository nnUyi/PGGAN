# -*- coding: utf-8 -*-
from ops import *

def g_conv(input_x, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
    to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layer = input_x
    layer +=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    # he kaiming's initializer
    # he_init(layer, nonlinearity='conv2d', param=None)
    he_init(layer[-1], init, param)
    
    if use_wscale:
        layer += [w_scale_layer(layer[-1])]

    #layer += [nonlinearity]

    if use_batchnorm:
        layer += [nn.BatchNorm2d(out_channels)]
    
    layer += [nonlinearity]
    
    if use_pixelnorm:
        layer += [pixel_norm_layer()]
    if to_sequential:
        return nn.Sequential(*layer)
        # what does * mean here
    else:
        return layer

def to_from_rgb(input_x, in_channels, out_channels, nonlinearity, init, param=None, use_wscale=True, to_sequential=True):
    layer = input_x
    layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
    
    he_init(layer[-1], init, param)
    
    if use_wscale:
        layer += [w_scale_layer(layer[-1])]
    if not(nonlinearity=='linear'):
        layer += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layer)
    else:
        return layer

class generator(nn.Module):
    def __init__(self, 
                num_channels        = 3,        # Overridden based on dataset.
                resolution          = 32,       # Overridden based on dataset.
                #label_size         = 0,        # Overridden based on dataset.
                feature_map_base    = 4096,
                feature_map_decay   = 1.0,
                feature_map_max     = 512,
                latent_size         = None,
                normalize_latents   = True,
                use_wscale          = True,
                use_pixelnorm       = True,
                use_leakyrelu       = True,
                use_batchnorm       = False,
                tanh_at_end         = None
                ):
        
        super(generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution

        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.feature_map_max = feature_map_max
        self.latent_size = latent_size
        self.normalize_latents = normalize_latents
        self.use_wscale = use_wscale
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.use_batchnorm = use_batchnorm
        self.tanh_at_end = tanh_at_end
        
        R = int(np.log2(resolution))
        if self.latent_size is None:
            # self.latent_size = self.feature_map_max
            self.latent_size = self.get_num_fmaps(0)
        
        slope = 0.2
        act = nn.LeakyReLU(negative_slope=slope) if self.use_leakyrelu else nn.ReLU()
        init_act = 'leaky_relu' if self.use_leakyrelu else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_init_act = 'tanh' if self.tanh_at_end else 'linear'

        pre = None
        net_layers = nn.ModuleList()
        rgb_layer = nn.ModuleList()
        layer = []

        if self.normalize_latents:
            pre = pixel_norm_layer()

        #if self.label_size:
        #    layer += [concate_layer()]

        # first block
        layer += [reshape_layer([self.latent_size, 1, 1])]
        """g_conv(input_x, 
                in_channels, 
                out_channels, 
                kernel_size, 
                padding, 
                nonlinearity, 
                init, 
                param=None,
                to_sequential=True, 
                use_wscale=True, 
                use_batchnorm=False, 
                use_pixelnorm==True):
        """
        layer = g_conv(layer, self.latent_size, self.get_num_fmaps(1), 4, 3, act, init_act, slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
        net = g_conv(layer, latent_size, self.get_num_fmaps(1), 3, 1, act, init_act, slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
        
        net_layers.append(net)
        # to_rgb layer
        rgb_layer.append(to_from_rgb([], self.get_num_fmaps(1), self.num_channels, output_act, output_init_act, None, True, self.use_wscale))  

        for r in range(2, R):  # following blocks
            in_channels = self.get_num_fmaps(r-1)
            out_channels = self.get_num_fmaps(r)
            # upsample
            layer = [nn.Upsample(scale_factor=2, mode='nearest')]
            #layer = [nn.UpsamplingNearest2d(scale_factor=2)]
            layer = g_conv(layer, in_channels, out_channels, 3, 1, act, init_act, slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            net = g_conv(layer, out_channels, out_channels, 3, 1, act, init_act, slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            net_layers.append(net)
            # to_rgb layer
            rgb_layer.append(to_from_rgb([], out_channels, self.num_channels, output_act, output_init_act, None, True, self.use_wscale))  

        self.output_layer = g_select_layer(pre, net_layers, rgb_layer)

    def get_num_fmaps(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.feature_map_max)
        
    def forward(self, input_x, input_y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(input_x, input_y, cur_level, insert_y_at)


def d_conv(input_x, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None,to_sequential=True, use_wscale=True, use_gdrop=True, use_instance_norm=False, gdrop_param=dict()):
    layer = input_x
    if use_gdrop:
       layer += [g_drop_layer(**gdrop_param)]

    layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layer[-1], init, param)
    
    if use_wscale:
        layer += [w_scale_layer(layer[-1])]

    layer += [nonlinearity]

    if use_instance_norm:
        layer += [instance_norm_layer()]

    #layer += [nonlinearity]
    
    if to_sequential:
        return nn.Sequential(*layer)
    else:
        return layer

class discriminator(nn.Module):
    def __init__(self,
                 num_channels    = 3,        # Overridden based on dataset.
                 resolution      = 32,       # Overridden based on dataset.
               feature_map_base  = 4096,
               feature_map_decay = 1.0,
                 feature_map_max = 256,
                 mbstat_avg      = 'all',
                 mbdisc_kernels  = None,
                 use_wscale      = True,
                 use_gdrop       = True,
                 use_layernorm   = False,
                 sigmoid_at_end  = False):
        super(discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.feature_map_max = feature_map_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end

        R = int(np.log2(resolution))
        slope = 0.2
        act = nn.LeakyReLU(negative_slope = slope)
        init_act = 'leaky_relu'
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_init_act = 'sigmoid' if self.sigmoid_at_end else 'linear'
        
        gdrop_strength = 0.0
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        net_layers = nn.ModuleList()
        rgb_layer = nn.ModuleList()
        pre = None
        layer = []
    
        rgb_layer.append(to_from_rgb([], self.num_channels, self.get_num_fmaps(R-1), act, init_act, slope, True, self.use_wscale))

        for r in range(R-1, 1, -1):
            in_channels = self.get_num_fmaps(r)
            out_channels = self.get_num_fmaps(r-1)
            
            layer = d_conv([], in_channels, in_channels, 3, 1, act, init_act, slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            layer = d_conv(layer, in_channels, out_channels, 3, 1, act, init_act, slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            layer += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            
            net_layers.append(nn.Sequential(*layer))
            
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            rgb_layer_ = to_from_rgb([], self.num_channels, out_channels, act, init_act, slope, True, self.use_wscale)
            rgb_layer.append(rgb_layer_)

        layer = []
        in_channels = self.get_num_fmaps(1)
        out_channels = self.get_num_fmaps(1)
        
        if self.mbstat_avg is not None:
            layer += [minibatch_stddev_concatlayer(averaging=self.mbstat_avg)]
            in_channels += 1
        
        layer = d_conv(layer, in_channels, out_channels, 3, 1, act, init_act, slope, False, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        layer = d_conv(layer, out_channels, self.get_num_fmaps(0), 4, 0, act, init_act, slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

        #if self.mbdisc_kernels:
        #    layer += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]
        
        net_layers.append(to_from_rgb(layer, self.get_num_fmaps(0), out_channels, output_act, output_init_act, None, True, self.use_wscale))

        self.output_layer = d_select_layer(pre, net_layers, rgb_layer)

    def get_num_fmaps(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.feature_map_max)

    def forward(self, input_x, input_y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(input_x, input_y, cur_level, insert_y_at)

if __name__=='__main__':
    print('test in model')
    G = generator(latent_size=512)
    D = discriminator()
    print(G)
    print(D)
