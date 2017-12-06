import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.init import kaiming_normal, calculate_gain

def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity == 'leaky_relu':
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)

class w_scale_layer(nn.Module):
    def __init__(self, input_x):
        super(w_scale_layer, self).__init__()
        self.input_x = input_x
        self.c = (torch.mean(self.input_x.weight.data**2))**0.5
        self.input_x.weight.data.copy_(self.input_x.weight.data/self.c)
        self.bias = None
        if self.input_x.bias is not None:
            self.bias = self.input_x.bias
            self.input_x.bias = None
        
    def forward(self, input_x):
        input_x = self.c*input_x
        if self.bias is not None:
            input_x = input_x+self.bias.view(1, self.bias.size()[0],1,1)
        return input_x

class pixel_norm_layer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(pixel_norm_layer, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, input_x):
        input_x = input_x / (torch.mean(input_x**2, dim=1, keepdim=True) + self.epsilon)**0.5
        return input_x
        
class instance_norm_layer(nn.Module):
    def __init__(self, input_x, epsilon=1e-4):
        super(instance_norm_layer, self).__init__()
        self.input_x = input_x
        self.epsilon = epsilon
        self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.bias = None

        if self.input_x.bias is not None:
            self.bias = self.input_x.bias
            self.input_x.bias = None

    def forward(self, x):
        x = x - mean(x, axis=range(1, len(x.size())))
        x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.epsilon))
        x = x * self.gain
        if self.bias is not None:
            x += self.bias
        return x

def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, axis=ax, **kwargs)
    return tensor

class minibatch_stddev_concatlayer(nn.Module):
    def __init__(self, averaging='all'):
        super(minibatch_stddev_concatlayer, self).__init__()
        self.averaging = averaging.lower()
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = shape.copy()
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, keepdim=True)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

class concate_layer(nn.Module):
    def __init__(self):
        super(concate_layer, self).__init__()
        
    def forward(self, head, tail):
        return torch.cat([head, tail], 1)

class reshape_layer(nn.Module):
    def __init__(self, new_shape):
        super(reshape_layer, self).__init__()
        self.new_shape = new_shape  # not include minibatch dimension

    def forward(self, input_x):
        return input_x.view(-1, *self.new_shape)

class g_select_layer(nn.Module):
    def __init__(self, pre, chain, to_rgb):
        super(g_select_layer, self).__init__()
        assert len(chain) == len(to_rgb)
        self.pre = pre
        self.chain = chain
        self.to_rgb = to_rgb
        self.num = len(self.chain)
        
    def forward(self, input_x, input_y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.num
        if input_y is not None:
            pass
        min_level = int(np.floor(cur_level-1))
        max_level = int(np.ceil(cur_level-1))
        min_level_weight = int(cur_level+1)-cur_level
        max_level_weight = cur_level-int(cur_level)
        
        _from, _to, _step = 0, max_level+1, 1
        if self.pre is not None:        
            input_x = self.pre(input_x)
        
        out = {}
        for level in range(_from, _to, _step):
            if level == insert_y_at:
                input_x = self.chain[level](input_x, input_y)
            else:
                input_x = self.chain[level](input_x)
                
            if level == min_level:
                out['min_level'] = self.to_rgb[level](input_x)
            if level == max_level:
                out['max_level'] = self.to_rgb[level](input_x)
                input_x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                        out['max_level'] * max_level_weight
        return input_x
        
class d_select_layer(nn.Module):
    def __init__(self, pre, chain, from_rgb):
        super(d_select_layer, self).__init__()
        assert len(chain) == len(from_rgb)
        self.pre = pre
        self.chain = chain
        self.from_rgb = from_rgb
        self.num = len(self.chain)
    def forward(self, input_x, input_y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.num
        if input_y is not None:
            pass
        max_level = int(np.floor(self.num-cur_level))
        min_level = int(np.ceil(self.num-cur_level))
        min_level_weight = int(cur_level+1)-cur_level
        max_level_weight = cur_level - int(cur_level)
        
        _from, _to, _step = min_level+1, self.num, 1

        if self.pre is not None:
            input_x = self.pre(input_x)

        if max_level == min_level:
            input_x = self.from_rgb[max_level](input_x)
            if max_level == insert_y_at:
                input_x = self.chain[max_level](input_x, input_y)
            else:
                input_x = self.chain[max_level](input_x)
        else:
            out = {}
            tmp = self.from_rgb[max_level](input_x)
            if max_level == insert_y_at:
                tmp = self.chain[max_level](tmp, input_y)
            else:
                tmp = self.chain[max_level](tmp)
                
            out['max_level'] = tmp
            out['min_level'] = self.from_rgb[min_level](input_x)
            
            input_x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                                out['max_level'] * max_level_weight
            # ---------------------------------
            if min_level == insert_y_at:
                input_x = self.chain[min_level](input_x, input_y)
            else:
                input_x = self.chain[min_level](input_x)

        for level in range(_from, _to, _step):
            if level == insert_y_at:
                input_x = self.chain[level](input_x, input_y)
            else:
                input_x = self.chain[level](input_x)
                
        return input_x

def resize_activations(v, so):
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so) and si[0] == so[0]
    
    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = (si[2] // so[2], si[3] // so[3])
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

    if si[2] < so[2]: 
        assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]  # currently only support this case
        v = F.upsample(v, scale_factor=so[2]//si[2], mode='nearest')

    # Increase feature maps.
    if si[1] < so[1]:
        z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
        v = torch.cat([v, z], 1)
    return v

class g_drop_layer(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(g_drop_layer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1
            
        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd
