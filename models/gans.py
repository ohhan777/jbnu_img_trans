
import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block =[nn.ReflectionPad2d(1),
                     nn.Conv2d(dim, dim, kernel_size=3, padding_mode='reflect', bias=True),
                     nn.InstanceNorm2d(dim),
                     nn.ReLU(True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(dim, dim, kernel_size=3, padding_mode='reflect', bias=True),
                     nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
             

class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_features=64, num_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, num_features, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(num_features),
                 nn.ReLU(True)]
        
        num_downsampling = 2

        for i in range(num_downsampling):   # add downsampling layers
            factor = 2 ** i
            model += [nn.Conv2d(num_features*factor, num_features*factor*2, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(num_features*factor*2),
                      nn.ReLU(True)]
        
        factor = 2 ** num_downsampling
        
        for i in range(num_blocks):
            model += [ResnetBlock(num_features*factor)]

        for i in range(num_downsampling):   # add upsampling layers
            factor = 2 ** (num_downsampling - i)
            model += [nn.ConvTranspose2d(num_features*factor, int(num_features*factor/2), kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=True),
                      nn.InstanceNorm2d(int(num_features*factor/2)),
                      nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(num_features, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        init_weights(self.model, init_type='normal', init_gain=0.02)
               
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, num_features=64, num_layers=3):
        super(Discriminator, self).__init__()
        kernel_size = 4
        padding = 1
        model = [nn.Conv2d(input_nc, num_features, kernel_size=kernel_size, stride=2, padding=padding),
                 nn.LeakyReLU(0.2, True)]
        factor = 1
        prev_factor = 1
        for n in range(1, num_layers):         # gradually increase the number of filters
            prev_factor = factor
            factor = min(2 ** n, 8)
            model += [nn.Conv2d(num_features*prev_factor, num_features*factor, kernel_size=kernel_size, stride=2, padding=padding, bias=True),
                      nn.InstanceNorm2d(num_features*factor),
                      nn.LeakyReLU(0.2, True)]
            
        prev_factor = factor
        factor = min(2 ** num_layers, 8)
        model += [nn.Conv2d(num_features*prev_factor,num_features*factor, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
                  nn.InstanceNorm2d(num_features*factor),
                  nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(num_features*factor, 1, kernel_size=kernel_size, stride=1, padding=padding)]    # output 1 channel prediction map
        self.model = nn.Sequential(*model)

        init_weights(self.model, init_type='normal', init_gain=0.02)
               
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print('Test')
    G = Generator()
    D = Discriminator()
    hr_img = torch.rand(4,1,512,512)
    print(G)
    print(D)
    fake_lr_img = G(hr_img)
    print(fake_lr_img.shape)
    p = D(fake_lr_img)
    print(p.shape)
    with torch.no_grad():
        print("Mean=", p.mean().item())