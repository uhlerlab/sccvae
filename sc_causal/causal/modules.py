import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import collections

class PTBEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)

        layer_sizes = [input_dim] + hidden_dims
        
        self.fc_hidden_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in,
                                n_out,
                                bias=True,
                            ),
                            self.act,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layer_sizes[:-1], layer_sizes[1:])
                    )
                ]
            )
        )

        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ptb):
        h = self.fc_hidden_layers(ptb)

        out = self.out_layer(h)
        return self.softmax(out)


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims):
        super().__init__()

        self.relu = nn.ReLU()

        self.epsilon = 1e-4

        self.encoder = FCLayers(
            n_in = input_dim, 
            n_out = hidden_dims[-1], 
            n_layers = len(hidden_dims), 
            n_hidden = hidden_dims[0]
        )


        self.mean_encoder = nn.Linear(hidden_dims[-1], z_dim)
        self.var_encoder = nn.Linear(hidden_dims[-1], z_dim)
        self.var_activation = torch.exp

    def forward(self, y):
        h = self.encoder(y)

        h_m = self.mean_encoder(h)
        h_v = self.var_activation(self.var_encoder(h)) + self.epsilon

        distrib = Normal(h_m, h_v.sqrt())
        latent = distrib.rsample()

        return distrib, latent

class Decoder_Gaussian(nn.Module):
    def __init__(self, output_dim, z_dim, hidden_dims):
        super().__init__()

        self.relu = nn.ReLU()

        self.stack = FCLayers(
            n_in = z_dim, 
            n_out = hidden_dims[-1], 
            n_layers = len(hidden_dims), 
            n_hidden = hidden_dims[0]
        )

        self.layer_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        y = self.stack(z)
        return self.layer_out(y)


class FCLayers(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_layers = 1,
        n_hidden = 128,
        bias = True,
        activation_fn = nn.LeakyReLU(0.2),
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.n_cat_list = []
        
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in,
                                n_out,
                                bias=bias,
                            ),
                            activation_fn,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x):
        one_hot_cat_list = []

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
