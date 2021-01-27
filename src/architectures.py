import torch.nn as nn


def get_smooth_layers(dim_encoding, n_layers_encoder, n_layers_decoder, n_neurons_in=115):
    reduction_factor = (dim_encoding / n_neurons_in) ** (1 / n_layers_encoder)
    augmentation_factor = (n_neurons_in / dim_encoding) ** (1 / n_layers_decoder)

    hidden_layers = [int(n_neurons_in * (reduction_factor ** (i + 1))) for i in range(n_layers_encoder - 1)] + \
                    [dim_encoding] + \
                    [int(dim_encoding * (augmentation_factor ** (i + 1))) for i in range(n_layers_decoder - 1)]

    return hidden_layers


class SimpleAutoencoder(nn.Module):
    def __init__(self, activation_function, hidden_layers, verbose=False):
        super(SimpleAutoencoder, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, n_neurons_in, bias=True))

        if verbose:
            print(self.seq)

    def forward(self, x):
        return self.seq(x)


class BinaryClassifier(nn.Module):
    def __init__(self, activation_function, hidden_layers, verbose=False):
        super(BinaryClassifier, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, 1, bias=True))
        self.seq.add_module('sigmoid', nn.Sigmoid())

        if verbose:
            print(self.seq)

    def forward(self, x):
        return self.seq(x)


class NormalizingBinaryClassifier(nn.Module):
    def __init__(self, activation_function, hidden_layers, sub, div, verbose=False):
        super(NormalizingBinaryClassifier, self).__init__()
        self.sub = nn.Parameter(sub, requires_grad=False)
        self.div = nn.Parameter(div, requires_grad=False)
        self.bc = BinaryClassifier(activation_function, hidden_layers, verbose)

    def forward(self, x):
        normalized_x = (x - self.sub) / self.div
        return self.bc(x)

