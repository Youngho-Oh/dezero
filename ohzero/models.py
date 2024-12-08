from ohzero import layers
from ohzero import utils
from ohzero.functions import sigmoid

class Model(layers.Layer):
    def plot(self, *inputs, to_file='model.svg'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model) :
    def __init__(self, fc_output_sizes, activation=sigmoid) :
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = layers.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)