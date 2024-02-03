from ohzero import layers
from ohzero import utils

class Model(layers.Layer):
    def plot(self, *inputs, to_file='model.svg'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)