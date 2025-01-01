if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import common.gridworld as gw

from collections import defaultdict

env = gw.GridWorld()

# V = {}
# for state in env.states() :
#     V[state] = 0
V = defaultdict(lambda: 0)

state = (1, 2)
print(V[state])
