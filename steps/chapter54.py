if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero import test_mode
import ohzero.functions as F

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode() :
    y = F.dropout(x)
    print(y)