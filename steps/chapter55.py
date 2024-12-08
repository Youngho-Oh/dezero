if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.utils import get_conv_outsize

H, W = 4, 4     # input size
KH, KW = 3, 3   # Kernal size
SH, SW = 1, 1   # stride
PH, PW = 1, 1   # padding

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)