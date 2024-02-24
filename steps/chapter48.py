if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ohzero

x, t = ohzero.get_spiral(train=True)
print(x.shape)
print(t.shape)


print(x[10], t[10])
print(x[110], t[110])