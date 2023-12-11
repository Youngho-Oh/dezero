if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import unittest
from ohzero import Variable
from ohzero.square import square
import numpy as np

# test using "python3 -m unittest test.py"

def numerical_diff(f, x, eps=1e-4) :
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase) :
    def test_forward(self) :
        x = Variable(np.array(2.0))
        y = square(x)
        expected = Variable(np.array(4.0))
        self.assertEqual(y.data, expected.data)
    
    def test_backward(self) :
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self) :
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)