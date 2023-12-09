import unittest
from dezero import Variable
from dezero.square import square
import numpy as np

class SquareTest(unittest.TestCase) :
    def test_forward(self) :
        x = Variable(np.array(2.0))
        y = square(x)
        expected = Variable(np.array(4.0))
        self.assertEqual(y.data, expected.data)