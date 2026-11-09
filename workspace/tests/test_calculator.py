import unittest

# Import the functions to be tested from the calculator module
from calculator import add, subtract, multiply, divide, power, square, cube


class TestCalculator(unittest.TestCase):
    # Existing tests (if any) would be here
    # ...

    def test_cube_positive(self):
        """Test that cube returns the correct result for a positive integer."""
        self.assertEqual(cube(3), 27)

    def test_cube_negative(self):
        """Test that cube returns the correct result for a negative integer."""
        self.assertEqual(cube(-2), -8)

    def test_cube_zero(self):
        """Test that cube returns zero when the input is zero."""
        self.assertEqual(cube(0), 0)


if __name__ == "__main__":
    unittest.main()
