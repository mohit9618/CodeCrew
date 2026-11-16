import unittest
from calculator import (
    add,
    subtract,
    multiply,
    divide,
    power,
    square,
    cube,
    square_root,
    factorial,
)


class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, -1), -2)


class TestSubtract(unittest.TestCase):
    def test_subtract_positive(self):
        self.assertEqual(subtract(5, 3), 2)

    def test_subtract_negative(self):
        self.assertEqual(subtract(-5, -3), -2)


class TestMultiply(unittest.TestCase):
    def test_multiply_positive(self):
        self.assertEqual(multiply(4, 5), 20)

    def test_multiply_by_zero(self):
        self.assertEqual(multiply(7, 0), 0)


class TestDivide(unittest.TestCase):
    def test_divide_positive(self):
        self.assertEqual(divide(10, 2), 5)

    def test_divide_by_one(self):
        self.assertEqual(divide(9, 1), 9)


class TestPower(unittest.TestCase):
    def test_power_positive(self):
        self.assertEqual(power(2, 3), 8)

    def test_power_zero_exponent(self):
        self.assertEqual(power(5, 0), 1)


class TestSquare(unittest.TestCase):
    def test_square_positive(self):
        self.assertEqual(square(4), 16)

    def test_square_negative(self):
        self.assertEqual(square(-3), 9)


class TestCube(unittest.TestCase):
    def test_cube_positive(self):
        self.assertEqual(cube(3), 27)

    def test_cube_zero(self):
        self.assertEqual(cube(0), 0)


class TestSquareRoot(unittest.TestCase):
    def test_square_root_positive(self):
        self.assertEqual(square_root(9), 3.0)

    def test_square_root_zero(self):
        self.assertEqual(square_root(0), 0.0)

    def test_square_root_negative_raises(self):
        with self.assertRaises(ValueError):
            square_root(-4)


class TestFactorial(unittest.TestCase):
    """Unit tests for the ``factorial`` function."""

    def test_factorial_of_0(self):
        """0! should be 1."""
        self.assertEqual(factorial(0), 1)

    def test_factorial_of_1(self):
        """1! should be 1."""
        self.assertEqual(factorial(1), 1)

    def test_factorial_of_5(self):
        """5! should be 120."""
        self.assertEqual(factorial(5), 120)

    def test_factorial_of_7(self):
        """7! should be 5040."""
        self.assertEqual(factorial(7), 5040)

    def test_factorial_negative_raises(self):
        """Factorial of a negative number should raise ``ValueError``."""
        with self.assertRaises(ValueError):
            factorial(-3)

    def test_factorial_non_integer_raises(self):
        """Factorial of a non‑integer should raise ``TypeError``."""
        with self.assertRaises(TypeError):
            factorial(3.5)


if __name__ == "__main__":
    unittest.main()
