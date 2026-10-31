import pytest
from calculator import add, subtract, multiply, divide, power


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(4, 3) == 12
    assert multiply(-2, 3) == -6


def test_divide():
    assert divide(10, 2) == 5
    assert divide(5, 2) == 2.5


def test_power():
    assert power(2, 3) == 8
    assert power(5, 0) == 1
    assert power(9, 0.5) == 3.0
    # Test negative exponent
    assert power(2, -2) == 0.25
