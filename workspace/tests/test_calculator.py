from calculator import add, subtract, multiply, divide


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_subtract_negative():
    assert subtract(3, 5) == -2


def test_multiply_positive():
    assert multiply(4, 5) == 20


def test_multiply_negative():
    assert multiply(-3, 6) == -18


def test_multiply_zero():
    assert multiply(0, 10) == 0
    assert multiply(7, 0) == 0


def test_divide_positive():
    assert divide(10, 2) == 5


def test_divide_negative():
    assert divide(-9, 3) == -3


def test_divide_zero_divisor():
    import pytest
    with pytest.raises(ZeroDivisionError):
        divide(5, 0)
