import math


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b


def power(a, b):
    """Return a raised to the power of b.

    Args:
        a (int or float): The base number.
        b (int or float): The exponent.

    Returns:
        int or float: The result of a ** b.
    """
    return a ** b



def cube(a):
    """Return the cube of a number.

    Args:
        a (int or float): The number to cube.

    Returns:
        int or float: The result of a ** 3.
    """
    return a ** 3


def square_root(a: float) -> float:
    """Return the square root of a number.

    Args:
        a (int or float): The number to compute the square root of. Must be non‑negative.

    Raises:
        ValueError: If ``a`` is negative.

    Returns:
        float: The square root of ``a``.
    """
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number")
    return math.sqrt(a)


def factorial(n):
    """Return the factorial of a non‑negative integer ``n``.

    Args:
        n (int): A non‑negative integer whose factorial is to be computed.

    Returns:
        int: The factorial of ``n``.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative.
    """
    if not isinstance(n, int):
        raise TypeError("factorial() only accepts integer arguments")
    if n < 0:
        raise ValueError("factorial() not defined for negative values")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
