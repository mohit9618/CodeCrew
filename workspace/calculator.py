def add(a, b):
    return a + b


def subtract(a, b):    return a - b


def multiply(a, b):
    """Return the product of a and b.

    Supports integers, floats, and other numeric types that implement
    the multiplication operator.
    """
    return a * b


def divide(a, b):
    """Return the result of dividing *a* by *b*.

    Raises:
        ZeroDivisionError: If *b* is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
