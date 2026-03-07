

def to_millions(value: float) -> float:
    """
    Convert a number to millions with 3 decimal places.
    Preserves the sign of the input.
    """
    result = round(abs(value) / 1_000_000, 3)
    return result if value > 0 else -result


def to_thousands(value: float) -> float:
    """
    Convert a number to thousands with 2 decimal places.
    Preserves the sign of the input.
    """
    result = round(abs(value) / 1_000, 2)
    return result if value > 0 else -result