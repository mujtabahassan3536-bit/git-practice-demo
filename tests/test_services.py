import pytest

from app.services import calculate_discount, is_valid_email


@pytest.mark.parametrize("price, discount, expected", [
    (100.0, 10.0, 90.0),
    (200.0, 50.0, 100.0),
    (50.0, 0.0, 50.0),
])
def test_calculate_discount_success(price, discount, expected):
    assert calculate_discount(price, discount) == expected

def test_calculate_discount_invalid_price():
    with pytest.raises(ValueError, match="Price cannot be negative"):
        calculate_discount(-10.0, 20.0)

def test_calculate_discount_invalid_percent():
    with pytest.raises(ValueError, match="Discount percent must be between 0 and 100"):
        calculate_discount(100.0, 150.0)

@pytest.mark.parametrize("email, expected", [
    ("user@example.com", True),
    ("user.name@domain.co.uk", True),
    ("invalid-email", False),
    ("user@domain", False),
])
def test_is_valid_email(email, expected):
    assert is_valid_email(email) == expected