def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price with validation."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not (0 <= discount_percent <= 100):
        raise ValueError("Discount percent must be between 0 and 100")
    
    return price * (1 - discount_percent / 100)

def is_valid_email(email: str) -> bool:
    """Basic email validation check."""
    return "@" in email and "." in email.split("@")[-1]