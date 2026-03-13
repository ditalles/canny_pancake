"""Payment processing via Stripe for pancake print orders."""

import logging
import os

logger = logging.getLogger(__name__)

# Stripe keys from environment variables
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
PANCAKE_PRICE_CENTS = int(os.environ.get("PANCAKE_PRICE_CENTS", "500"))  # $5.00 default
CURRENCY = os.environ.get("PAYMENT_CURRENCY", "usd")


def is_configured():
    """Check if Stripe keys are set."""
    return bool(STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY)


def create_payment_intent(amount_cents=None):
    """Create a Stripe PaymentIntent for a pancake order.

    Args:
        amount_cents: Price in cents (defaults to PANCAKE_PRICE_CENTS)

    Returns:
        dict with 'client_secret' and 'payment_intent_id' on success,
        or 'error' message on failure.
    """
    if not is_configured():
        return {"error": "Payment not configured. Set STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY."}

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        amount = amount_cents or PANCAKE_PRICE_CENTS
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=CURRENCY,
            metadata={"product": "pancake_portrait"},
        )

        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "amount": amount,
            "currency": CURRENCY,
        }

    except Exception as e:
        logger.error("Stripe error: %s", e)
        return {"error": str(e)}


def verify_payment(payment_intent_id):
    """Verify that a PaymentIntent has been successfully paid.

    Args:
        payment_intent_id: Stripe PaymentIntent ID

    Returns:
        True if payment succeeded, False otherwise
    """
    if not is_configured():
        return False

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        return intent.status == "succeeded"

    except Exception as e:
        logger.error("Payment verification error: %s", e)
        return False


def get_price_display():
    """Return a formatted price string for the UI."""
    amount = PANCAKE_PRICE_CENTS / 100.0
    symbols = {"usd": "$", "eur": "€", "gbp": "£"}
    symbol = symbols.get(CURRENCY, CURRENCY.upper() + " ")
    return f"{symbol}{amount:.2f}"
