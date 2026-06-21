from __future__ import annotations

from collections.abc import Callable


class GuardrailError(ValueError):
    """Raised by a guardrail to reject a message."""


# A guardrail is any callable str -> str that may raise GuardrailError.
# It can pass through, modify, or reject the text.
InputGuardrail = Callable[[str], str]
OutputGuardrail = Callable[[str], str]
