"""Paper-trading infrastructure: turn a registered strategy's monthly decision into
orders, safely, on a broker, with everything journaled and reproducible.

No strategy logic lives here (see the module docstrings of ``decide`` and ``planner``):
this package only knows how to ask a registered ``qr`` strategy for its target weights
as of a date, turn those into an order plan against a broker's current account, and
submit, journal, and report the result. The paper-only guard in :mod:`qr.live.safety`
cannot be switched off by configuration.
"""

from __future__ import annotations
