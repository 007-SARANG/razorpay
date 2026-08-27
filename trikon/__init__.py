"""Trikon -- three-way settlement reconciliation controller.

Reconciles a merchant's own order ledger against Razorpay settlement recon rows
against bank statement credits, reports a measured match rate, and produces an honest
list of the records it refused to resolve.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
