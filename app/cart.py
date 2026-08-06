"""
cart.py
-------
In-memory implementation of the ERD described in the proposal:
  Product (catalog)  <--  Cart_Items (bridge, qty)  --  Cart/Session

A production build would back this with a real DB (SQLite/Postgres) via
the schema in /docs/erd.sql. For the demo, an in-memory session object is
enough to prove the workflow end-to-end and is what `app.py` binds to
Streamlit's `session_state` so each visitor gets their own cart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from product_catalog import Product, get_product


@dataclass
class CartLine:
    product: Product
    quantity: int = 0


@dataclass
class Cart:
    session_id: str
    created_at: float = field(default_factory=time.time)
    lines: dict[str, CartLine] = field(default_factory=dict)  # keyed by product_id
    log: list[str] = field(default_factory=list)

    def add_by_class_name(self, class_name: str, confidence: float, min_confidence: float = 0.5) -> Product | None:
        """Confidence-threshold filtering, per the proposal's requirement."""
        if confidence < min_confidence:
            return None
        product = get_product(class_name)
        if product is None:
            return None
        line = self.lines.setdefault(product.product_id, CartLine(product=product, quantity=0))
        line.quantity += 1
        self.log.append(
            f"[{time.strftime('%H:%M:%S')}] + {product.display_name} "
            f"(conf={confidence:.2f}) -> qty {line.quantity}"
        )
        return product

    def remove_one(self, product_id: str) -> None:
        if product_id in self.lines:
            self.lines[product_id].quantity -= 1
            if self.lines[product_id].quantity <= 0:
                del self.lines[product_id]

    @property
    def total_amount(self) -> float:
        return sum(line.product.price * line.quantity for line in self.lines.values())

    @property
    def item_count(self) -> int:
        return sum(line.quantity for line in self.lines.values())

    def as_rows(self):
        return [
            {
                "Product": line.product.display_name,
                "SKU": line.product.sku,
                "Qty": line.quantity,
                "Unit Price (PKR)": line.product.price,
                "Subtotal (PKR)": line.product.price * line.quantity,
            }
            for line in sorted(self.lines.values(), key=lambda l: l.product.display_name)
        ]

    def clear(self):
        self.lines.clear()
        self.log.clear()
