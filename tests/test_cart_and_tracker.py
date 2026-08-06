import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from cart import Cart  # noqa: E402
from tracker import CentroidTracker  # noqa: E402


def test_cart_add_and_total():
    cart = Cart(session_id="test")
    cart.add_by_class_name("bottle", confidence=0.9, min_confidence=0.5)
    cart.add_by_class_name("bottle", confidence=0.9, min_confidence=0.5)
    cart.add_by_class_name("apple", confidence=0.9, min_confidence=0.5)
    assert cart.item_count == 3
    assert cart.total_amount > 0


def test_cart_confidence_filter_rejects_low_confidence():
    cart = Cart(session_id="test")
    result = cart.add_by_class_name("bottle", confidence=0.2, min_confidence=0.5)
    assert result is None
    assert cart.item_count == 0


def test_cart_unknown_class_ignored():
    cart = Cart(session_id="test")
    result = cart.add_by_class_name("dinosaur", confidence=0.99, min_confidence=0.5)
    assert result is None


def test_cart_remove():
    cart = Cart(session_id="test")
    cart.add_by_class_name("apple", confidence=0.9, min_confidence=0.5)
    product_id = next(iter(cart.lines))
    cart.remove_one(product_id)
    assert cart.item_count == 0


def test_tracker_confirms_after_min_hits_not_before():
    tracker = CentroidTracker(min_hits_to_count=3)
    box = (10, 10, 50, 50)
    # frame 1: seen once -> not yet confirmed
    confirmed = tracker.update([("bottle", box, 0.9)])
    assert confirmed == []
    # frame 2: seen again (same box -> IoU 1.0) -> still not confirmed
    confirmed = tracker.update([("bottle", box, 0.9)])
    assert confirmed == []
    # frame 3: third consecutive hit -> confirmed exactly once
    confirmed = tracker.update([("bottle", box, 0.9)])
    assert len(confirmed) == 1
    assert confirmed[0].class_name == "bottle"


def test_tracker_does_not_double_count_same_item():
    tracker = CentroidTracker(min_hits_to_count=2)
    box = (10, 10, 50, 50)
    tracker.update([("bottle", box, 0.9)])
    tracker.update([("bottle", box, 0.9)])  # confirms here
    confirmed_again = tracker.update([("bottle", box, 0.9)])
    # already counted -> should not be returned a second time
    assert confirmed_again == []


def test_tracker_distinguishes_two_simultaneous_items():
    tracker = CentroidTracker(min_hits_to_count=1)
    box_a = (10, 10, 50, 50)
    box_b = (200, 200, 250, 250)
    confirmed = tracker.update([("bottle", box_a, 0.9), ("apple", box_b, 0.9)])
    assert {t.class_name for t in confirmed} == {"bottle", "apple"}
