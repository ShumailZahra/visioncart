-- VisionCart ERD, as described in the project proposal (Section 5)
-- Lean, speed-optimized schema for a localized CV application (no heavy
-- relational web-app modeling needed -- three tables cover the workflow).

CREATE TABLE product (
    product_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name        TEXT NOT NULL,
    visual_class_label  TEXT NOT NULL UNIQUE,   -- must match the YOLO model's class name
    price               DECIMAL(10, 2) NOT NULL,
    sku                 TEXT NOT NULL UNIQUE
);

CREATE TABLE cart_session (
    session_id     TEXT PRIMARY KEY,            -- uuid
    timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_amount   DECIMAL(10, 2) DEFAULT 0.00,
    status         TEXT DEFAULT 'active'        -- active | completed | abandoned
);

CREATE TABLE cart_items (
    session_id   TEXT NOT NULL REFERENCES cart_session(session_id),
    product_id   INTEGER NOT NULL REFERENCES product(product_id),
    quantity     INTEGER NOT NULL DEFAULT 1,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, product_id)
);

-- Example: recompute a session's running total (mirrors Cart.total_amount
-- in app/cart.py, which does this in memory for the demo)
-- UPDATE cart_session
-- SET total_amount = (
--     SELECT SUM(p.price * ci.quantity)
--     FROM cart_items ci JOIN product p ON p.product_id = ci.product_id
--     WHERE ci.session_id = cart_session.session_id
-- )
-- WHERE session_id = ?;
