"""
product_catalog.py
--------------------
VisionCart product catalog.

In a real deployment this table would be populated from the retailer's POS
system and `visual_class_label` would map to a class in a *custom-trained*
YOLO model fine-tuned on that retailer's own SKUs (see /training in this
repo for the fine-tuning pipeline).

For this reference implementation / demo we ship YOLOv8's stock COCO
weights (no GPU / custom dataset required to run it), so the catalog below
maps a curated subset of the 80 COCO classes onto plausible retail products
with PKR prices. Swap this dict (and the trained weights in `detector.py`)
for your own SKUs and it becomes a real store's catalog.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    product_id: str
    display_name: str
    price: float
    sku: str
    visual_class_label: str  # must match the underlying model's class name


# COCO class name -> Product
CATALOG: dict[str, Product] = {
    "bottle": Product("P001", "Mineral Water Bottle", 80.0, "SKU-0001", "bottle"),
    "cup": Product("P002", "Disposable Coffee Cup", 60.0, "SKU-0002", "cup"),
    "wine glass": Product("P003", "Glass Tumbler", 250.0, "SKU-0003", "wine glass"),
    "banana": Product("P004", "Banana (1 dozen)", 120.0, "SKU-0004", "banana"),
    "apple": Product("P005", "Apple (1 kg)", 300.0, "SKU-0005", "apple"),
    "orange": Product("P006", "Orange (1 kg)", 220.0, "SKU-0006", "orange"),
    "sandwich": Product("P007", "Packaged Sandwich", 350.0, "SKU-0007", "sandwich"),
    "cake": Product("P008", "Slice of Cake", 400.0, "SKU-0008", "cake"),
    "donut": Product("P009", "Donut", 150.0, "SKU-0009", "donut"),
    "book": Product("P010", "Notebook / Book", 200.0, "SKU-0010", "book"),
    "cell phone": Product("P011", "Phone Accessory Pack", 999.0, "SKU-0011", "cell phone"),
    "laptop": Product("P012", "Laptop Sleeve", 1500.0, "SKU-0012", "laptop"),
    "keyboard": Product("P013", "USB Keyboard", 1800.0, "SKU-0013", "keyboard"),
    "mouse": Product("P014", "Wireless Mouse", 1200.0, "SKU-0014", "mouse"),
    "remote": Product("P015", "TV Remote", 700.0, "SKU-0015", "remote"),
    "scissors": Product("P016", "Scissors", 180.0, "SKU-0016", "scissors"),
    "toothbrush": Product("P017", "Toothbrush", 90.0, "SKU-0017", "toothbrush"),
    "handbag": Product("P018", "Handbag", 3500.0, "SKU-0018", "handbag"),
    "backpack": Product("P019", "Backpack", 2800.0, "SKU-0019", "backpack"),
    "umbrella": Product("P020", "Umbrella", 650.0, "SKU-0020", "umbrella"),
    "fork": Product("P021", "Fork (set)", 250.0, "SKU-0021", "fork"),
    "knife": Product("P022", "Kitchen Knife", 400.0, "SKU-0022", "knife"),
    "spoon": Product("P023", "Spoon (set)", 250.0, "SKU-0023", "spoon"),
    "bowl": Product("P024", "Bowl", 300.0, "SKU-0024", "bowl"),
    "teddy bear": Product("P025", "Teddy Bear Toy", 1200.0, "SKU-0025", "teddy bear"),
    "clock": Product("P026", "Wall Clock", 1100.0, "SKU-0026", "clock"),
    "vase": Product("P027", "Decorative Vase", 900.0, "SKU-0027", "vase"),
    "hair drier": Product("P028", "Hair Dryer", 2200.0, "SKU-0028", "hair drier"),
    "toaster": Product("P029", "Toaster", 3200.0, "SKU-0029", "toaster"),
}


def get_tracked_class_names() -> list[str]:
    """COCO class names VisionCart will recognise as sellable items."""
    return list(CATALOG.keys())


def get_product(class_name: str) -> Product | None:
    return CATALOG.get(class_name)
