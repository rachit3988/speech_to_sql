"""Create a deterministic e-commerce demo database (Postgres or SQLite).

    python -m speech_to_sql.demo.seed --url sqlite:///./data/demo.db
    python -m speech_to_sql.demo.seed --url postgresql://user:pass@localhost/speech_demo --reset

Refuses to touch existing demo tables unless --reset is given; --if-missing makes an
existing dataset a no-op (useful for container start-up).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import random
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

TABLES = ["order_items", "orders", "customers", "products", "categories"]

DDL = [
    """CREATE TABLE categories (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    )""",
    """CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category_id INTEGER NOT NULL REFERENCES categories(id),
        unit_price NUMERIC(10, 2) NOT NULL,
        launched_on DATE NOT NULL
    )""",
    """CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        company TEXT,
        city TEXT NOT NULL,
        country TEXT NOT NULL,
        segment TEXT NOT NULL,
        signup_date DATE NOT NULL
    )""",
    """CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(id),
        order_date DATE NOT NULL,
        status TEXT NOT NULL,
        channel TEXT NOT NULL
    )""",
    """CREATE TABLE order_items (
        order_id INTEGER NOT NULL REFERENCES orders(id),
        line_no INTEGER NOT NULL,
        product_id INTEGER NOT NULL REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price NUMERIC(10, 2) NOT NULL,
        discount NUMERIC(4, 3) NOT NULL DEFAULT 0,
        PRIMARY KEY (order_id, line_no)
    )""",
    "CREATE INDEX orders_customer_idx ON orders (customer_id)",
    "CREATE INDEX orders_date_idx ON orders (order_date)",
    "CREATE INDEX order_items_product_idx ON order_items (product_id)",
]

COMMENTS = [
    "COMMENT ON TABLE categories IS 'Product categories'",
    "COMMENT ON TABLE products IS 'Catalog of products for sale'",
    "COMMENT ON COLUMN products.unit_price IS 'Current list price in USD'",
    "COMMENT ON TABLE customers IS 'People and businesses who have placed orders'",
    "COMMENT ON COLUMN customers.company IS 'Employer / account name; NULL for consumers'",
    "COMMENT ON COLUMN customers.segment IS 'One of: consumer, small_business, enterprise'",
    "COMMENT ON TABLE orders IS 'One row per order'",
    "COMMENT ON COLUMN orders.status IS 'One of: pending, shipped, delivered, cancelled, returned'",
    "COMMENT ON COLUMN orders.channel IS 'One of: web, mobile, retail'",
    "COMMENT ON TABLE order_items IS 'Order lines. Line revenue = quantity * unit_price * (1 - discount)'",
    "COMMENT ON COLUMN order_items.unit_price IS 'Price actually charged per unit in USD'",
    "COMMENT ON COLUMN order_items.discount IS 'Fractional discount, e.g. 0.100 = 10% off'",
]

CATEGORIES = ["Laptops", "Phones", "Audio", "Monitors", "Accessories", "Smart Home", "Cameras", "Gaming"]
PRODUCT_WORDS = {
    "Laptops": [
        "Aero 14",
        "Aero 16",
        "Forge Pro 15",
        "Forge Air 13",
        "Nimbus X1",
        "Nimbus X2",
        "Ridge 17",
        "Quill 12",
    ],
    "Phones": ["Pulse 9", "Pulse 9 Pro", "Pulse Mini", "Orbit S", "Orbit Ultra", "Vista Lite", "Vista Max"],
    "Audio": [
        "EchoBuds",
        "EchoBuds Pro",
        "Halo Headphones",
        "Halo Studio",
        "Boom Speaker",
        "Boom Mini",
        "Soundbar 500",
    ],
    "Monitors": [
        "ClearView 24",
        "ClearView 27",
        "ClearView 32 4K",
        "Arc Curve 34",
        "Arc Curve 49",
        "Pixel Pro 27",
    ],
    "Accessories": [
        "USB-C Hub",
        "Travel Charger",
        "Wireless Mouse",
        "Mechanical Keyboard",
        "Laptop Sleeve",
        "HDMI Cable",
        "Webcam HD",
        "Docking Station",
    ],
    "Smart Home": ["Glow Bulb", "Glow Strip", "Sentinel Cam", "Thermo Smart", "Hub Home", "Lock Smart"],
    "Cameras": ["Snap 200", "Snap 400", "Snap Action 4K", "Lens Pro Kit", "Tripod Carbon"],
    "Gaming": [
        "Raptor Controller",
        "Raptor Headset",
        "Arena Chair",
        "Arena Desk",
        "Quest Console",
        "Quest Console Slim",
    ],
}
PRICE_RANGE = {
    "Laptops": (799, 2799),
    "Phones": (399, 1399),
    "Audio": (49, 449),
    "Monitors": (179, 1299),
    "Accessories": (12, 249),
    "Smart Home": (19, 299),
    "Cameras": (149, 1499),
    "Gaming": (39, 599),
}
FIRST = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Susan",
    "Priya",
    "Arjun",
    "Wei",
    "Mei",
    "Carlos",
    "Sofia",
    "Ahmed",
    "Fatima",
    "Hiroshi",
    "Yuki",
    "Olga",
    "Ivan",
    "Chloe",
    "Liam",
    "Noah",
    "Emma",
    "Rahul",
    "Ananya",
    "Lucas",
    "Isabella",
]
LAST = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Martinez",
    "Lopez",
    "Wilson",
    "Anderson",
    "Taylor",
    "Thomas",
    "Moore",
    "Jackson",
    "Patel",
    "Sharma",
    "Chen",
    "Wang",
    "Kim",
    "Nguyen",
    "Tanaka",
    "Ivanova",
    "Muller",
    "Rossi",
    "Dubois",
    "Silva",
    "Khan",
    "Singh",
]
COMPANIES = [
    "Acme Corporation",
    "Globex Inc",
    "Initech",
    "Umbrella Health",
    "Stark Industries",
    "Wayne Enterprises",
    "Hooli",
    "Pied Piper",
    "Vandelay Industries",
    "Soylent Foods",
    "Cyberdyne Systems",
    "Tyrell Corp",
    "Wonka Industries",
    "Oceanic Airlines",
    "Massive Dynamic",
    "Northwind Traders",
]
CITIES = [
    ("New York", "United States"),
    ("San Francisco", "United States"),
    ("Austin", "United States"),
    ("Chicago", "United States"),
    ("Toronto", "Canada"),
    ("Vancouver", "Canada"),
    ("London", "United Kingdom"),
    ("Manchester", "United Kingdom"),
    ("Berlin", "Germany"),
    ("Munich", "Germany"),
    ("Paris", "France"),
    ("Mumbai", "India"),
    ("Bengaluru", "India"),
    ("Delhi", "India"),
    ("Tokyo", "Japan"),
    ("Osaka", "Japan"),
    ("Sydney", "Australia"),
    ("Melbourne", "Australia"),
    ("Sao Paulo", "Brazil"),
    ("Singapore", "Singapore"),
]

START = dt.date(2024, 1, 1)
END = dt.date(2026, 9, 30)


def generate(seed: int = 42) -> dict[str, list[tuple[Any, ...]]]:
    rng = random.Random(seed)
    categories = [(i, name) for i, name in enumerate(CATEGORIES, 1)]
    products = []
    pid = 0
    for cid, cname in categories:
        lo, hi = PRICE_RANGE[cname]
        for name in PRODUCT_WORDS[cname]:
            pid += 1
            price = Decimal(rng.randrange(lo, hi)) + Decimal("0.99")
            launched = START + dt.timedelta(days=rng.randrange(0, 700))
            products.append((pid, name, cid, price, launched))

    customers = []
    for cid in range(1, 401):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        city, country = rng.choice(CITIES)
        segment = rng.choices(["consumer", "small_business", "enterprise"], weights=[6, 3, 1])[0]
        company = None if segment == "consumer" else rng.choice(COMPANIES)
        signup = START + dt.timedelta(days=rng.randrange(0, (END - START).days - 30))
        email = f"{first}.{last}{cid}@example.com".lower()
        customers.append((cid, f"{first} {last}", email, company, city, country, segment, signup))

    orders, items = [], []
    oid = 0
    total_days = (END - START).days
    for day in range(total_days + 1):
        date = START + dt.timedelta(days=day)
        growth = 1 + day / total_days  # demand roughly doubles over the period
        seasonal = 1.6 if date.month in (11, 12) else 1.0
        for _ in range(_poisson(rng, 4.0 * growth * seasonal)):
            cust = rng.choice(customers)
            if cust[7] > date:
                continue
            oid += 1
            age = (END - date).days
            if age < 5:
                status = rng.choices(["pending", "shipped"], weights=[3, 2])[0]
            else:
                status = rng.choices(["delivered", "cancelled", "returned", "shipped"], weights=[86, 6, 5, 3])[0]
            channel = rng.choices(["web", "mobile", "retail"], weights=[5, 3, 2])[0]
            orders.append((oid, cust[0], date, status, channel))
            for line in range(1, rng.choice([1, 1, 1, 2, 2, 3, 4]) + 1):
                prod = rng.choice(products)
                qty = rng.choices([1, 2, 3, 5, 10], weights=[70, 15, 8, 4, 3])[0]
                if cust[6] == "enterprise":
                    qty *= rng.choice([1, 2, 5])
                discount = rng.choices(
                    [Decimal("0"), Decimal("0.05"), Decimal("0.10"), Decimal("0.20")], weights=[70, 15, 10, 5]
                )[0]
                items.append((oid, line, prod[0], qty, prod[3], discount))
    return {
        "categories": categories,
        "products": products,
        "customers": customers,
        "orders": orders,
        "order_items": items,
    }


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm; fine for small lambda.
    limit, k, p = pow(2.718281828459045, -lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class AlreadySeeded(Exception):
    pass


def seed_sqlite(path: str, data: dict[str, list[tuple[Any, ...]]], reset: bool) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if existing & set(TABLES):
            if not reset:
                raise AlreadySeeded(f"{path} already has demo tables; pass --reset to recreate them")
            for table in TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        for stmt in DDL:
            conn.execute(stmt)
        for table in reversed(TABLES):
            rows = [tuple(_sqlite_value(v) for v in row) for row in data[table]]
            placeholders = ", ".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        conn.commit()
    finally:
        conn.close()


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


async def seed_postgres(url: str, data: dict[str, list[tuple[Any, ...]]], reset: bool) -> None:
    import asyncpg

    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://", 1))
    try:
        existing = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename = ANY($1::text[])",
            TABLES,
        )
        async with conn.transaction():
            if existing:
                if not reset:
                    raise AlreadySeeded("demo tables already exist; pass --reset to recreate them")
                await conn.execute(f"DROP TABLE IF EXISTS {', '.join(TABLES)} CASCADE")
            for stmt in DDL + COMMENTS:
                await conn.execute(stmt)
            for table in reversed(TABLES):
                await conn.copy_records_to_table(table, records=data[table])
        await conn.execute("ANALYZE")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        except asyncpg.PostgresError as exc:  # needs privileges; fuzzy value search is optional
            print(f"note: pg_trgm not enabled ({exc.__class__.__name__}); fuzzy search will use ILIKE only")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="postgresql://... or sqlite:///path.db")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the demo tables")
    parser.add_argument("--if-missing", action="store_true", help="exit successfully if already seeded")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate(args.seed)
    try:
        if args.url.startswith("sqlite:///"):
            seed_sqlite(args.url.removeprefix("sqlite:///"), data, args.reset)
        elif args.url.startswith(("postgresql://", "postgres://", "postgresql+asyncpg://")):
            asyncio.run(seed_postgres(args.url, data, args.reset))
        else:
            raise SystemExit("--url must start with sqlite:/// or postgresql://")
    except AlreadySeeded as exc:
        if args.if_missing:
            print("demo data already present; nothing to do")
            return
        raise SystemExit(str(exc)) from exc
    counts = ", ".join(f"{t}={len(data[t])}" for t in reversed(TABLES))
    print(f"seeded {counts}")


if __name__ == "__main__":
    main()
