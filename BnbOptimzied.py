import pandas as pd
import numpy as np
import sys
import os


# ==============================
# HAVERSINE DISTANCE (VECTORIZED)
# ==============================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(np.array(lat1, dtype=float))
    lon1 = np.radians(np.array(lon1, dtype=float))
    lat2 = np.radians(np.array(lat2, dtype=float))
    lon2 = np.radians(np.array(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ==============================
# AUTO-DETECT COLUMNS
# ==============================

COLUMN_ALIASES = {
    "city":                 ["City", "city", "city_name", "CityName"],
    "room_type":            ["Room Type", "room_type", "roomtype", "RoomType"],
    "property_type":        ["Property Type", "property_type", "propertytype"],
    "bedrooms":             ["Bedrooms", "bedrooms", "bedroom_count"],
    "bathrooms":            ["Bathrooms", "bathrooms", "bathroom_count"],
    "latitude":             ["Latitude", "latitude", "lat"],
    "longitude":            ["Longitude", "longitude", "lon", "lng"],
    "review_scores_rating": ["Review Scores Rating", "review_scores_rating", "Rating",
                             "rating", "review_score", "ReviewScoresRating"],
    "number_of_reviews":    ["Number of Reviews", "number_of_reviews", "reviews",
                             "review_count", "NumberOfReviews"],
    "price":                ["Price", "price", "nightly_price", "daily_price"],
    "amenities":            ["Amenities", "amenities"],
    "neighbourhood":        ["Neighbourhood Cleansed", "Neighbourhood", "neighbourhood_cleansed",
                             "neighbourhood", "Neighborhood", "neighborhood",
                             "Neighbourhood Group Cleansed"],
}

def detect_delimiter(path):
    """Sniff the delimiter from the first line of the file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","

def resolve_columns(actual_columns):
    mapping = {}
    actual_lower = {c.lower().strip(): c for c in actual_columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in actual_columns:
                mapping[canonical] = alias
                break
            if alias.lower().strip() in actual_lower:
                mapping[canonical] = actual_lower[alias.lower().strip()]
                break
    return mapping


# ==============================
# LOAD CITY DATA — CHUNKED
# ==============================

def load_city_data(path, city):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place 'airbnb-listings.csv' in the same folder as this script,\n"
            "or pass the full path as a CLI argument:  python airbnb_analyzer.py /path/to/file.csv"
        )

    # Auto-detect delimiter (comma vs semicolon)
    sep = detect_delimiter(path)
    print(f"  Detected delimiter: '{sep}'")

    # Peek at headers only
    header_df = pd.read_csv(path, nrows=0, sep=sep)
    col_map = resolve_columns(header_df.columns.tolist())

    required = ["city", "room_type", "property_type", "bedrooms", "bathrooms",
                "latitude", "longitude", "review_scores_rating",
                "number_of_reviews", "price"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(
            f"Could not find required columns: {missing}\n"
            f"Columns detected in file: {header_df.columns.tolist()}"
        )

    use_cols = list(col_map.values())
    city_col = col_map["city"]

    print("  Reading file in chunks (optimised for large files)…")
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep=sep,
        usecols=use_cols,
        chunksize=100_000,
        engine="c",
        on_bad_lines="skip",
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk[city_col] = chunk[city_col].astype(str).str.strip()
        match = chunk[chunk[city_col].str.lower() == city.strip().lower()]
        if not match.empty:
            chunks.append(match)

    print(f"  Scanned {total_rows:,} total rows.")

    if not chunks:
        raise ValueError(
            f"No listings found for city '{city}'.\n"
            "City names are case-insensitive but must match exactly otherwise."
        )

    df = pd.concat(chunks, ignore_index=True)

    # Rename to canonical names
    reverse_map = {v: k for k, v in col_map.items()}
    df = df.rename(columns=reverse_map)

    # Clean price — handles "$1,234.00" and plain "1234"
    df["price"] = df["price"].astype(str).str.replace(r"[\$,\s]", "", regex=True)
    df = df[df["price"].str.match(r"^\d+(\.\d+)?$", na=False)]
    df["price"] = df["price"].astype(float)
    df = df[df["price"] > 0]

    # Numeric coercions
    for col in ["bedrooms", "bathrooms", "latitude", "longitude",
                "review_scores_rating", "number_of_reviews"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["bedrooms", "bathrooms", "latitude", "longitude",
                           "review_scores_rating", "number_of_reviews", "price"])

    # Amenities count
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].fillna("").apply(
            lambda x: len([i for i in str(x).split(",") if i.strip()])
        )
    else:
        df["amenities_count"] = 0

    return df


# ==============================
# RATING ADJUSTMENT
# ==============================

def rating_adjustment(rating):
    """
    Dollar adjustment vs market average:
      >= 4.0  -> positive bonus
      3–3.99  -> mild penalty
      < 3.0   -> larger penalty
    """
    if rating >= 4:
        return 5 + (rating - 4) * 10
    elif rating >= 3:
        return -(3.99 - rating) * 20
    else:
        return -20 - ((3 - rating) / 3) * 10


# ==============================
# FIND COMPARABLE LISTINGS
# ==============================

def find_comps(df, listing):

    # Strict match: room type + property type + beds + baths
    comps = df[
        (df["room_type"].astype(str).str.lower() == listing["room_type"].lower()) &
        (df["property_type"].astype(str).str.lower() == listing["property_type"].lower()) &
        (df["bedrooms"] == listing["bedrooms"]) &
        (df["bathrooms"] == listing["bathrooms"])
    ].copy()

    if len(comps) < 5:
        print("  Not enough strict matches — relaxing to bedrooms/bathrooms only…")
        comps = df[
            (df["bedrooms"] == listing["bedrooms"]) &
            (df["bathrooms"] == listing["bathrooms"])
        ].copy()

    if comps.empty:
        return None

    # Neighbourhood first
    if "neighbourhood" in comps.columns:
        nb = comps[
            comps["neighbourhood"].astype(str).str.lower() == listing["neighbourhood"].lower()
        ]
        if len(nb) >= 20:
            print(f"  Using {len(nb):,} neighbourhood comps.")
            nb = nb.copy()
            nb["distance_km"] = 0.0
            return nb

    # Expand by distance radius
    comps["distance_km"] = haversine_vectorized(
        listing["latitude"], listing["longitude"],
        comps["latitude"].values, comps["longitude"].values,
    )

    for radius in range(5, 35, 5):
        rc = comps[comps["distance_km"] <= radius]
        if len(rc) >= 20:
            print(f"  Using {len(rc):,} comps within {radius} km.")
            return rc

    # Last resort: all available comps
    if len(comps) >= 5:
        print(f"  Warning: only {len(comps)} comps found — using all available.")
        return comps

    return None


# ==============================
# FAIR PRICE CALCULATION
# ==============================

def compute_fair_price(comps, listing):
    comps = comps.copy()

    comps["w_distance"] = np.exp(-0.5 * comps["distance_km"].fillna(0))
    comps["w_reviews_raw"] = np.log1p(comps["number_of_reviews"])
    max_r = comps["w_reviews_raw"].max()
    comps["w_reviews"] = comps["w_reviews_raw"] / max_r if max_r > 0 else 1.0
    comps["weight"] = comps["w_distance"] * comps["w_reviews"]

    total_w = comps["weight"].sum()
    if total_w == 0:
        comps["weight"] = 1.0
        total_w = float(len(comps))

    fair_base = np.sum(comps["price"] * comps["weight"]) / total_w
    variance = np.sum(comps["weight"] * (comps["price"] - fair_base) ** 2) / total_w
    sigma = np.sqrt(variance)

    amenity_diff = listing["amenities_count"] - comps["amenities_count"].mean()
    amenity_adj = float(np.clip(amenity_diff * 2, -20, 20))
    rating_adj = rating_adjustment(listing["rating"])

    fair_final = fair_base + rating_adj + amenity_adj
    lower = max(0.0, fair_final - sigma)
    upper = fair_final + sigma

    return fair_final, lower, upper, fair_base, rating_adj, amenity_adj, sigma


# ==============================
# CLASSIFICATION
# ==============================

def classify(price, lower, upper):
    if price < lower:
        return "UNDERPRICED"
    elif price > upper:
        return "OVERPRICED"
    else:
        return "FAIRLY PRICED"


# ==============================
# INPUT HELPER
# ==============================

def prompt(label, cast=str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            return cast(raw)
        except ValueError:
            print(f"    Invalid input — expected {cast.__name__}. Try again.")


# ==============================
# MAIN
# ==============================

def run():
    print("\n╔══════════════════════════════════════╗")
    print("║  Airbnb Fair Market Value Analyzer   ║")
    print("╚══════════════════════════════════════╝\n")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "airbnb-listings.csv"
    city = prompt("City")

    print()
    try:
        df = load_city_data(csv_path, city)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print(f"  {len(df):,} listings loaded for '{city}'.\n")

    print("Enter Listing Details")
    print("─" * 42)

    listing = {
        "room_type":       prompt("Room Type       (e.g. Entire home/apt)"),
        "property_type":   prompt("Property Type   (e.g. Apartment)"),
        "bedrooms":        prompt("Bedrooms", float),
        "bathrooms":       prompt("Bathrooms", float),
        "latitude":        prompt("Latitude", float),
        "longitude":       prompt("Longitude", float),
        "neighbourhood":   prompt("Neighbourhood"),
        "rating":          prompt("Rating (1–5)", float),
        "amenities_count": prompt("Number of Amenities", int),
        "price":           prompt("Current Listing Price ($)", float),
    }

    if not (0 < listing["rating"] <= 5):
        print("  ERROR: Rating must be between 0 and 5.")
        sys.exit(1)

    print("\n  Finding comparable listings…")
    comps = find_comps(df, listing)

    if comps is None:
        print("\n  ERROR: Not enough comparable listings found (need at least 5).")
        sys.exit(1)

    fair, lower, upper, base, rating_adj, amenity_adj, sigma = compute_fair_price(comps, listing)
    classification = classify(listing["price"], lower, upper)
    pct = (listing["price"] - fair) / fair * 100

    # Result display
    print("\n╔══════════════════════════════════════╗")
    print("║              RESULT                  ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Comps used           : {len(comps):,}")
    print(f"  Base weighted price  : ${base:>9.2f}")
    print(f"  Rating adjustment    : ${rating_adj:>+9.2f}   (rating: {listing['rating']})")
    print(f"  Amenity adjustment   : ${amenity_adj:>+9.2f}   (amenities: {listing['amenities_count']})")
    print(f"  {'─' * 38}")
    print(f"  Fair Market Price    : ${fair:>9.2f}")
    print(f"  Fair Price Range     : ${lower:.2f}  –  ${upper:.2f}")
    print(f"  {'─' * 38}")
    print(f"  Your Listed Price    : ${listing['price']:>9.2f}")
    print(f"  Difference           : {pct:>+.2f}%")

    icon = {"UNDERPRICED": "🔵", "OVERPRICED": "🔴", "FAIRLY PRICED": "🟢"}.get(classification, "")
    print(f"\n  {icon} Classification: {classification}\n")


if __name__ == "__main__":
    run()
