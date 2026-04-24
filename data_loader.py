"""
Data Loader & Preprocessor for E-Commerce Recommendation Engine.

Supports two dataset formats:

  1. Amazon Electronics Ratings (Kaggle):
     https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews
     Columns (no header): userId, productId, rating, timestamp

  2. Online Retail (Kaggle):
     https://www.kaggle.com/datasets/carrie1/ecommerce-data
     Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
              UnitPrice, CustomerID, Country

Dataset is auto-downloaded from Kaggle on first run using kagglehub.
Requires a free Kaggle account — see https://www.kaggle.com/docs/api
"""

import os
import shutil
import glob
import pandas as pd
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PRODUCTS_CACHE = os.path.join(DATA_DIR, 'products_clean.csv')
INTERACTIONS_CACHE = os.path.join(DATA_DIR, 'interactions_clean.csv')

KAGGLE_DATASET = "saurav9786/amazon-product-reviews"


def _download_dataset():
    """
    Auto-download the Amazon Electronics Ratings dataset from Kaggle
    using kagglehub. Requires a free Kaggle account.

    On first run, kagglehub will prompt for Kaggle credentials
    (username + API key from https://www.kaggle.com/settings).
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required to auto-download the dataset.\n"
            "Install it with: pip install kagglehub\n"
            "Or manually download from:\n"
            "  https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews\n"
            "and place the CSV in the 'data/' folder."
        )

    print(f"[DataLoader] Downloading dataset from Kaggle: {KAGGLE_DATASET}")
    print("[DataLoader] If prompted, enter your Kaggle credentials.")
    print("[DataLoader] Get your API key from: https://www.kaggle.com/settings\n")

    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[DataLoader] Downloaded to: {download_path}")

    os.makedirs(DATA_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in downloaded dataset at {download_path}"
        )

    for csv_file in csv_files:
        dest = os.path.join(DATA_DIR, os.path.basename(csv_file))
        if not os.path.exists(dest):
            shutil.copy2(csv_file, dest)
            print(f"[DataLoader] Copied: {os.path.basename(csv_file)} → data/")

    return True


def _detect_raw_file():
    """Auto-detect the raw CSV file inside the data/ directory.
    If no raw file is found, attempts to download from Kaggle."""
    os.makedirs(DATA_DIR, exist_ok=True)

    csvs = [f for f in os.listdir(DATA_DIR)
            if f.endswith('.csv') and not f.endswith('_clean.csv')]
    if csvs:
        return os.path.join(DATA_DIR, csvs[0])

    print("[DataLoader] No raw dataset found in data/ folder.")
    print("[DataLoader] Attempting auto-download from Kaggle...\n")
    _download_dataset()

    csvs = [f for f in os.listdir(DATA_DIR)
            if f.endswith('.csv') and not f.endswith('_clean.csv')]
    if csvs:
        return os.path.join(DATA_DIR, csvs[0])

    return None


def _detect_format(path):
    """
    Detect whether the CSV is Amazon Ratings or Online Retail format.

    Amazon Ratings: no header, 4 columns (userId, productId, rating, timestamp)
    Online Retail:  has header with 'InvoiceNo', 'StockCode', etc.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline().strip()

    if 'InvoiceNo' in first_line or 'StockCode' in first_line:
        return 'online_retail'

    parts = first_line.split(',')
    if len(parts) == 4:
        try:
            float(parts[2])
            int(parts[3])
            return 'amazon_ratings'
        except ValueError:
            pass

    return 'unknown'


def load_and_clean(force_reload=False, sample_frac=None):
    """
    Load the raw dataset, clean it, and return (products_df, interactions_df).

    Automatically detects the dataset format and processes accordingly.

    Parameters
    ----------
    force_reload : bool
        If True, re-process from the raw CSV even if cached files exist.
    sample_frac : float or None
        If set (0.0–1.0), sample this fraction of users for faster experimentation.

    Returns
    -------
    products_df : pd.DataFrame
        Columns: product_id, title, category
    interactions_df : pd.DataFrame
        Columns: user_id, product_id, rating, interaction_type
    """
    if not force_reload and os.path.exists(PRODUCTS_CACHE) and os.path.exists(INTERACTIONS_CACHE):
        products_df = pd.read_csv(PRODUCTS_CACHE)
        interactions_df = pd.read_csv(INTERACTIONS_CACHE)
        print(f"[DataLoader] Loaded cached data: {len(products_df)} products, "
              f"{len(interactions_df)} interactions")
        return products_df, interactions_df

    raw_path = _detect_raw_file()
    if raw_path is None:
        raise FileNotFoundError(
            f"No CSV file found in '{DATA_DIR}/'.\n"
            "Please download one of these datasets from Kaggle:\n"
            "  1. Amazon Electronics Ratings:\n"
            "     https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews\n"
            "  2. Online Retail:\n"
            "     https://www.kaggle.com/datasets/carrie1/ecommerce-data\n"
            "Extract the CSV and place it in the 'data/' folder."
        )

    fmt = _detect_format(raw_path)
    print(f"[DataLoader] Detected format: {fmt}")
    print(f"[DataLoader] Loading raw data from: {raw_path}")

    if fmt == 'amazon_ratings':
        products_df, interactions_df = _load_amazon_ratings(raw_path)
    elif fmt == 'online_retail':
        products_df, interactions_df = _load_online_retail(raw_path)
    else:
        raise ValueError(
            f"Unrecognized CSV format in '{raw_path}'.\n"
            "Expected either Amazon Ratings or Online Retail format."
        )

    if sample_frac and 0 < sample_frac < 1:
        print(f"[DataLoader] Sampling {sample_frac*100:.0f}% of users...")
        unique_users = interactions_df['user_id'].unique()
        n_sample = max(1, int(len(unique_users) * sample_frac))
        rng = np.random.RandomState(42)
        sampled_users = rng.choice(unique_users, n_sample, replace=False)
        interactions_df = interactions_df[interactions_df['user_id'].isin(sampled_users)]
        products_df = products_df[products_df['product_id'].isin(
            interactions_df['product_id'].unique()
        )]

    products_df.to_csv(PRODUCTS_CACHE, index=False)
    interactions_df.to_csv(INTERACTIONS_CACHE, index=False)

    print(f"[DataLoader] Final dataset: {len(products_df)} products, "
          f"{len(interactions_df)} interactions, "
          f"{interactions_df['user_id'].nunique()} unique users")

    return products_df, interactions_df


def _load_amazon_ratings(path):
    """
    Load the Amazon Electronics Ratings CSV.
    Format (no header): userId, productId, rating, timestamp
    """
    print("[DataLoader] Reading Amazon ratings...")
    df = pd.read_csv(
        path,
        header=None,
        names=['user_id', 'product_id', 'rating', 'timestamp'],
        dtype={'user_id': str, 'product_id': str, 'rating': float, 'timestamp': int}
    )
    print(f"[DataLoader] Raw: {len(df):,} ratings")

    min_ratings = 5
    print(f"[DataLoader] Filtering for min {min_ratings} ratings per user/product...")
    for _ in range(3):
        product_counts = df['product_id'].value_counts()
        df = df[df['product_id'].isin(product_counts[product_counts >= min_ratings].index)]

        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_ratings].index)]

    print(f"[DataLoader] After 5-core filtering: {len(df):,} ratings, "
          f"{df['user_id'].nunique():,} users, {df['product_id'].nunique():,} products")

    MAX_USERS = 5000
    MAX_PRODUCTS = 3000

    if df['user_id'].nunique() > MAX_USERS:
        top_users = df['user_id'].value_counts().head(MAX_USERS).index
        df = df[df['user_id'].isin(top_users)]

    if df['product_id'].nunique() > MAX_PRODUCTS:
        top_products = df['product_id'].value_counts().head(MAX_PRODUCTS).index
        df = df[df['product_id'].isin(top_products)]

    print(f"[DataLoader] After capping: {len(df):,} ratings, "
          f"{df['user_id'].nunique():,} users, {df['product_id'].nunique():,} products")

    product_stats = (
        df.groupby('product_id')
        .agg(
            avg_rating=('rating', 'mean'),
            n_ratings=('rating', 'count'),
        )
        .reset_index()
    )

    product_stats['title'] = product_stats['product_id'].apply(
        lambda x: f"Electronics #{x[-4:]}"
    )
    product_stats['category'] = product_stats.apply(_classify_electronics, axis=1)
    product_stats['avg_price'] = 0.0

    products_df = product_stats[['product_id', 'title', 'category', 'avg_price']]

    interactions_df = df[['user_id', 'product_id', 'rating']].copy()
    interactions_df['interaction_type'] = 'rating'
    interactions_df['rating'] = interactions_df['rating'].round().astype(int)

    return products_df, interactions_df


def _classify_electronics(row):
    """Classify electronics products into subcategories based on rating patterns."""
    avg = row['avg_rating']
    n = row['n_ratings']

    if n > 50 and avg > 4.0:
        return 'Premium Electronics'
    elif n > 50 and avg <= 3.0:
        return 'Budget Electronics'
    elif n > 20:
        return 'Popular Electronics'
    elif avg >= 4.5:
        return 'Highly Rated'
    elif avg >= 3.5:
        return 'Well Rated'
    elif avg >= 2.5:
        return 'Mixed Reviews'
    else:
        return 'Low Rated'


def _load_online_retail(path):
    """
    Load the Online Retail dataset.
    Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
             UnitPrice, CustomerID, Country
    """
    print("[DataLoader] Reading Online Retail data...")
    df = pd.read_csv(path, encoding='unicode_escape')

    df = df.dropna(subset=['CustomerID', 'Description'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['Description'] = df['Description'].str.strip().str.title()

    exclude_codes = ['POST', 'DOT', 'M', 'BANK CHARGES', 'PADS', 'CRUK',
                     'C2', 'D', 'S', 'AMAZONFEE', 'B']
    df = df[~df['StockCode'].astype(str).str.upper().isin(exclude_codes)]

    print(f"[DataLoader] After cleaning: {len(df):,} transaction rows")

    products_df = (
        df.groupby('StockCode')
        .agg(
            title=('Description', 'first'),
            total_qty=('Quantity', 'sum'),
            avg_price=('UnitPrice', 'mean'),
        )
        .reset_index()
        .rename(columns={'StockCode': 'product_id'})
    )
    products_df['category'] = products_df['title'].apply(_infer_retail_category)
    products_df = products_df[['product_id', 'title', 'category', 'avg_price']]

    interaction_agg = (
        df.groupby(['CustomerID', 'StockCode'])
        .agg(
            purchase_count=('InvoiceNo', 'nunique'),
            total_qty=('Quantity', 'sum'),
        )
        .reset_index()
        .rename(columns={'CustomerID': 'user_id', 'StockCode': 'product_id'})
    )

    interaction_agg['rating'] = _compute_ratings(
        interaction_agg['purchase_count'], interaction_agg['total_qty']
    )
    interaction_agg['interaction_type'] = 'purchase'

    interactions_df = interaction_agg[['user_id', 'product_id', 'rating', 'interaction_type']]
    interactions_df = interactions_df.copy()
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)
    interactions_df['product_id'] = interactions_df['product_id'].astype(str)

    return products_df, interactions_df


def _infer_retail_category(title):
    """Keyword-based category inference from product titles."""
    title_lower = str(title).lower()

    category_keywords = {
        'Home & Kitchen': ['candle', 'holder', 'jar', 'frame', 'clock', 'vase',
                           'cushion', 'drawer', 'cabinet', 'shelf', 'mirror',
                           'doormat', 'curtain', 'lamp', 'light', 'lantern'],
        'Party & Gifts': ['party', 'bunting', 'banner', 'birthday', 'christmas',
                          'xmas', 'gift', 'card', 'wrap', 'ribbon', 'garland'],
        'Kitchen & Dining': ['mug', 'cup', 'plate', 'bowl', 'spoon', 'fork',
                             'kitchen', 'cook', 'bake', 'tea', 'coffee',
                             'napkin', 'coaster', 'tray', 'bottle'],
        'Bags & Storage': ['bag', 'box', 'case', 'basket', 'tin', 'container',
                           'storage', 'pouch'],
        'Stationery': ['pen', 'pencil', 'notebook', 'diary', 'paper', 'stamp',
                       'sticker', 'letter', 'memo', 'pad'],
        'Toys & Games': ['toy', 'game', 'doll', 'puzzle', 'play', 'craft',
                         'paint', 'colour', 'color'],
        'Fashion & Accessories': ['jewel', 'necklace', 'bracelet', 'ring',
                                   'earring', 'scarf', 'hat', 'glove', 'purse',
                                   'charm', 'heart'],
        'Garden & Outdoor': ['garden', 'plant', 'flower', 'pot', 'seed',
                              'bird', 'outdoor', 'flag'],
        'Vintage & Retro': ['vintage', 'retro', 'antique', 'classic', 'shabby',
                             'chic', 'old', 'union jack'],
    }

    for category, keywords in category_keywords.items():
        if any(kw in title_lower for kw in keywords):
            return category

    return 'General'


def _compute_ratings(purchase_count, total_qty):
    """Convert purchase frequency and quantity into a 1–5 rating scale."""
    score = np.log1p(purchase_count) + np.log1p(total_qty)
    try:
        ratings = pd.qcut(score, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        ratings = ratings.astype(int)
    except ValueError:
        ratings = pd.cut(score, bins=5, labels=[1, 2, 3, 4, 5])
        ratings = ratings.astype(int)
    return ratings


if __name__ == '__main__':
    print("=" * 60)
    print("  E-Commerce Data Loader — Preprocessing Pipeline")
    print("=" * 60)
    products, interactions = load_and_clean(force_reload=True)
    print(f"\n--- Products Sample ---")
    print(products.head(10).to_string(index=False))
    print(f"\n--- Interactions Sample ---")
    print(interactions.head(10).to_string(index=False))
    print(f"\n--- Category Distribution ---")
    print(products['category'].value_counts().to_string())
