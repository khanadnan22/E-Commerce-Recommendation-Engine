# 🛒 E-Commerce Recommendation Engine

A Data Science project that builds a product recommendation system using **real e-commerce data from Kaggle**, featuring **4 recommendation algorithms**, a **model evaluation framework**, and **user behavior analytics**.

## ✨ Features

### Recommendation Algorithms
- **Content-Based Filtering** — TF-IDF on product titles & categories + cosine similarity
- **User-User Collaborative Filtering** — Finds similar users and recommends what they liked
- **Item-Item Collaborative Filtering** — Recommends products similar to what the user already bought
- **Hybrid Engine** — Blends collaborative + content-based scores with a tunable α weight
- **Popularity-Based** — Cold-start fallback using most-purchased products

### Model Evaluation
- **Precision@K, Recall@K, NDCG@K** — Standard ranking metrics
- **Hit Rate** — Percentage of users with at least one relevant recommendation
- **Catalog Coverage** — Percentage of products that get recommended
- **Algorithm Comparison** — Side-by-side radar charts and bar charts

### User Analytics
- **K-Means Clustering** — Segments users into behavioral groups (Power Users, Active Shoppers, etc.)
- **RFM-Inspired Features** — Frequency, rating behavior, category diversity
- **Interactive Scatter Plots** — Explore user clusters and feature distributions

### Dashboard
- **Streamlit Web UI** with premium glassmorphism styling
- **Plotly Interactive Charts** for evaluation metrics and analytics
- **8 pages** — Dataset Overview, 5 recommendation strategies, Model Evaluation, User Analytics

## 📊 Dataset

This project uses the **Amazon Electronics Ratings** dataset from Kaggle:
- 🔗 https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews
- ~7.8M real ratings across 4.2M users and 476K products
- Automatically filtered and sampled for fast experimentation

Also supports the **Online Retail Dataset**:
- 🔗 https://www.kaggle.com/datasets/carrie1/ecommerce-data
- ~541K real transactions from a UK-based online retailer

## 🚀 Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle API credentials (one-time)
#    → Go to https://www.kaggle.com/settings
#    → Click "Create New Token" under API section
#    → Place the downloaded kaggle.json in ~/.kaggle/

# 4. Launch web dashboard (dataset downloads automatically on first run)
streamlit run app.py
```

> **Note:** The dataset (~300MB) is automatically downloaded from Kaggle on the first run
> using `kagglehub`. You need a free [Kaggle account](https://www.kaggle.com/account/login)
> and an API token to enable auto-download.

## 📁 Project Structure

```
DS/
├── app.py                   # Streamlit web dashboard (8 pages)
├── recommendation_engine.py # Core ML engine (4 algorithms + evaluation)
├── data_loader.py           # Data loading, preprocessing & auto-download
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── data/                    # Auto-generated on first run
    ├── ratings_Electronics.csv  # Raw dataset (auto-downloaded from Kaggle)
    ├── products_clean.csv       # Preprocessed product catalog
    └── interactions_clean.csv   # Preprocessed user-product interactions
```

## 🧠 Algorithms

| Strategy | Method | Use Case |
|----------|--------|----------|
| Content-Based | TF-IDF + Cosine Similarity | "Products similar to X" |
| User-User CF | User Cosine Similarity on Ratings | "Recommended for user Y" |
| Item-Item CF | Item Cosine Similarity on Ratings | "Users who bought X also bought Y" |
| Hybrid | Weighted Blend (α × CF + (1-α) × Content) | Best of both worlds |
| Popularity | Purchase Frequency Ranking | Cold-start / new users |

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of recommended items that are relevant |
| Recall@K | Fraction of relevant items that are recommended |
| NDCG@K | Normalized Discounted Cumulative Gain — ranking quality |
| Hit Rate | % of users with at least one correct recommendation |
| Coverage | % of catalog that appears in recommendations |

## 🛠 Tech Stack

- **Python 3.x** — Core language
- **pandas / NumPy** — Data manipulation
- **scikit-learn** — TF-IDF, cosine similarity, K-Means clustering
- **Streamlit** — Interactive web dashboard
- **Plotly** — Rich interactive visualizations
