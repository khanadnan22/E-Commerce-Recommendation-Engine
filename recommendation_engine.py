"""
E-Commerce Recommendation Engine — powered by real Kaggle data.

Implements four recommendation strategies:
  1. Content-Based Filtering  — TF-IDF on product titles + categories
  2. User-User Collaborative  — Cosine similarity between users
  3. Item-Item Collaborative  — Cosine similarity between items
  4. Hybrid Filtering         — Weighted blend of collaborative + content-based

Includes evaluation metrics:
  - Precision@K, Recall@K, NDCG@K, Hit Rate, Catalog Coverage
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_and_clean


class RecommendationEngine:
    """Core recommendation engine with multiple filtering strategies."""

    def __init__(self, sample_frac=None, force_reload=False):
        """
        Initialize the engine by loading data and building similarity matrices.

        Parameters
        ----------
        sample_frac : float or None
            Fraction of users to sample (0.0–1.0). Use for faster loading.
            Set to None to use the full dataset.
        force_reload : bool
            If True, force re-processing of raw CSV data.
        """
        print("[Engine] Loading data...")
        self.products, self.interactions = load_and_clean(
            force_reload=force_reload,
            sample_frac=sample_frac
        )

        print("[Engine] Building interaction matrix...")
        self.interaction_matrix = self.interactions.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            aggfunc='mean'
        ).fillna(0)

        print("[Engine] Computing user similarity...")
        user_sim = cosine_similarity(self.interaction_matrix)
        self.user_similarity_df = pd.DataFrame(
            user_sim,
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )

        print("[Engine] Computing item similarity...")
        self._build_item_similarity()

        print("[Engine] Building content similarity...")
        self.products['text_features'] = (
            self.products['category'].fillna('') + ' ' +
            self.products['title'].fillna('')
        )
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.products['text_features'])
        self.product_similarity = cosine_similarity(self.tfidf_matrix)

        self._pid_to_idx = {
            pid: idx for idx, pid in enumerate(self.products['product_id'])
        }

        print(f"[Engine] Ready — {len(self.products)} products, "
              f"{len(self.interaction_matrix)} users")

    def _build_item_similarity(self):
        """Build item-item cosine similarity from the interaction matrix."""
        item_matrix = self.interaction_matrix.T
        item_sim = cosine_similarity(item_matrix)
        self.item_similarity_df = pd.DataFrame(
            item_sim,
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns
        )

    def get_similar_products(self, product_id, top_n=5):
        """
        Find products most similar to the given product using
        TF-IDF + cosine similarity on title and category text.

        Returns a DataFrame of recommended products.
        """
        if product_id not in self._pid_to_idx:
            return pd.DataFrame()

        prod_idx = self._pid_to_idx[product_id]
        sim_scores = list(enumerate(self.product_similarity[prod_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top_indices = [i for i, _ in sim_scores[1:top_n + 1]]
        top_scores = [s for _, s in sim_scores[1:top_n + 1]]

        result = self.products.iloc[top_indices].copy()
        result['similarity_score'] = top_scores
        return result[['product_id', 'title', 'category', 'avg_price', 'similarity_score']]

    def get_personalized_recommendations(self, user_id, top_n=5, n_neighbors=10):
        """
        Recommend products using user-user collaborative filtering.

        Returns a DataFrame of recommended products.
        """
        if user_id not in self.interaction_matrix.index:
            return pd.DataFrame()

        sim_users = (
            self.user_similarity_df[user_id]
            .sort_values(ascending=False)
            .index[1:n_neighbors + 1]
        )

        neighbor_ratings = self.interaction_matrix.loc[sim_users]
        weighted_scores = neighbor_ratings.mean(axis=0)

        user_rated = self.interaction_matrix.loc[user_id]
        unseen = weighted_scores[user_rated == 0]

        top_product_ids = unseen.sort_values(ascending=False).head(top_n).index
        result = self.products[self.products['product_id'].isin(top_product_ids)].copy()

        result['predicted_score'] = result['product_id'].map(
            unseen[top_product_ids].to_dict()
        ).round(3)
        result = result.sort_values('predicted_score', ascending=False)

        return result[['product_id', 'title', 'category', 'avg_price', 'predicted_score']]

    def get_item_based_recommendations(self, user_id, top_n=5):
        """
        Recommend products using item-item collaborative filtering.

        For each unrated item, compute a weighted average of its similarity
        to items the user has already rated.

        Returns a DataFrame of recommended products.
        """
        if user_id not in self.interaction_matrix.index:
            return pd.DataFrame()

        user_ratings = self.interaction_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        unrated_items = user_ratings[user_ratings == 0].index

        if rated_items.empty or len(unrated_items) == 0:
            return pd.DataFrame()

        scores = {}
        rated_idx = rated_items.index
        for item in unrated_items:
            if item not in self.item_similarity_df.columns:
                continue
            sim_scores = self.item_similarity_df.loc[item, rated_idx]
            weighted_sum = (sim_scores * rated_items).sum()
            sim_sum = sim_scores.abs().sum()
            if sim_sum > 0:
                scores[item] = weighted_sum / sim_sum

        if not scores:
            return pd.DataFrame()

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_product_ids = [pid for pid, _ in top_items]
        top_scores_dict = {pid: round(score, 3) for pid, score in top_items}

        result = self.products[self.products['product_id'].isin(top_product_ids)].copy()
        result['predicted_score'] = result['product_id'].map(top_scores_dict)
        result = result.sort_values('predicted_score', ascending=False)

        return result[['product_id', 'title', 'category', 'avg_price', 'predicted_score']]

    def get_hybrid_recommendations(self, user_id, top_n=5, alpha=0.5, n_neighbors=10):
        """
        Hybrid recommendations combining user-user collaborative filtering
        with content-based similarity.

        Parameters
        ----------
        user_id : str
            Target user.
        top_n : int
            Number of recommendations to return.
        alpha : float
            Weight for collaborative score (0–1). Content weight = 1 - alpha.
        n_neighbors : int
            Number of neighbor users for collaborative filtering.

        Returns a DataFrame with hybrid scores and component scores.
        """
        if user_id not in self.interaction_matrix.index:
            return pd.DataFrame()

        pool_size = min(top_n * 10, 100)
        collab_recs = self.get_personalized_recommendations(
            user_id, top_n=pool_size, n_neighbors=n_neighbors
        )

        if collab_recs.empty:
            return pd.DataFrame()

        user_ratings = self.interaction_matrix.loc[user_id]
        rated_pids = user_ratings[user_ratings > 0].index.tolist()

        content_scores = {}
        for _, row in collab_recs.iterrows():
            pid = row['product_id']
            if pid in self._pid_to_idx:
                pid_idx = self._pid_to_idx[pid]
                sims = []
                for rated_pid in rated_pids:
                    if rated_pid in self._pid_to_idx:
                        rated_idx = self._pid_to_idx[rated_pid]
                        sims.append(self.product_similarity[pid_idx, rated_idx])
                content_scores[pid] = np.mean(sims) if sims else 0.0

        result = collab_recs.copy()
        result['content_score'] = result['product_id'].map(content_scores).fillna(0)

        collab_min = result['predicted_score'].min()
        collab_max = result['predicted_score'].max()
        content_min = result['content_score'].min()
        content_max = result['content_score'].max()

        if collab_max > collab_min:
            result['collab_norm'] = (
                (result['predicted_score'] - collab_min) / (collab_max - collab_min)
            )
        else:
            result['collab_norm'] = 0.5

        if content_max > content_min:
            result['content_norm'] = (
                (result['content_score'] - content_min) / (content_max - content_min)
            )
        else:
            result['content_norm'] = 0.5

        result['hybrid_score'] = (
            alpha * result['collab_norm'] + (1 - alpha) * result['content_norm']
        ).round(3)

        result = result.sort_values('hybrid_score', ascending=False).head(top_n)

        return result[['product_id', 'title', 'category', 'avg_price',
                        'hybrid_score', 'predicted_score', 'content_score']]

    def get_popular_products(self, top_n=10, category=None):
        """
        Return the most popular products based on number of unique buyers.
        Optionally filter by category. Used as a cold-start fallback.
        """
        popularity = (
            self.interactions.groupby('product_id')
            .agg(
                n_buyers=('user_id', 'nunique'),
                avg_rating=('rating', 'mean')
            )
            .reset_index()
        )

        result = self.products.merge(popularity, on='product_id', how='inner')

        if category:
            result = result[result['category'] == category]

        result = result.sort_values(
            ['n_buyers', 'avg_rating'], ascending=[False, False]
        ).head(top_n)

        return result[['product_id', 'title', 'category', 'avg_price',
                        'n_buyers', 'avg_rating']]

    def evaluate_models(self, test_size=0.2, k_values=None, max_eval_users=200):
        """
        Evaluate recommendation algorithms using a train/test split.

        Parameters
        ----------
        test_size : float
            Fraction of each user's interactions to hold out for testing.
        k_values : list of int
            Values of K for Precision@K, Recall@K, NDCG@K.
        max_eval_users : int
            Maximum number of users to evaluate (for speed).

        Returns
        -------
        tuple : (results_dict, meta_dict)
        """
        if k_values is None:
            k_values = [5, 10, 20]

        rng = np.random.RandomState(42)

        test_items_map = {}

        for user_id in self.interaction_matrix.index:
            user_items = self.interactions[
                self.interactions['user_id'] == user_id
            ]['product_id'].unique()

            if len(user_items) < 3:
                continue

            n_test = max(1, int(len(user_items) * test_size))
            test_pids = set(rng.choice(user_items, n_test, replace=False))
            test_items_map[user_id] = test_pids

        eval_users = list(test_items_map.keys())
        if len(eval_users) > max_eval_users:
            eval_users = list(rng.choice(eval_users, max_eval_users, replace=False))

        algorithms = {
            'User-User CF': 'user_cf',
            'Item-Item CF': 'item_cf',
            'Content-Based': 'content',
            'Popularity': 'popularity',
        }

        results = {}

        for algo_name, algo_key in algorithms.items():
            algo_results = {}

            for k in k_values:
                precisions, recalls, ndcgs, hits = [], [], [], []
                all_recommended = set()

                for user_id in eval_users:
                    true_items = test_items_map[user_id]

                    try:
                        if algo_key == 'user_cf':
                            recs = self.get_personalized_recommendations(
                                user_id, top_n=k
                            )
                        elif algo_key == 'item_cf':
                            recs = self.get_item_based_recommendations(
                                user_id, top_n=k
                            )
                        elif algo_key == 'content':
                            user_pids = self.interactions[
                                self.interactions['user_id'] == user_id
                            ]['product_id'].tolist()
                            if user_pids:
                                recs = self.get_similar_products(
                                    user_pids[0], top_n=k
                                )
                            else:
                                continue
                        elif algo_key == 'popularity':
                            recs = self.get_popular_products(top_n=k)
                        else:
                            continue
                    except Exception:
                        continue

                    if recs.empty:
                        continue

                    rec_pids = set(recs['product_id'].tolist())
                    all_recommended.update(rec_pids)

                    relevant = rec_pids & true_items

                    precisions.append(len(relevant) / k)

                    recalls.append(
                        len(relevant) / len(true_items) if true_items else 0
                    )

                    hits.append(1 if relevant else 0)

                    rec_list = recs['product_id'].tolist()
                    dcg = sum(
                        1.0 / np.log2(i + 2)
                        for i, pid in enumerate(rec_list)
                        if pid in true_items
                    )
                    ideal_hits = min(len(true_items), k)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
                    ndcgs.append(dcg / idcg if idcg > 0 else 0)

                coverage = (
                    len(all_recommended) / len(self.products)
                    if len(self.products) > 0 else 0
                )

                algo_results[k] = {
                    'Precision@K': round(np.mean(precisions), 4) if precisions else 0,
                    'Recall@K': round(np.mean(recalls), 4) if recalls else 0,
                    'NDCG@K': round(np.mean(ndcgs), 4) if ndcgs else 0,
                    'Hit Rate': round(np.mean(hits), 4) if hits else 0,
                    'Coverage': round(coverage, 4),
                    'Users Evaluated': len(precisions),
                }

            results[algo_name] = algo_results

        return results, {
            'total_test_users': len(eval_users),
            'test_size': test_size,
        }

    def get_user_segments(self):
        """
        Segment users into clusters based on behavioral features using K-Means.

        Features: frequency, total_interactions, avg_rating, rating_std, n_categories

        Returns a DataFrame with user features and cluster assignments.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        interactions_cat = self.interactions.merge(
            self.products[['product_id', 'category']],
            on='product_id', how='left'
        )

        user_features = (
            interactions_cat.groupby('user_id')
            .agg(
                frequency=('product_id', 'nunique'),
                total_interactions=('product_id', 'count'),
                avg_rating=('rating', 'mean'),
                rating_std=('rating', 'std'),
                n_categories=('category', 'nunique'),
            )
            .reset_index()
        )
        user_features['rating_std'] = user_features['rating_std'].fillna(0)

        feature_cols = [
            'frequency', 'total_interactions', 'avg_rating',
            'rating_std', 'n_categories'
        ]
        X = user_features[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        user_features['cluster'] = kmeans.fit_predict(X_scaled)

        cluster_names = {}
        freq_med = user_features['frequency'].median()
        rating_med = user_features['avg_rating'].median()

        for c in range(4):
            cdata = user_features[user_features['cluster'] == c]
            high_freq = cdata['frequency'].mean() > freq_med
            high_rating = cdata['avg_rating'].mean() > rating_med

            if high_freq and high_rating:
                cluster_names[c] = '⭐ Power Users'
            elif high_freq:
                cluster_names[c] = '🛒 Active Shoppers'
            elif high_rating:
                cluster_names[c] = '😊 Satisfied Casual'
            else:
                cluster_names[c] = '💤 Low Engagement'

        user_features['segment'] = user_features['cluster'].map(cluster_names)

        return user_features

    def get_stats(self):
        """Return a dictionary of dataset statistics for the dashboard."""
        return {
            'total_products': len(self.products),
            'total_users': self.interaction_matrix.shape[0],
            'total_interactions': len(self.interactions),
            'categories': self.products['category'].nunique(),
            'avg_rating': round(self.interactions['rating'].mean(), 2),
            'sparsity': round(
                1 - (self.interactions.shape[0] /
                     (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1])),
                4
            ) * 100,
            'category_distribution': self.products['category'].value_counts().to_dict(),
            'rating_distribution': self.interactions['rating'].value_counts().sort_index().to_dict(),
        }
