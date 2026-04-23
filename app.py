"""
E-Commerce Recommendation Engine — Streamlit Dashboard.

Interactive web UI for product recommendations, model evaluation,
and user behavior analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommendation_engine import RecommendationEngine

st.set_page_config(
    page_title="E-Commerce Recommendation Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #c8c8e0 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #fff !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    opacity: 0.7;
}

.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.45) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(102, 126, 234, 0.15);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}

hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, #667eea40, transparent) !important;
    margin: 1.5rem 0 !important;
}

.section-header {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border-left: 4px solid #667eea;
    padding: 12px 20px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1.2rem;
}
.section-header h3 {
    margin: 0;
    color: #667eea;
}

.algo-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.3s;
}
.algo-card:hover {
    border-color: #667eea;
}
.algo-card h4 { margin: 0 0 0.4rem 0; color: #667eea; }
.algo-card p { margin: 0; font-size: 0.9rem; opacity: 0.8; }

[data-testid="stExpander"] {
    border: 1px solid rgba(102,126,234,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load the recommendation engine (cached across reruns)."""
    return RecommendationEngine(sample_frac=0.3)


@st.cache_data(show_spinner=False)
def run_evaluation(_engine, test_size, k_values_tuple):
    """Run model evaluation (cached)."""
    return _engine.evaluate_models(
        test_size=test_size,
        k_values=list(k_values_tuple)
    )


@st.cache_data(show_spinner=False)
def get_user_segments(_engine):
    """Get user segments (cached)."""
    return _engine.get_user_segments()


def page_dataset_overview(engine):
    st.header("📊 Dataset Overview")
    stats = engine.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", f"{stats['total_products']:,}")
    col2.metric("Users", f"{stats['total_users']:,}")
    col3.metric("Interactions", f"{stats['total_interactions']:,}")
    col4.metric("Categories", stats['categories'])

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Rating", stats['avg_rating'])
    col6.metric("Sparsity", f"{stats['sparsity']:.1f}%")
    col7.metric("Avg Items/User",
                f"{stats['total_interactions'] / max(stats['total_users'], 1):.1f}")
    col8.metric("Avg Users/Item",
                f"{stats['total_interactions'] / max(stats['total_products'], 1):.1f}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Category Distribution")
        cat_df = pd.DataFrame(
            list(stats['category_distribution'].items()),
            columns=['Category', 'Count']
        ).sort_values('Count', ascending=True)

        fig = px.bar(
            cat_df, x='Count', y='Category', orientation='h',
            color='Count',
            color_continuous_scale=['#667eea', '#764ba2'],
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=0, r=20, t=10, b=0),
            yaxis=dict(title=''),
            xaxis=dict(title='Number of Products', gridcolor='rgba(102,126,234,0.1)'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Rating Distribution")
        rat_df = pd.DataFrame(
            list(stats['rating_distribution'].items()),
            columns=['Rating', 'Count']
        ).sort_values('Rating')
        rat_df['Rating'] = rat_df['Rating'].astype(str) + ' ⭐'

        fig = px.bar(
            rat_df, x='Rating', y='Count',
            color='Count',
            color_continuous_scale=['#667eea', '#764ba2'],
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis=dict(title=''),
            yaxis=dict(title='Count', gridcolor='rgba(102,126,234,0.1)'),
        )
        st.plotly_chart(fig, use_container_width=True)


def page_similar_products(engine):
    st.header("🔍 Find Similar Products")
    st.markdown(
        '<div class="algo-card">'
        '<h4>Content-Based Filtering</h4>'
        '<p>Uses TF-IDF vectorization on product titles and categories, '
        'then computes cosine similarity to find the most similar items.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    categories = ['All'] + sorted(engine.products['category'].unique().tolist())
    selected_cat = st.selectbox("Filter by category:", categories)

    if selected_cat == 'All':
        filtered = engine.products
    else:
        filtered = engine.products[engine.products['category'] == selected_cat]

    product_options = filtered['title'].tolist()
    if not product_options:
        st.warning("No products in this category.")
        return

    selected_product = st.selectbox(
        "Select a product:",
        product_options,
        help="Choose a product to find similar items"
    )

    top_n = st.slider("Number of recommendations:", 3, 20, 5)

    if st.button("🔎 Get Similar Products", type="primary"):
        prod_id = filtered[
            filtered['title'] == selected_product
        ]['product_id'].values[0]

        with st.spinner("Computing similarities..."):
            recommendations = engine.get_similar_products(prod_id, top_n=top_n)

        if not recommendations.empty:
            st.success(
                f"Top {len(recommendations)} products similar to "
                f"**{selected_product}**:"
            )
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'product_id': 'Product ID',
                    'title': 'Product Name',
                    'category': 'Category',
                    'avg_price': st.column_config.NumberColumn(
                        'Avg Price (£)', format='£%.2f'
                    ),
                    'similarity_score': st.column_config.ProgressColumn(
                        'Similarity', min_value=0, max_value=1, format='%.3f'
                    ),
                }
            )
        else:
            st.info("No similar products found for this item.")


def page_user_user_recs(engine):
    st.header("👤 User-User Recommendations")
    st.markdown(
        '<div class="algo-card">'
        '<h4>User-User Collaborative Filtering</h4>'
        '<p>Finds users with similar purchase patterns and recommends '
        'products that similar users liked but the target user hasn\'t seen yet.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    user_ids = sorted(engine.interaction_matrix.index.tolist())
    selected_user = st.selectbox(
        "Select User ID:",
        user_ids,
        help="Choose a customer to see their personalized picks"
    )

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of recommendations:", 3, 20, 5, key='uu_n')
    with col2:
        n_neighbors = st.slider("Neighbor users to consider:", 3, 30, 10)

    if st.button("🎯 Get Recommendations", type="primary"):
        user_products = engine.interactions[
            engine.interactions['user_id'] == selected_user
        ]['product_id'].unique()
        user_history = engine.products[
            engine.products['product_id'].isin(user_products)
        ]

        with st.expander(
            f"📋 User {selected_user}'s Purchase History "
            f"({len(user_history)} products)", expanded=False
        ):
            st.dataframe(
                user_history[['product_id', 'title', 'category', 'avg_price']],
                use_container_width=True,
                hide_index=True
            )

        with st.spinner("Computing personalized recommendations..."):
            recommendations = engine.get_personalized_recommendations(
                selected_user, top_n=top_n, n_neighbors=n_neighbors
            )

        if not recommendations.empty:
            st.success(
                f"Top {len(recommendations)} recommendations for "
                f"User {selected_user}:"
            )
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'product_id': 'Product ID',
                    'title': 'Product Name',
                    'category': 'Category',
                    'avg_price': st.column_config.NumberColumn(
                        'Avg Price (£)', format='£%.2f'
                    ),
                    'predicted_score': st.column_config.ProgressColumn(
                        'Predicted Score', min_value=0, max_value=5,
                        format='%.3f'
                    ),
                }
            )
        else:
            st.info("No personalized recommendations available.")
            popular = engine.get_popular_products(top_n=top_n)
            st.write("Showing popular items as fallback:")
            st.dataframe(popular, use_container_width=True, hide_index=True)


def page_item_item_recs(engine):
    st.header("🔗 Item-Item Recommendations")
    st.markdown(
        '<div class="algo-card">'
        '<h4>Item-Item Collaborative Filtering</h4>'
        '<p>Computes similarity between items based on how users co-rated them. '
        'For each unrated product, predicts a score using a weighted average of '
        'similarities to products the user has already interacted with.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    user_ids = sorted(engine.interaction_matrix.index.tolist())
    selected_user = st.selectbox(
        "Select User ID:",
        user_ids,
        key='ii_user',
        help="Choose a customer to see item-based recommendations"
    )

    top_n = st.slider("Number of recommendations:", 3, 20, 5, key='ii_n')

    if st.button("🔗 Get Item-Based Recommendations", type="primary"):
        user_products = engine.interactions[
            engine.interactions['user_id'] == selected_user
        ]['product_id'].unique()
        user_history = engine.products[
            engine.products['product_id'].isin(user_products)
        ]

        with st.expander(
            f"📋 User {selected_user}'s Purchase History "
            f"({len(user_history)} products)", expanded=False
        ):
            st.dataframe(
                user_history[['product_id', 'title', 'category', 'avg_price']],
                use_container_width=True,
                hide_index=True
            )

        with st.spinner("Computing item-item recommendations..."):
            recommendations = engine.get_item_based_recommendations(
                selected_user, top_n=top_n
            )

        if not recommendations.empty:
            st.success(
                f"Top {len(recommendations)} item-based recommendations for "
                f"User {selected_user}:"
            )
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'product_id': 'Product ID',
                    'title': 'Product Name',
                    'category': 'Category',
                    'avg_price': st.column_config.NumberColumn(
                        'Avg Price (£)', format='£%.2f'
                    ),
                    'predicted_score': st.column_config.ProgressColumn(
                        'Predicted Score', min_value=0, max_value=5,
                        format='%.3f'
                    ),
                }
            )
        else:
            st.info("No item-based recommendations available for this user.")


def page_hybrid_recs(engine):
    st.header("🔀 Hybrid Recommendations")
    st.markdown(
        '<div class="algo-card">'
        '<h4>Hybrid Recommendation Engine</h4>'
        '<p>Blends collaborative filtering (what similar users liked) with '
        'content-based similarity (product feature matching). Adjust the '
        '<strong>alpha slider</strong> to control the balance — higher alpha '
        'favors collaborative signals, lower alpha favors content similarity.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    user_ids = sorted(engine.interaction_matrix.index.tolist())
    selected_user = st.selectbox(
        "Select User ID:",
        user_ids,
        key='hyb_user',
        help="Choose a customer for hybrid recommendations"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Number of recommendations:", 3, 20, 5, key='hyb_n')
    with col2:
        alpha = st.slider(
            "Alpha (Collaborative ↔ Content):",
            0.0, 1.0, 0.5, 0.05,
            help="1.0 = 100% collaborative, 0.0 = 100% content-based"
        )
    with col3:
        n_neighbors = st.slider(
            "Neighbor users:", 3, 30, 10, key='hyb_neighbors'
        )

    col_l, col_r = st.columns(2)
    col_l.markdown(f"**Collaborative weight:** {alpha:.0%}")
    col_r.markdown(f"**Content-based weight:** {1 - alpha:.0%}")

    if st.button("🔀 Get Hybrid Recommendations", type="primary"):
        with st.spinner("Blending algorithms..."):
            recommendations = engine.get_hybrid_recommendations(
                selected_user, top_n=top_n, alpha=alpha,
                n_neighbors=n_neighbors
            )

        if not recommendations.empty:
            st.success(
                f"Top {len(recommendations)} hybrid recommendations for "
                f"User {selected_user} "
                f"(α={alpha:.2f}):"
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'product_id': 'Product ID',
                    'title': 'Product Name',
                    'category': 'Category',
                    'avg_price': st.column_config.NumberColumn(
                        'Avg Price (£)', format='£%.2f'
                    ),
                    'hybrid_score': st.column_config.ProgressColumn(
                        'Hybrid Score', min_value=0, max_value=1,
                        format='%.3f'
                    ),
                    'predicted_score': st.column_config.NumberColumn(
                        'Collab Score', format='%.3f'
                    ),
                    'content_score': st.column_config.NumberColumn(
                        'Content Score', format='%.4f'
                    ),
                }
            )

            st.subheader("Score Breakdown")
            chart_df = recommendations[['title', 'predicted_score', 'content_score']].copy()
            chart_df = chart_df.rename(columns={
                'predicted_score': 'Collaborative',
                'content_score': 'Content-Based'
            })
            chart_melted = chart_df.melt(
                id_vars='title',
                var_name='Algorithm',
                value_name='Score'
            )

            fig = px.bar(
                chart_melted, x='title', y='Score', color='Algorithm',
                barmode='group',
                color_discrete_map={
                    'Collaborative': '#667eea',
                    'Content-Based': '#764ba2'
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='', tickangle=45),
                yaxis=dict(
                    title='Score',
                    gridcolor='rgba(102,126,234,0.1)'
                ),
                legend=dict(orientation='h', y=1.1),
                height=400,
                margin=dict(l=0, r=0, t=30, b=100),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hybrid recommendations available for this user.")


def page_popular_products(engine):
    st.header("🔥 Popular / Trending Products")
    st.write("Most purchased products across all customers.")

    categories = ['All'] + sorted(engine.products['category'].unique().tolist())
    selected_cat = st.selectbox(
        "Filter by category:", categories, key='pop_cat'
    )
    top_n = st.slider("Number of products:", 5, 50, 10, key='pop_n')

    cat_filter = None if selected_cat == 'All' else selected_cat
    popular = engine.get_popular_products(top_n=top_n, category=cat_filter)

    if not popular.empty:
        st.dataframe(
            popular,
            use_container_width=True,
            hide_index=True,
            column_config={
                'product_id': 'Product ID',
                'title': 'Product Name',
                'category': 'Category',
                'avg_price': st.column_config.NumberColumn(
                    'Avg Price (£)', format='£%.2f'
                ),
                'n_buyers': 'Unique Buyers',
                'avg_rating': st.column_config.NumberColumn(
                    'Avg Rating', format='%.2f'
                ),
            }
        )
    else:
        st.info("No products found for the selected category.")


def page_model_evaluation(engine):
    st.header("📈 Model Evaluation")
    st.markdown(
        '<div class="algo-card">'
        '<h4>Offline Evaluation Framework</h4>'
        '<p>Evaluates all four recommendation algorithms using a held-out '
        'test set. For each user, 20% of interactions are hidden and the '
        'algorithms are measured on how well they recover those items.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test set size:", 0.1, 0.4, 0.2, 0.05,
            help="Fraction of each user's interactions held out for testing"
        )
    with col2:
        k_options = st.multiselect(
            "K values for @K metrics:",
            [3, 5, 10, 15, 20],
            default=[5, 10, 20]
        )

    if not k_options:
        st.warning("Please select at least one K value.")
        return

    if st.button("🚀 Run Evaluation", type="primary"):
        with st.status("Running evaluation...", expanded=True) as status:
            st.write("Splitting data into train/test sets...")
            st.write("Evaluating User-User CF, Item-Item CF, "
                     "Content-Based, Popularity...")
            st.write("Computing Precision@K, Recall@K, NDCG@K, "
                     "Hit Rate, Coverage...")

            results, meta = run_evaluation(
                engine, test_size, tuple(sorted(k_options))
            )

            status.update(
                label="Evaluation complete!", state="complete", expanded=False
            )

        st.success(
            f"Evaluated {meta['total_test_users']} users with "
            f"{meta['test_size']:.0%} test split."
        )

        tabs = st.tabs([f"K = {k}" for k in sorted(k_options)])

        for tab, k in zip(tabs, sorted(k_options)):
            with tab:
                rows = []
                for algo_name, algo_results in results.items():
                    if k in algo_results:
                        row = {'Algorithm': algo_name}
                        row.update(algo_results[k])
                        rows.append(row)

                if not rows:
                    st.info(f"No results for K={k}")
                    continue

                df = pd.DataFrame(rows)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Precision@K': st.column_config.NumberColumn(format='%.4f'),
                        'Recall@K': st.column_config.NumberColumn(format='%.4f'),
                        'NDCG@K': st.column_config.NumberColumn(format='%.4f'),
                        'Hit Rate': st.column_config.NumberColumn(format='%.4f'),
                        'Coverage': st.column_config.NumberColumn(format='%.4f'),
                    }
                )

                metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate']
                chart_df = df[['Algorithm'] + metrics].melt(
                    id_vars='Algorithm',
                    var_name='Metric',
                    value_name='Value'
                )

                fig = px.bar(
                    chart_df, x='Metric', y='Value', color='Algorithm',
                    barmode='group',
                    color_discrete_sequence=[
                        '#667eea', '#764ba2', '#f093fb', '#4facfe'
                    ],
                    title=f'Algorithm Comparison at K={k}'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(
                        title='Score',
                        gridcolor='rgba(102,126,234,0.1)',
                        range=[0, max(0.3, chart_df['Value'].max() * 1.2)]
                    ),
                    xaxis=dict(title=''),
                    legend=dict(orientation='h', y=-0.15),
                    height=450,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Algorithm Strength Profiles")

        radar_k = sorted(k_options)[len(k_options) // 2]
        radar_metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate', 'Coverage']

        fig_radar = go.Figure()
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

        for idx, (algo_name, algo_results) in enumerate(results.items()):
            if radar_k not in algo_results:
                continue
            values = [algo_results[radar_k].get(m, 0) for m in radar_metrics]
            values.append(values[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=algo_name,
                line=dict(color=colors[idx % len(colors)]),
                fillcolor=colors[idx % len(colors)].replace(')', ', 0.1)').replace(
                    '#', 'rgba(').replace(
                    'rgba(667eea', 'rgba(102,126,234').replace(
                    'rgba(764ba2', 'rgba(118,75,162').replace(
                    'rgba(f093fb', 'rgba(240,147,251').replace(
                    'rgba(4facfe', 'rgba(79,172,254'),
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation='h', y=-0.1),
            height=500,
            title=f"Strength Profile (K={radar_k})",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("💡 Key Insights")

        best_k = sorted(k_options)[0]
        if best_k in results.get('User-User CF', {}):
            best_algo = max(
                results.items(),
                key=lambda x: x[1].get(best_k, {}).get('NDCG@K', 0)
            )
            st.markdown(
                f"- **Best overall algorithm** (by NDCG@{best_k}): "
                f"**{best_algo[0]}**"
            )

        coverages = {
            name: res.get(best_k, {}).get('Coverage', 0)
            for name, res in results.items()
        }
        best_coverage = max(coverages, key=coverages.get)
        st.markdown(
            f"- **Best catalog coverage** at K={best_k}: "
            f"**{best_coverage}** ({coverages[best_coverage]:.1%})"
        )
        st.markdown(
            "- **Recommendation**: Use the Hybrid approach to balance "
            "accuracy and diversity."
        )


def page_user_analytics(engine):
    st.header("👥 User Behavior Analytics")
    st.markdown(
        '<div class="algo-card">'
        '<h4>User Segmentation & Clustering</h4>'
        '<p>Segments users into behavioral clusters using K-Means on '
        'features like purchase frequency, rating behavior, and category '
        'diversity. Inspired by RFM (Recency-Frequency-Monetary) analysis.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Segmenting users..."):
        user_df = get_user_segments(engine)

    segments = user_df['segment'].value_counts()
    cols = st.columns(len(segments))
    for i, (seg, count) in enumerate(segments.items()):
        cols[i].metric(seg, f"{count:,} users")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📊 Segment Distribution",
        "🔬 Feature Analysis",
        "📋 User Details"
    ])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Segment Sizes")
            fig = px.pie(
                user_df, names='segment',
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                hole=0.45,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Segment Profiles")
            profile_df = (
                user_df.groupby('segment')
                [['frequency', 'total_interactions', 'avg_rating', 'n_categories']]
                .mean()
                .round(2)
                .reset_index()
            )
            st.dataframe(profile_df, use_container_width=True, hide_index=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Frequency vs Avg Rating")
            fig = px.scatter(
                user_df, x='frequency', y='avg_rating',
                color='segment',
                size='total_interactions',
                color_discrete_sequence=[
                    '#667eea', '#764ba2', '#f093fb', '#4facfe'
                ],
                opacity=0.6,
                hover_data=['user_id', 'n_categories'],
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Products Purchased',
                    gridcolor='rgba(102,126,234,0.1)'
                ),
                yaxis=dict(
                    title='Average Rating',
                    gridcolor='rgba(102,126,234,0.1)'
                ),
                legend=dict(orientation='h', y=-0.15),
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Category Diversity vs Interactions")
            fig = px.scatter(
                user_df, x='n_categories', y='total_interactions',
                color='segment',
                color_discrete_sequence=[
                    '#667eea', '#764ba2', '#f093fb', '#4facfe'
                ],
                opacity=0.6,
                hover_data=['user_id', 'avg_rating'],
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Categories Explored',
                    gridcolor='rgba(102,126,234,0.1)'
                ),
                yaxis=dict(
                    title='Total Interactions',
                    gridcolor='rgba(102,126,234,0.1)'
                ),
                legend=dict(orientation='h', y=-0.15),
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Distributions by Segment")
        feature_to_plot = st.selectbox(
            "Select feature:",
            ['frequency', 'total_interactions', 'avg_rating',
             'rating_std', 'n_categories']
        )

        fig = px.box(
            user_df, x='segment', y=feature_to_plot,
            color='segment',
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(title=''),
            yaxis=dict(
                title=feature_to_plot.replace('_', ' ').title(),
                gridcolor='rgba(102,126,234,0.1)'
            ),
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("User Lookup")
        segment_filter = st.selectbox(
            "Filter by segment:",
            ['All'] + sorted(user_df['segment'].unique().tolist()),
            key='seg_filter'
        )

        if segment_filter == 'All':
            display_df = user_df
        else:
            display_df = user_df[user_df['segment'] == segment_filter]

        sort_by = st.selectbox(
            "Sort by:", ['frequency', 'total_interactions', 'avg_rating'],
            key='seg_sort'
        )

        st.dataframe(
            display_df.sort_values(sort_by, ascending=False).head(100)[
                ['user_id', 'segment', 'frequency', 'total_interactions',
                 'avg_rating', 'rating_std', 'n_categories']
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                'user_id': 'User ID',
                'segment': 'Segment',
                'frequency': 'Products Purchased',
                'total_interactions': 'Total Interactions',
                'avg_rating': st.column_config.NumberColumn(
                    'Avg Rating', format='%.2f'
                ),
                'rating_std': st.column_config.NumberColumn(
                    'Rating Std Dev', format='%.2f'
                ),
                'n_categories': 'Categories Explored',
            }
        )


def main():
    st.title("🛒 E-Commerce Recommendation Engine")
    st.caption("Powered by real Amazon / Online Retail data from Kaggle")

    try:
        engine = load_engine()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "**How to set up:**\n"
            "1. Download the dataset from "
            "[Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)\n"
            "2. Extract the CSV file\n"
            "3. Place it inside the `data/` folder\n"
            "4. Refresh this page"
        )
        return

    st.sidebar.header("🧭 Navigation")

    st.sidebar.markdown("**📋 Overview**")
    option = st.sidebar.radio(
        "Choose a feature:",
        [
            "📊 Dataset Overview",
            "───────────────",
            "🔍 Similar Products",
            "👤 User-User CF",
            "🔗 Item-Item CF",
            "🔀 Hybrid Engine",
            "🔥 Popular Products",
            "────────────────",
            "📈 Model Evaluation",
            "👥 User Analytics",
        ],
        label_visibility="collapsed"
    )

    if option.startswith("──"):
        st.info("👈 Please select a feature from the sidebar.")
        return

    if option == "📊 Dataset Overview":
        page_dataset_overview(engine)
    elif option == "🔍 Similar Products":
        page_similar_products(engine)
    elif option == "👤 User-User CF":
        page_user_user_recs(engine)
    elif option == "🔗 Item-Item CF":
        page_item_item_recs(engine)
    elif option == "🔀 Hybrid Engine":
        page_hybrid_recs(engine)
    elif option == "🔥 Popular Products":
        page_popular_products(engine)
    elif option == "📈 Model Evaluation":
        page_model_evaluation(engine)
    elif option == "👥 User Analytics":
        page_user_analytics(engine)


if __name__ == "__main__":
    main()
