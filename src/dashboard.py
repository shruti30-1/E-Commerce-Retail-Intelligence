"""
Phase 7 — Streamlit Dashboard
Retail Demand & Customer Intelligence Platform
Olist Brazilian E-Commerce Dataset

Run with:  streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Platform | Olist",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #9b9ec4 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }

    /* Section headers */
    h1 { color: #7c8cf8 !important; }
    h2 { color: #a78bfa !important; border-bottom: 1px solid #2a2d3e; padding-bottom: 8px; }
    h3 { color: #c4b5fd !important; }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly { border-radius: 12px; }

    /* Tab styling */
    [data-testid="stTab"] { font-weight: 600; }

    /* Dividers */
    hr { border-color: #2a2d3e; }

    /* Info boxes */
    .stInfo { background: rgba(124, 140, 248, 0.1); border: 1px solid #7c8cf8; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE CONNECTION (cached for performance)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to database...")
def get_engine():
    """Create a cached SQLAlchemy engine. Cached so we don't reconnect on every rerun."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    if not url:
        st.error("SUPABASE_URL not found in .env file. Please check your environment variables.")
        st.stop()
    return create_engine(url)

@st.cache_data(ttl=3600, show_spinner="Loading data from Supabase...")
def load_data(_engine):
    """
    Load and cache all analytical data from Supabase.
    ttl=3600 means data refreshes every hour automatically.
    Returns a dict of DataFrames.
    """
    data = {}

    # KPI summary
    data['kpis'] = pd.read_sql("""
        SELECT
            COUNT(DISTINCT o.order_id)                                     AS total_orders,
            COUNT(DISTINCT c.customer_unique_id)                           AS unique_customers,
            ROUND(SUM(p.payment_value)::numeric, 2)                        AS total_gmv,
            ROUND(AVG(p.payment_value)::numeric, 2)                        AS avg_order_value,
            ROUND(AVG(r.review_score)::numeric, 2)                         AS avg_review_score,
            ROUND(AVG(EXTRACT(EPOCH FROM (
                o.order_delivered_customer_date::timestamp -
                o.order_purchase_timestamp::timestamp
            ))/86400.0)::numeric, 1)                                       AS avg_delivery_days
        FROM orders o
        LEFT JOIN customers c   ON o.customer_id = c.customer_id
        LEFT JOIN payments p    ON o.order_id    = p.order_id
        LEFT JOIN reviews r     ON o.order_id    = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
    """, _engine)

    # Monthly GMV trend
    data['monthly'] = pd.read_sql("""
        SELECT
            DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS month,
            COUNT(DISTINCT o.order_id)    AS order_count,
            ROUND(SUM(p.payment_value)::numeric, 2) AS revenue
        FROM orders o
        LEFT JOIN payments p ON o.order_id = p.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """, _engine)

    # Top categories
    data['categories'] = pd.read_sql("""
        SELECT
            COALESCE(ct.product_category_name_english, 'Unknown') AS category,
            COUNT(DISTINCT oi.order_id)                            AS orders,
            ROUND(SUM(oi.price)::numeric, 2)                      AS revenue
        FROM order_items oi
        JOIN products pr ON oi.product_id = pr.product_id
        LEFT JOIN category_translation ct ON pr.product_category_name = ct.product_category_name
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT 20
    """, _engine)

    # Orders by state
    data['states'] = pd.read_sql("""
        SELECT
            c.customer_state,
            COUNT(DISTINCT o.order_id)              AS orders,
            ROUND(SUM(p.payment_value)::numeric, 2) AS revenue,
            ROUND(AVG(r.review_score)::numeric, 2)  AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN payments p ON o.order_id = p.order_id
        LEFT JOIN reviews r  ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY 1
        ORDER BY orders DESC
    """, _engine)

    # Review score distribution
    data['reviews'] = pd.read_sql("""
        SELECT
            review_score,
            COUNT(*) AS count
        FROM reviews
        WHERE review_score IS NOT NULL
        GROUP BY review_score
        ORDER BY review_score
    """, _engine)

    # Delivery delay vs review score
    data['delay_review'] = pd.read_sql("""
        SELECT
            r.review_score,
            ROUND(AVG(EXTRACT(EPOCH FROM (
                o.order_delivered_customer_date::timestamp -
                o.order_estimated_delivery_date::timestamp
            ))/86400.0)::numeric, 2) AS avg_delay_days,
            COUNT(*) AS order_count
        FROM orders o
        JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """, _engine)

    # Payment method breakdown
    data['payments'] = pd.read_sql("""
        SELECT
            payment_type,
            COUNT(DISTINCT order_id)                AS orders,
            ROUND(AVG(payment_value)::numeric, 2)   AS avg_value,
            ROUND(SUM(payment_value)::numeric, 2)   AS total_value
        FROM payments
        GROUP BY payment_type
        ORDER BY orders DESC
    """, _engine)

    # Seller performance
    data['sellers'] = pd.read_sql("""
        SELECT
            s.seller_id,
            s.seller_state,
            COUNT(DISTINCT oi.order_id)              AS total_orders,
            ROUND(SUM(oi.price)::numeric, 2)         AS total_revenue,
            ROUND(AVG(r.review_score)::numeric, 2)   AS avg_review
        FROM sellers s
        JOIN order_items oi ON s.seller_id = oi.seller_id
        LEFT JOIN reviews r ON oi.order_id = r.order_id
        GROUP BY s.seller_id, s.seller_state
        HAVING COUNT(DISTINCT oi.order_id) > 10
        ORDER BY total_revenue DESC
        LIMIT 100
    """, _engine)

    return data


# ─────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(30,34,53,0.8)",
    plot_bgcolor="rgba(30,34,53,0.0)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
engine = get_engine()
data = load_data(engine)
kpi = data['kpis'].iloc[0]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size: 2.5rem;">🛒</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #7c8cf8;">Retail Intelligence</div>
        <div style="font-size: 0.75rem; color: #9b9ec4;">Olist E-Commerce Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Executive Overview", "📈 Revenue Trends", "👥 Customer Segments",
         "🚚 Delivery & Reviews", "🏆 Seller Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #9b9ec4; text-align: center;">
        Dataset: Olist Brazilian E-Commerce<br>
        Source: Kaggle Public Dataset<br>
        Period: 2016 – 2018
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown("*Key performance indicators across the entire Olist marketplace*")
    st.markdown("---")

    # KPI Row 1
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Orders",        f"{int(kpi['total_orders']):,}")
    c2.metric("Unique Customers",     f"{int(kpi['unique_customers']):,}")
    c3.metric("Total GMV (BRL)",      f"R$ {float(kpi['total_gmv']):,.0f}")
    c4.metric("Avg Order Value",      f"R$ {float(kpi['avg_order_value']):.2f}")
    c5.metric("Avg Review Score",     f"⭐ {float(kpi['avg_review_score']):.2f} / 5")
    c6.metric("Avg Delivery (days)",  f"{float(kpi['avg_delivery_days']):.1f}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📅 Monthly Order Volume")
        df_m = data['monthly'].copy()
        df_m['month'] = pd.to_datetime(df_m['month'])
        df_m = df_m.iloc[1:-1]  # remove partial months

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_m['month'], y=df_m['order_count'],
            mode='lines+markers', name='Orders',
            line=dict(color='#7c8cf8', width=2.5),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(124,140,248,0.1)'
        ))
        fig.update_layout(**PLOTLY_THEME, height=320,
                          xaxis_title="Month", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 💳 Payment Methods")
        df_pay = data['payments']
        fig2 = px.pie(df_pay, values='orders', names='payment_type',
                      color_discrete_sequence=px.colors.sequential.Viridis,
                      hole=0.45)
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           legend=dict(orientation="h", y=-0.1))
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    # Category bar chart
    st.markdown("### 🛍️ Top Product Categories by Revenue")
    df_cat = data['categories'].head(15)
    fig3 = px.bar(df_cat, x='revenue', y='category', orientation='h',
                  color='revenue', color_continuous_scale='Viridis',
                  labels={'revenue': 'Revenue (BRL)', 'category': ''},
                  text=df_cat['revenue'].apply(lambda x: f"R$ {x/1000:.0f}K"))
    fig3.update_layout(**PLOTLY_THEME, height=480,
                       coloraxis_showscale=False,
                       yaxis=dict(autorange='reversed'))
    fig3.update_traces(textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: REVENUE TRENDS
# ─────────────────────────────────────────────
elif page == "📈 Revenue Trends":
    st.markdown("# 📈 Revenue Trends")
    st.markdown("*Monthly GMV and order volume trends across the Olist marketplace*")
    st.markdown("---")

    df_m = data['monthly'].copy()
    df_m['month'] = pd.to_datetime(df_m['month'])
    df_m = df_m.iloc[1:-1]
    df_m['mom_growth'] = df_m['order_count'].pct_change() * 100
    df_m['revenue_k'] = df_m['revenue'] / 1000

    # Revenue + Orders dual axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_m['month'], y=df_m['revenue_k'],
                         name='Revenue (BRL 000s)', marker_color='rgba(124,140,248,0.7)'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_m['month'], y=df_m['order_count'],
                              name='Order Count', mode='lines+markers',
                              line=dict(color='#f97316', width=2.5), marker=dict(size=7)),
                  secondary_y=True)
    fig.update_layout(**PLOTLY_THEME, height=400,
                      title="Monthly Revenue & Order Volume",
                      legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="Revenue (BRL 000s)", secondary_y=False)
    fig.update_yaxes(title_text="Order Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # MoM Growth
    st.markdown("### Month-over-Month Order Growth (%)")
    colors_growth = ['#e74c3c' if v < 0 else '#2ecc71' for v in df_m['mom_growth'].fillna(0)]
    fig2 = go.Figure(go.Bar(
        x=df_m['month'], y=df_m['mom_growth'],
        marker_color=colors_growth,
        text=df_m['mom_growth'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else ""),
        textposition='outside'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(**PLOTLY_THEME, height=350,
                       yaxis_title="MoM Growth (%)")
    st.plotly_chart(fig2, use_container_width=True)

    # State heatmap
    st.markdown("### 🗺️ Revenue by Brazilian State")
    df_st = data['states']
    fig3 = px.choropleth(
        df_st,
        locations='customer_state',
        locationmode='geojson-id',
        color='revenue',
        hover_name='customer_state',
        hover_data={'orders': True, 'avg_score': True},
        color_continuous_scale='Viridis',
        scope='south america',
        labels={'revenue': 'Revenue (BRL)', 'customer_state': 'State'}
    )
    fig3.update_geos(
        showcoastlines=True, coastlinecolor="Gray",
        showland=True, landcolor="rgb(30,34,53)",
        showocean=True, oceancolor="rgb(15,17,23)",
        center={"lat": -14.2, "lon": -51.9}, projection_scale=3
    )
    fig3.update_layout(**PLOTLY_THEME, height=500,
                       title="Orders & Revenue Distribution Across Brazilian States")
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.markdown("**Top 10 States by Revenue**")
    col1.dataframe(df_st[['customer_state','orders','revenue','avg_score']].head(10).rename(
        columns={'customer_state':'State','orders':'Orders','revenue':'Revenue (BRL)','avg_score':'Avg ⭐'}
    ).style.background_gradient(cmap='Blues'), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: CUSTOMER SEGMENTS
# ─────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.markdown("# 👥 Customer Segments")
    st.markdown("*RFM-based customer segmentation from Phase 3 & 4 analysis*")
    st.markdown("---")

    # Try to load segments CSV
    seg_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_segments.csv')
    feat_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_features.csv')

    if os.path.exists(seg_path):
        df_seg = pd.read_csv(seg_path)
        st.success(f"Loaded {len(df_seg):,} customer records with segment labels.")

        # Segment distribution
        col1, col2 = st.columns([2, 3])
        with col1:
            seg_counts = df_seg['segment'].value_counts().reset_index()
            seg_counts.columns = ['segment', 'count']
            palette = {
                'Champions': '#2ecc71', 'Loyal Customers': '#3498db',
                'At-Risk': '#e67e22', 'Lost / Inactive': '#e74c3c',
                'Others': '#9b9ec4'
            }
            fig_pie = px.pie(seg_counts, values='count', names='segment',
                             color='segment', color_discrete_map=palette, hole=0.5)
            fig_pie.update_layout(**PLOTLY_THEME, height=350, title="Segment Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # RFM profile per segment
            numeric_cols = ['recency_days','frequency','monetary_value','avg_order_value']
            available = [c for c in numeric_cols if c in df_seg.columns]
            if available:
                profile = df_seg.groupby('segment')[available].mean().round(2)
                st.markdown("**Segment Profiles (Mean Values)**")
                st.dataframe(profile.style.background_gradient(cmap='RdYlGn', axis=0),
                             use_container_width=True)

        # Scatter: Recency vs Monetary
        if 'recency_days' in df_seg.columns and 'monetary_value' in df_seg.columns:
            st.markdown("### RFM Scatter: Recency vs Monetary Value")
            df_sample = df_seg.sample(min(5000, len(df_seg)), random_state=42)
            fig_sc = px.scatter(df_sample, x='recency_days', y='monetary_value',
                                color='segment', color_discrete_map=palette,
                                opacity=0.5, size_max=8,
                                labels={'recency_days': 'Recency (days since last order)',
                                        'monetary_value': 'Total Spend (BRL)'},
                                title="Customer Scatter: Recency vs Total Spend")
            fig_sc.update_layout(**PLOTLY_THEME, height=450)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Business recommendations
        st.markdown("### 💡 Marketing Recommendations by Segment")
        recs = {
            'Champions':       ('🏆', '#2ecc71', 'Reward & Upsell', 'Loyalty program, early access, referral bonuses'),
            'Loyal Customers': ('❤️', '#3498db', 'Nurture & Retain', 'Personalized offers, birthday discounts, subscriptions'),
            'At-Risk':         ('⚠️', '#e67e22', 'Re-engage',       'Win-back campaign, 20% coupon, churn survey'),
            'Lost / Inactive': ('💤', '#e74c3c', 'Last attempt',    'Single re-engagement push, exclude from paid campaigns'),
        }
        rcols = st.columns(4)
        for i, (seg, (icon, color, action, tactic)) in enumerate(recs.items()):
            count = seg_counts[seg_counts['segment'] == seg]['count'].sum() if seg in seg_counts['segment'].values else 0
            pct = count / len(df_seg) * 100 if len(df_seg) > 0 else 0
            rcols[i].markdown(f"""
            <div style="background:rgba(30,34,53,0.8);border:1px solid {color};border-radius:10px;padding:14px;text-align:center;">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="color:{color};font-weight:700;font-size:1rem">{seg}</div>
                <div style="color:#9b9ec4;font-size:0.75rem">{count:,} customers ({pct:.1f}%)</div>
                <hr style="border-color:{color};margin:8px 0">
                <div style="font-size:0.8rem;font-weight:600;color:#e0e0e0">{action}</div>
                <div style="font-size:0.72rem;color:#9b9ec4;margin-top:4px">{tactic}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("""
        **Customer segments file not found.**
        Run `notebooks/04_rfm_feature_engineering.ipynb` and 
        `notebooks/05_customer_segmentation.ipynb` first to generate 
        `data/customer_segments.csv`.
        """)
        st.info("Once generated, this page will show interactive customer segment analysis.")

# ─────────────────────────────────────────────
# PAGE: DELIVERY & REVIEWS
# ─────────────────────────────────────────────
elif page == "🚚 Delivery & Reviews":
    st.markdown("# 🚚 Delivery Performance & Review Analysis")
    st.markdown("*How delivery experience drives customer satisfaction*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⭐ Review Score Distribution")
        df_rev = data['reviews']
        colors_rev = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60']
        fig_rev = px.bar(df_rev, x='review_score', y='count',
                         color='review_score',
                         color_discrete_sequence=colors_rev,
                         labels={'review_score': 'Review Score', 'count': 'Number of Reviews'})
        fig_rev.update_layout(**PLOTLY_THEME, height=350,
                              showlegend=False)
        for _, row in df_rev.iterrows():
            fig_rev.add_annotation(x=row['review_score'], y=row['count'] + 500,
                                   text=f"{row['count']:,}", showarrow=False, font=dict(size=10))
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        st.markdown("### 📦 Avg Delivery Delay by Review Score")
        df_dr = data['delay_review']
        bar_colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60']
        fig_dr = go.Figure(go.Bar(
            x=df_dr['review_score'], y=df_dr['avg_delay_days'],
            marker_color=bar_colors,
            text=df_dr['avg_delay_days'].apply(lambda x: f"{x:+.1f}d"),
            textposition='outside'
        ))
        fig_dr.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        fig_dr.add_annotation(x=1.5, y=fig_dr.full_figure_for_development().layout.yaxis.range[1] if hasattr(fig_dr, 'full_figure_for_development') else 10,
                               text="Positive = late delivery", showarrow=False,
                               font=dict(color='#9b9ec4', size=10))
        fig_dr.update_layout(**PLOTLY_THEME, height=350,
                              xaxis_title="Review Score",
                              yaxis_title="Avg Delay vs Estimated (days)")
        st.plotly_chart(fig_dr, use_container_width=True)

    # Insight callout
    max_delay = df_dr.loc[df_dr['review_score'] == 1, 'avg_delay_days'].values
    if len(max_delay) > 0:
        st.info(f"📌 **Key Insight:** Orders with 1-star reviews have an average delivery delay of **{max_delay[0]:+.1f} days** compared to estimated. "
                f"On-time or early delivery correlates strongly with 4-5 star reviews.")

    # State-level delivery performance
    st.markdown("### 🗺️ Delivery Performance by State")
    df_state_delivery = pd.read_sql("""
        SELECT
            c.customer_state,
            COUNT(*)                                                    AS total_orders,
            ROUND(AVG(EXTRACT(EPOCH FROM (
                o.order_delivered_customer_date::timestamp -
                o.order_purchase_timestamp::timestamp
            ))/86400.0)::numeric, 1)                                   AS avg_delivery_days,
            ROUND((SUM(CASE
                WHEN o.order_delivered_customer_date::timestamp > o.order_estimated_delivery_date::timestamp
                THEN 1 ELSE 0 END)::float / COUNT(*) * 100)::numeric, 1) AS late_pct
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
        GROUP BY c.customer_state
        HAVING COUNT(*) > 100
        ORDER BY late_pct DESC
    """, engine)

    fig_del = px.scatter(df_state_delivery,
                         x='avg_delivery_days', y='late_pct',
                         size='total_orders', color='late_pct',
                         text='customer_state',
                         color_continuous_scale='RdYlGn_r',
                         labels={'avg_delivery_days': 'Avg Delivery Time (days)',
                                 'late_pct': 'Late Delivery Rate (%)',
                                 'total_orders': 'Total Orders'},
                         title="State-Level Delivery Performance: Speed vs On-Time Rate")
    fig_del.update_traces(textposition='top center')
    fig_del.update_layout(**PLOTLY_THEME, height=500)
    st.plotly_chart(fig_del, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: SELLER ANALYTICS
# ─────────────────────────────────────────────
elif page == "🏆 Seller Analytics":
    st.markdown("# 🏆 Seller Analytics")
    st.markdown("*Top seller performance: revenue, order volume, and customer satisfaction*")
    st.markdown("---")

    df_sell = data['sellers']

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Sellers (>10 orders)", f"{len(df_sell):,}")
    col2.metric("Avg Revenue per Seller", f"R$ {df_sell['total_revenue'].mean():,.0f}")
    col3.metric("Avg Review Score", f"⭐ {df_sell['avg_review'].mean():.2f}")

    st.markdown("---")

    # Bubble chart: revenue vs review score
    fig_bub = px.scatter(df_sell, x='avg_review', y='total_revenue',
                          size='total_orders', color='seller_state',
                          hover_data=['seller_id', 'total_orders'],
                          labels={'avg_review': 'Avg Review Score',
                                  'total_revenue': 'Total Revenue (BRL)',
                                  'total_orders': 'Total Orders'},
                          title="Seller Performance: Revenue vs Customer Satisfaction")
    fig_bub.add_vline(x=4.0, line_dash="dash", line_color="yellow",
                       annotation_text="Quality threshold (4★)", annotation_position="top right")
    fig_bub.update_layout(**PLOTLY_THEME, height=500)
    st.plotly_chart(fig_bub, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏆 Top 10 Sellers by Revenue")
        top_sellers = df_sell.nlargest(10, 'total_revenue')[['seller_id','seller_state','total_orders','total_revenue','avg_review']]
        top_sellers['seller_id'] = top_sellers['seller_id'].str[:8] + '...'
        top_sellers.columns = ['Seller ID','State','Orders','Revenue (BRL)','⭐ Avg']
        st.dataframe(top_sellers.style.background_gradient(subset=['Revenue (BRL)'], cmap='Blues'),
                     use_container_width=True)

    with col_b:
        st.markdown("### ⚠️ Low Quality Sellers (>50 orders, ⭐<3.5)")
        low_q = df_sell[(df_sell['avg_review'] < 3.5) & (df_sell['total_orders'] > 50)].nsmallest(10, 'avg_review')
        if len(low_q) > 0:
            low_q_display = low_q[['seller_id','seller_state','total_orders','total_revenue','avg_review']].copy()
            low_q_display['seller_id'] = low_q_display['seller_id'].str[:8] + '...'
            low_q_display.columns = ['Seller ID','State','Orders','Revenue (BRL)','⭐ Avg']
            st.dataframe(low_q_display.style.background_gradient(subset=['⭐ Avg'], cmap='Reds_r'),
                         use_container_width=True)
        else:
            st.success("No sellers with >50 orders and avg score below 3.5. Marketplace quality looks good!")

    # Revenue by state (sellers)
    st.markdown("### 📍 Revenue by Seller State")
    state_rev = df_sell.groupby('seller_state').agg(
        revenue=('total_revenue', 'sum'),
        sellers=('seller_id', 'count')
    ).sort_values('revenue', ascending=False).reset_index()

    fig_st = px.bar(state_rev, x='seller_state', y='revenue',
                     color='revenue', color_continuous_scale='Viridis',
                     labels={'seller_state': 'Seller State', 'revenue': 'Total Revenue (BRL)'},
                     title="Total Seller Revenue by State")
    fig_st.update_layout(**PLOTLY_THEME, height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_st, use_container_width=True)
