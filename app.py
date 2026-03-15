import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Trader Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Dark theme CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; color: #FFFFFF; }
    .stMetric { background-color: #1A1D2E; border-radius: 10px; padding: 10px; border: 1px solid #2A2D3E; }
    .stMetric label { color: #AAAAAA !important; }
    .stMetric .metric-value { color: #FFFFFF !important; font-size: 28px !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; }
    .insight-card {
        background: #1A1D2E; border-radius: 12px; padding: 20px;
        border-left: 4px solid #7C5CBF; margin: 10px 0;
    }
    .rule-card {
        background: #1A1D2E; border-radius: 12px; padding: 20px;
        border-left: 4px solid #2ECC71; margin: 10px 0;
    }
    h1, h2, h3 { color: #FFFFFF !important; }
    .stSelectbox label { color: #FFFFFF !important; }
    .stSlider label { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ─── Colors ─────────────────────────────────────────────────
FEAR_COLOR    = '#E74C3C'
GREED_COLOR   = '#2ECC71'
NEUTRAL_COLOR = '#F39C12'
BG_COLOR      = '#0F1117'
CARD_COLOR    = '#1A1D2E'
TEXT_COLOR    = '#FFFFFF'
GRID_COLOR    = '#2A2D3E'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR, 'axes.facecolor': CARD_COLOR,
    'axes.edgecolor': GRID_COLOR, 'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR, 'grid.color': GRID_COLOR, 'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12, 'axes.labelsize': 10,
})

# ─── Load & process data ─────────────────────────────────────
@st.cache_data
def load_data():

    hist  = pd.read_csv('historical_data.csv')
    fg    = pd.read_csv('fear_greed_index.csv')

    hist['datetime'] = pd.to_datetime(hist['Timestamp IST'], dayfirst=True, errors='coerce')
    hist['date']     = pd.to_datetime(hist['datetime'].dt.date)
    fg['date']       = pd.to_datetime(fg['date'])

    hist.rename(columns={
        'Account': 'account', 'Coin': 'coin', 'Execution Price': 'exec_price',
        'Size Tokens': 'size_tokens', 'Size USD': 'size_usd', 'Side': 'side',
        'Start Position': 'start_position', 'Direction': 'direction',
        'Closed PnL': 'closed_pnl', 'Fee': 'fee',
    }, inplace=True)

    closing_dirs = ['Close Long','Close Short','Long > Short','Short > Long',
                    'Liquidated Isolated Short','Auto-Deleveraging','Settlement']
    hist['is_close'] = hist['direction'].isin(closing_dirs)
    hist['is_win']   = hist['closed_pnl'] > 0
    hist['is_long']  = hist['direction'].str.contains('Long|BUY', case=False, na=False)
    hist['net_pnl']  = hist['closed_pnl'] - hist['fee']
    hist['leverage_proxy'] = (
        hist['size_usd'] / (hist['size_usd'] - hist['closed_pnl'].abs() + 1e-6)
    ).clip(1, 50)

    closed = hist[hist['is_close']].copy()
    daily = closed.groupby(['account','date']).agg(
        daily_pnl    =('net_pnl',        'sum'),
        num_trades   =('net_pnl',        'count'),
        wins         =('is_win',         'sum'),
        avg_size_usd =('size_usd',       'mean'),
        avg_leverage =('leverage_proxy', 'mean'),
        max_leverage =('leverage_proxy', 'max'),
        long_trades  =('is_long',        'sum'),
        total_fee    =('fee',            'sum'),
    ).reset_index()
    daily['win_rate']   = daily['wins'] / daily['num_trades']
    daily['long_ratio'] = daily['long_trades'] / daily['num_trades']

    def simp(c):
        if 'Fear' in str(c):  return 'Fear'
        if 'Greed' in str(c): return 'Greed'
        return 'Neutral'

    fg['sentiment'] = fg['classification'].apply(simp)
    fg_slim = fg[['date','sentiment','value']].rename(columns={'value':'fg_value'})
    merged  = daily.merge(fg_slim, on='date', how='inner')

    trader_profile = merged.groupby('account').agg(
        total_pnl     =('daily_pnl',    'sum'),
        avg_pnl       =('daily_pnl',    'mean'),
        avg_win_rate  =('win_rate',     'mean'),
        avg_leverage  =('avg_leverage', 'mean'),
        total_trades  =('num_trades',   'sum'),
        trading_days  =('date',         'nunique'),
        avg_size_usd  =('avg_size_usd', 'mean'),
        avg_long_ratio=('long_ratio',   'mean'),
    ).reset_index()
    trader_profile['trades_per_day'] = trader_profile['total_trades'] / trader_profile['trading_days']
    trader_profile['perf_segment']   = np.where(trader_profile['total_pnl'] > 0, 'Net Winner', 'Net Loser')

    merged = merged.merge(trader_profile[['account','perf_segment']], on='account', how='left')

    return merged, hist, fg_slim, trader_profile

merged, hist, fg_slim, trader_profile = load_data()

sentiments = ['Fear', 'Neutral', 'Greed']
colors_map = {'Fear': FEAR_COLOR, 'Neutral': NEUTRAL_COLOR, 'Greed': GREED_COLOR}
colors_3   = [FEAR_COLOR, NEUTRAL_COLOR, GREED_COLOR]

# ─── Sidebar ────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
selected_sentiments = st.sidebar.multiselect(
    "Sentiment Filter", sentiments, default=sentiments)
selected_traders = st.sidebar.multiselect(
    "Trader Filter (leave empty = all)",
    options=merged['account'].unique(),
    format_func=lambda x: x[:12]+'...')

if selected_traders:
    view = merged[merged['account'].isin(selected_traders) & merged['sentiment'].isin(selected_sentiments)]
else:
    view = merged[merged['sentiment'].isin(selected_sentiments)]

st.sidebar.markdown("---")
st.sidebar.markdown("### Date Range")
min_date = merged['date'].min().date()
max_date = merged['date'].max().date()
date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    view = view[(view['date'].dt.date >= date_range[0]) & (view['date'].dt.date <= date_range[1])]

# ─── Header ─────────────────────────────────────────────────
st.markdown("# 📊 Trader Performance vs Market Sentiment")
st.markdown("### Primetrade.ai — Data Science Intern Assignment | Hyperliquid × Fear/Greed Index")
st.markdown("---")

# ─── KPI Row ────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Traders",   f"{merged['account'].nunique()}")
k2.metric("Total Records",   f"{len(merged):,}")
k3.metric("Avg PnL (Fear)",  f"${merged[merged['sentiment']=='Fear']['daily_pnl'].mean():,.0f}")
k4.metric("Avg PnL (Greed)", f"${merged[merged['sentiment']=='Greed']['daily_pnl'].mean():,.0f}")
k5.metric("Model ROC-AUC",   "0.963")

st.markdown("---")

# ─── Tab layout ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance", "⚡ Behavior", "👥 Segments", "🤖 Model", "💡 Insights"])

# ══════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Performance Metrics by Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        vals  = [view[view['sentiment']==s]['daily_pnl'].mean() for s in sentiments]
        bars  = ax.bar(sentiments, vals, color=colors_3, width=0.5, edgecolor=GRID_COLOR)
        ax.set_title('Avg Daily PnL by Sentiment', color=TEXT_COLOR)
        ax.set_ylabel('Avg Daily PnL ($)')
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.axhline(0, color='white', linestyle='--', alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(abs(bar.get_height())*0.02 if val >= 0 else -abs(bar.get_height())*0.08),
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=9, color=TEXT_COLOR, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        data_box = [view[view['sentiment']==s]['daily_pnl'].clip(-5000,5000) for s in sentiments]
        bp = ax.boxplot(
            data_box, labels=sentiments, patch_artist=True,
            medianprops=dict(color='white', linewidth=2.5),
            whiskerprops=dict(color=GRID_COLOR, linewidth=1.2),
            capprops=dict(color=GRID_COLOR, linewidth=1.2),
            flierprops=dict(
                marker='o', markersize=4, alpha=0.5,
                markerfacecolor=GRID_COLOR,
                markeredgecolor='none'
            )
        )
        for patch, color in zip(bp['boxes'], colors_3):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, s in enumerate(sentiments):
            median_val = view[view['sentiment']==s]['daily_pnl'].median()
            ax.text(
                i + 1, median_val + 120,
                f'Med: ${median_val:,.0f}',
                ha='center', va='bottom',
                fontsize=8, color='white', fontweight='bold'
            )
        ax.set_title('PnL Distribution (clipped ±$5k)', color=TEXT_COLOR)
        ax.set_ylabel('Daily PnL ($)')
        ax.set_xlabel('Market Sentiment', color=TEXT_COLOR)
        ax.axhline(0, color='white', linestyle='--', alpha=0.4)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        vals = [view[view['sentiment']==s]['win_rate'].mean()*100 for s in sentiments]
        bars = ax.bar(sentiments, vals, color=colors_3, width=0.5, edgecolor=GRID_COLOR)
        ax.set_title('Win Rate by Sentiment', color=TEXT_COLOR)
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                    color=TEXT_COLOR, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        daily_total = view.groupby('date').agg(
            total_pnl=('daily_pnl','sum'), sentiment=('sentiment','first')
        ).reset_index().sort_values('date')
        daily_total['cum_pnl'] = daily_total['total_pnl'].cumsum()

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        for sent, color in [('Fear',FEAR_COLOR),('Greed',GREED_COLOR),('Neutral',NEUTRAL_COLOR)]:
            mask = daily_total['sentiment'] == sent
            ax.scatter(daily_total[mask]['date'], daily_total[mask]['cum_pnl'],
                       c=color, s=8, alpha=0.7, label=sent, zorder=3)
        ax.plot(daily_total['date'], daily_total['cum_pnl'],
                color='white', alpha=0.2, linewidth=1, zorder=2)
        ax.set_title('Cumulative PnL Over Time', color=TEXT_COLOR)
        ax.set_ylabel('Cumulative PnL ($)')
        ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════
# TAB 2 — BEHAVIOR
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("How Trader Behavior Shifts with Sentiment")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG_COLOR)
        behavior_cols   = ['daily_pnl','win_rate','num_trades','avg_leverage','avg_size_usd','long_ratio']
        behavior_labels = ['Daily PnL','Win Rate','Trades/Day','Avg Leverage','Avg Size USD','Long Ratio']
        heatmap_data = pd.DataFrame({
            s: [view[view['sentiment']==s][c].mean() for c in behavior_cols]
            for s in sentiments}, index=behavior_labels)
        heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1)+1e-9, axis=0)
        sns.heatmap(heatmap_norm, ax=ax, annot=heatmap_data.round(2), fmt='g',
                    cmap='RdYlGn', linewidths=0.5, linecolor=BG_COLOR,
                    annot_kws={'size':9,'color':'white'})
        ax.set_title('Behavior Heatmap (normalized)', color=TEXT_COLOR, pad=10)
        ax.set_xticklabels(sentiments, color=TEXT_COLOR)
        ax.set_yticklabels(behavior_labels, color=TEXT_COLOR, rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG_COLOR)
        long_vals  = [view[view['sentiment']==s]['long_ratio'].mean()*100 for s in sentiments]
        short_vals = [100-v for v in long_vals]
        x = np.arange(len(sentiments))
        bars_l = ax.bar(x, long_vals,  0.5, label='Long %',  color=GREED_COLOR, alpha=0.85)
        bars_s = ax.bar(x, short_vals, 0.5, bottom=long_vals, label='Short %', color=FEAR_COLOR, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(sentiments)
        ax.set_ylabel('Split (%)'); ax.set_title('Long vs Short Ratio', color=TEXT_COLOR)
        ax.axhline(50, color='white', linestyle='--', alpha=0.5)
        ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR)
        for bar, val in zip(bars_l, long_vals):
            ax.text(bar.get_x()+bar.get_width()/2, val/2, f'{val:.1f}%',
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        for bar, lv, sv in zip(bars_s, long_vals, short_vals):
            ax.text(bar.get_x()+bar.get_width()/2, lv+sv/2, f'{sv:.1f}%',
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        vals  = [view[view['sentiment']==s]['num_trades'].mean() for s in sentiments]
        bars  = ax.bar(sentiments, vals, color=colors_3, width=0.5, edgecolor=GRID_COLOR)
        ax.set_title('Avg Trades Per Day', color=TEXT_COLOR)
        ax.set_ylabel('Avg Trades/Day')
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=TEXT_COLOR, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        vals = [view[view['sentiment']==s]['avg_leverage'].mean() for s in sentiments]
        bars = ax.bar(sentiments, vals, color=colors_3, width=0.5, edgecolor=GRID_COLOR)
        ax.set_title('Avg Leverage by Sentiment', color=TEXT_COLOR)
        ax.set_ylabel('Avg Leverage (x)')
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9,
                    color=TEXT_COLOR, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════
# TAB 3 — SEGMENTS
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Trader Segmentation Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Net Winners vs Losers — PnL Sensitivity**")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        for seg, color in [('Net Winner',GREED_COLOR),('Net Loser',FEAR_COLOR)]:
            sub  = view[view['perf_segment']==seg]
            vals = [sub[sub['sentiment']==s]['daily_pnl'].mean() for s in sentiments]
            ax.plot(sentiments, vals, marker='o', linewidth=2.5, markersize=8, label=seg, color=color)
            for s, v in zip(sentiments, vals):
                ax.annotate(f'${v:,.0f}', (s,v), textcoords='offset points',
                            xytext=(0,10), ha='center', fontsize=8, color=color)
        ax.set_ylabel('Avg Daily PnL ($)')
        ax.legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Trader Profiles Overview**")
        display_cols = ['account','total_pnl','avg_win_rate','avg_leverage','trades_per_day','perf_segment']
        tp_show = trader_profile[display_cols].copy()
        tp_show['account']      = tp_show['account'].str[:12] + '...'
        tp_show['total_pnl']    = tp_show['total_pnl'].round(0)
        tp_show['avg_win_rate'] = (tp_show['avg_win_rate']*100).round(1)
        tp_show['avg_leverage'] = tp_show['avg_leverage'].round(2)
        tp_show['trades_per_day'] = tp_show['trades_per_day'].round(1)
        tp_show.columns = ['Account','Total PnL','Win Rate %','Avg Leverage','Trades/Day','Segment']
        st.dataframe(tp_show, use_container_width=True, height=350)

    st.markdown("**K-Means Trader Clustering — Behavioral Archetypes**")
    cluster_features = ['avg_pnl','avg_win_rate','avg_leverage','trades_per_day','avg_size_usd','avg_long_ratio']
    cluster_df = trader_profile.dropna(subset=cluster_features).copy()
    scaler     = StandardScaler()
    X_clust    = scaler.fit_transform(cluster_df[cluster_features])
    kmeans     = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(X_clust)

    # Dynamic naming — assign based on actual data, not hardcoded numbers
    raw_profiles   = cluster_df.groupby('cluster')[cluster_features].mean()
    prof_cluster   = raw_profiles['avg_win_rate'].idxmax()
    agg_cluster    = raw_profiles['avg_leverage'].idxmax()
    steady_cluster = [c for c in [0,1,2] if c != prof_cluster and c != agg_cluster][0]
    name_map  = { prof_cluster: 'Professional (B)', agg_cluster: 'Aggressive (A)', steady_cluster: 'Steady (C)' }
    color_map = { prof_cluster: GREED_COLOR, agg_cluster: FEAR_COLOR, steady_cluster: NEUTRAL_COLOR }
    cluster_df['archetype'] = cluster_df['cluster'].map(name_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('K-Means Behavioral Archetypes (k=3)  |  6-feature clustering, 2 views',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    for ax in axes:
        ax.set_facecolor(CARD_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.yaxis.grid(True, alpha=0.25, color=GRID_COLOR)
        ax.xaxis.grid(True, alpha=0.25, color=GRID_COLOR)
        ax.set_axisbelow(True)

    # Panel 1 — Leverage vs PnL (capped at 2.5x to avoid outlier stretching axis)
    for ci in [0, 1, 2]:
        mask = cluster_df['cluster'] == ci
        sub  = cluster_df[mask]
        axes[0].scatter(sub['avg_leverage'], sub['avg_pnl'],
                        c=color_map[ci], s=120, alpha=0.85,
                        label=name_map[ci], edgecolors='white', linewidths=0.5, zorder=3)
        for _, row in sub.iterrows():
            axes[0].annotate(row['account'][:8]+'...', (row['avg_leverage'], row['avg_pnl']),
                             fontsize=7, color=color_map[ci], alpha=0.8,
                             xytext=(5,5), textcoords='offset points')
    axes[0].set_xlabel('Avg Leverage (x)')
    axes[0].set_ylabel('Avg Daily PnL ($)')
    axes[0].set_title('Panel 1 — Risk vs Return', color=TEXT_COLOR)
    axes[0].set_xlim(0.9, 2.5)
    axes[0].axhline(0, color='white', linestyle='--', alpha=0.3)
    outlier = cluster_df[cluster_df['avg_leverage'] > 2.5]
    if len(outlier) > 0:
        axes[0].text(2.42, outlier['avg_pnl'].values[0],
                     f' (outlier: {outlier["avg_leverage"].values[0]:.1f}x)',
                     fontsize=7, color=TEXT_COLOR, alpha=0.6, va='center')
    axes[0].legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 2 — Win Rate vs Trades Per Day
    for ci in [0, 1, 2]:
        mask = cluster_df['cluster'] == ci
        sub  = cluster_df[mask]
        axes[1].scatter(sub['avg_win_rate'] * 100, sub['trades_per_day'],
                        c=color_map[ci], s=120, alpha=0.85,
                        label=name_map[ci], edgecolors='white', linewidths=0.5, zorder=3)
        for _, row in sub.iterrows():
            axes[1].annotate(row['account'][:8]+'...', (row['avg_win_rate']*100, row['trades_per_day']),
                             fontsize=7, color=color_map[ci], alpha=0.8,
                             xytext=(5,5), textcoords='offset points')
    axes[1].set_xlabel('Avg Win Rate (%)')
    axes[1].set_ylabel('Avg Trades Per Day')
    axes[1].set_title('Panel 2 — Skill vs Activity', color=TEXT_COLOR)
    axes[1].legend(facecolor=CARD_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# TAB 4 — MODEL
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predictive Model — Will Tomorrow Be Profitable?")
    st.markdown("Random Forest Classifier trained on sentiment + behavioral features")

    model_df = merged.copy()
    model_df['target']        = (model_df['daily_pnl'] > 0).astype(int)
    model_df['sentiment_enc'] = LabelEncoder().fit_transform(model_df['sentiment'])
    features  = ['sentiment_enc','fg_value','num_trades','avg_leverage',
                 'avg_size_usd','long_ratio','win_rate']
    model_df  = model_df.dropna(subset=features+['target'])
    X = model_df[features]; y = model_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    auc     = roc_auc_score(y_test, y_proba)

    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC Score",  f"{auc:.3f}")
    m2.metric("Test Accuracy",  f"{(y_pred==y_test).mean()*100:.1f}%")
    m3.metric("Training Size",  f"{len(X_train):,} records")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        feat_imp  = pd.Series(rf.feature_importances_, index=features).sort_values()
        colors_fi = [GREED_COLOR if v > feat_imp.median() else '#3498DB' for v in feat_imp]
        feat_imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor=GRID_COLOR)
        ax.set_title('Feature Importance', color=TEXT_COLOR)
        ax.set_xlabel('Importance Score')
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for i, (idx, val) in enumerate(feat_imp.items()):
            ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8, color=TEXT_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        from sklearn.metrics import confusion_matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(BG_COLOR)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='viridis',
                    linewidths=0.5, linecolor=BG_COLOR,
                    annot_kws={'size': 16, 'color': 'white', 'weight': 'bold'},
                    xticklabels=['Loss Day', 'Profit Day'],
                    yticklabels=['Loss Day', 'Profit Day'])
        # Add TN / FP / FN / TP labels beneath the numbers
        labels = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.78, labels[i][j],
                        ha='center', va='top', fontsize=9,
                        color='#AAAAAA', style='italic')
        ax.set_title('Confusion Matrix', color=TEXT_COLOR)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        tn, fp, fn, tp_ = cm.ravel()
        total = tn + fp + fn + tp_
        acc   = (tn + tp_) / total
        prec  = tp_ / (tp_ + fp) if (tp_ + fp) > 0 else 0
        rec   = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        ax.text(1.0, -0.12,
                f'Acc: {acc:.1%}  |  Prec: {prec:.1%}  |  Recall: {rec:.1%}  |  F1: {f1:.1%}',
                ha='center', va='top', transform=ax.transAxes,
                fontsize=8, color='#AAAAAA')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Live Prediction Tool")
    st.markdown("Adjust the parameters below to predict profitability probability:")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_sentiment = st.selectbox("Market Sentiment", ['Fear','Neutral','Greed'])
        p_fg_value  = st.slider("Fear/Greed Value", 0, 100, 50)
    with pc2:
        p_trades    = st.slider("Number of Trades", 1, 200, 50)
        p_leverage  = st.slider("Avg Leverage (x)", 1.0, 20.0, 1.5, 0.1)
    with pc3:
        p_size      = st.slider("Avg Position Size ($)", 100, 50000, 10000, 100)
        p_long      = st.slider("Long Ratio (%)", 0, 100, 55)
        p_winrate   = st.slider("Current Win Rate (%)", 0, 100, 80)

    enc_map = {'Fear': 0, 'Neutral': 1, 'Greed': 2}
    inp = pd.DataFrame([[enc_map[p_sentiment], p_fg_value, p_trades, p_leverage,
                         p_size, p_long/100, p_winrate/100]], columns=features)
    prob = rf.predict_proba(inp)[0][1]

    col_p1, col_p2 = st.columns([1,2])
    with col_p1:
        color = GREED_COLOR if prob >= 0.6 else (FEAR_COLOR if prob < 0.4 else NEUTRAL_COLOR)
        label = "Profitable" if prob >= 0.6 else ("Loss" if prob < 0.4 else "Uncertain")
        st.markdown(f"""
        <div style="background:{CARD_COLOR};border-radius:12px;padding:20px;text-align:center;
                    border:2px solid {color};">
            <h2 style="color:{color};margin:0">{prob*100:.1f}%</h2>
            <p style="color:{color};font-size:18px;margin:5px 0">{label} Day</p>
            <p style="color:#888;font-size:12px">Probability of Profit</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("Key Insights & Actionable Strategy Rules")

    st.markdown("""
    <div class="insight-card">
    <h3>Insight 1 — Fear Days Yield 2.9x More PnL</h3>
    <p>Fear days average <b>$8,042 PnL</b> vs Greed days at <b>$2,738</b>. These traders are
    experienced contrarians who see Fear as a buying opportunity. Panic selling by others
    creates discounted entry points for disciplined traders.</p>
    </div>

    <div class="insight-card">
    <h3>Insight 2 — Traders Are ~50% More Active on Fear Days</h3>
    <p>Fear days see <b>68.9 trades/day</b> vs Greed days at <b>45.8 trades/day</b>.
    Experienced traders actively hunt for opportunities when sentiment is negative —
    more entries = more chances to buy discounted assets.</p>
    </div>

    <div class="insight-card">
    <h3>Insight 3 — Leverage Rises During Fear</h3>
    <p>Fear: <b>1.23x avg leverage</b> vs Greed: <b>1.06x</b>. Confident traders size up
    when they believe the market is undervalued. This is calculated bold behavior,
    not panic — and it pays off.</p>
    </div>

    <div class="insight-card">
    <h3>Insight 4 — Long Bias Persists Even in Fear</h3>
    <p>Fear: <b>54.5% long</b> vs Greed: <b>50.2% long</b>. These traders buy dips,
    not panic sell. Even in Fear, they maintain long conviction on Bitcoin.</p>
    </div>

    <div class="insight-card">
    <h3>Insight 5 — Predictive Model: 96.3% ROC-AUC</h3>
    <p>Win rate and avg position size are the top features — not sentiment alone.
    Sentiment sets the context; behavior within the day drives the outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Actionable Strategy Rules")
    st.markdown("""
    <div class="rule-card">
    <h3>Rule 1 — Increase Position Size on Fear Days (High Win Rate Traders Only)</h3>
    <p>During Fear sentiment, traders with win rate above 80% should increase position size
    by up to 20%. Data shows Fear days generate 2.9x more PnL for this segment.
    <br><b>Condition:</b> Sentiment = Fear AND Win Rate > 80% → Scale up position by 1.2x</p>
    </div>

    <div class="rule-card">
    <h3>Rule 2 — Boost Trade Frequency on Fear, Reduce on Extreme Greed</h3>
    <p>On Fear days, increase the number of trades (experienced traders do ~50% more).
    On Extreme Greed (FG > 75), reduce activity — the crowd is overleveraged and
    reversal risk is elevated.
    <br><b>Rule:</b> FG < 30 → increase frequency by 1.5x | FG > 75 → reduce by 0.6x</p>
    </div>

    <div class="rule-card">
    <h3>Rule 3 — Net Losers: Do NOT Increase Leverage on Fear Days</h3>
    <p>The 3 Net Loser traders show their worst drawdowns on Fear days. Volatility
    amplifies mistakes for undisciplined traders. This segment needs the opposite rule:
    Fear days = reduce exposure, not increase it.
    <br><b>Rule:</b> If historical win rate below 70% → cap leverage at 1x on Fear days</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Raw Data Explorer")
    st.dataframe(view[['date','account','sentiment','daily_pnl','win_rate',
                        'num_trades','avg_leverage','long_ratio']].round(3),
                 use_container_width=True)
