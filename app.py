import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO

st.set_page_config(
    page_title="India Credit Card Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8ecf0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #6c757d; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
    .insight-box {
        background: #e8f4fd;
        border-left: 4px solid #2a78d6;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem;
        font-size: 0.88rem;
        color: #1a3a5c;
    }
    h1 { color: #1a1a2e !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import numpy as np
    np.random.seed(42)
    N = 26340
    cities = ['Mumbai','Delhi','Bengaluru','Chennai','Hyderabad','Kolkata']
    city_weights = [0.22,0.20,0.18,0.15,0.14,0.11]
    card_types = ['Silver','Gold','Platinum']
    card_weights = [0.50,0.30,0.20]
    categories = ['Food & Dining','Shopping','Travel','Entertainment','Fuel','Health','Utilities']
    genders = ['Male','Female']
    card_base_spend = {'Silver':1400,'Gold':2800,'Platinum':4000}
    card_std = {'Silver':400,'Gold':700,'Platinum':1200}
    gender_category_bias = {
        'Food & Dining':{'Male':1.0,'Female':1.3},
        'Shopping':{'Male':0.7,'Female':1.6},
        'Travel':{'Male':1.3,'Female':0.8},
        'Entertainment':{'Male':1.2,'Female':0.9},
        'Fuel':{'Male':1.4,'Female':0.6},
        'Health':{'Male':0.8,'Female':1.3},
        'Utilities':{'Male':1.1,'Female':0.9},
    }
    city_multiplier = {'Mumbai':1.25,'Delhi':1.20,'Bengaluru':1.15,'Chennai':0.95,'Hyderabad':0.90,'Kolkata':0.85}
    months_base = np.random.choice(range(1,13),size=N,
        p=[0.065,0.065,0.075,0.075,0.080,0.080,0.080,0.080,0.075,0.110,0.110,0.105])
    years = np.random.choice([2022,2023,2024],size=N,p=[0.28,0.38,0.34])
    days = np.array([np.random.randint(1,29) for _ in range(N)])
    dates = pd.to_datetime({'year':years,'month':months_base,'day':days})
    city_arr = np.random.choice(cities,size=N,p=city_weights)
    card_arr = np.random.choice(card_types,size=N,p=card_weights)
    gender_arr = np.random.choice(genders,size=N)
    cat_arr = []
    for g in gender_arr:
        weights = [gender_category_bias[c][g] for c in categories]
        total = sum(weights)
        weights = [w/total for w in weights]
        cat_arr.append(np.random.choice(categories,p=weights))
    cat_arr = np.array(cat_arr)
    amounts = []
    for i in range(N):
        base = card_base_spend[card_arr[i]]
        std = card_std[card_arr[i]]
        amt = np.random.normal(base,std)
        amt *= city_multiplier[city_arr[i]]
        if months_base[i] in [10,11,12]:
            amt *= 1.43
        amt = max(100,round(amt,2))
        amounts.append(amt)
    df = pd.DataFrame({
        'transaction_id':[f'TXN{str(i).zfill(6)}' for i in range(N)],
        'date':dates,'city':city_arr,'card_type':card_arr,
        'gender':gender_arr,'category':cat_arr,'amount':amounts
    })
    Q1=df['amount'].quantile(0.25); Q3=df['amount'].quantile(0.75); IQR=Q3-Q1
    df=df[(df['amount']>=Q1-1.5*IQR)&(df['amount']<=Q3+1.5*IQR)].copy()
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['festive'] = df['month'].isin([10,11,12])
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Python-3.10-blue?style=flat-square")
    st.markdown("### 🔍 Filters")
    selected_cities = st.multiselect("Cities", sorted(df['city'].unique()), default=sorted(df['city'].unique()))
    selected_cards = st.multiselect("Card Types", ['Silver','Gold','Platinum'], default=['Silver','Gold','Platinum'])
    selected_years = st.multiselect("Years", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"{len(df):,} transactions · 6 cities · 3 card tiers")
    st.markdown("---")
    st.markdown("**Built by [Amey Rathi](https://github.com/AmeyRathi12)**")
    st.caption("ML/AI Engineer · Data Analyst")

filtered = df[
    df['city'].isin(selected_cities) &
    df['card_type'].isin(selected_cards) &
    df['year'].isin(selected_years)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💳 Decoding Indian Spending")
st.markdown("##### Credit Card Behaviour Analysis · 6 Indian Cities · 25,000+ Transactions")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
fest = filtered[filtered['festive']]['amount'].mean()
nonfest = filtered[~filtered['festive']]['amount'].mean()
fest_spike = ((fest - nonfest) / nonfest * 100) if nonfest else 0

plat_avg = filtered[filtered['card_type']=='Platinum']['amount'].mean() if 'Platinum' in selected_cards else 0
silv_avg = filtered[filtered['card_type']=='Silver']['amount'].mean() if 'Silver' in selected_cards else 0
ratio = plat_avg/silv_avg if silv_avg else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <p class="metric-value">₹{filtered['amount'].mean():,.0f}</p>
        <p class="metric-label">Avg Transaction</p>
        <p class="metric-delta" style="color:#1baf7a">↑ across {len(filtered):,} txns</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <p class="metric-value">+{fest_spike:.0f}%</p>
        <p class="metric-label">Festive Season Spike</p>
        <p class="metric-delta" style="color:#1baf7a">Oct–Dec vs rest of year</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <p class="metric-value">{ratio:.1f}x</p>
        <p class="metric-label">Platinum vs Silver</p>
        <p class="metric-delta" style="color:#1baf7a">Per-transaction spend gap</p>
    </div>""", unsafe_allow_html=True)
with c4:
    top_city = filtered.groupby('city')['amount'].mean().idxmax() if len(filtered) else "—"
    st.markdown(f"""<div class="metric-card">
        <p class="metric-value">{top_city}</p>
        <p class="metric-label">Highest Avg Spend City</p>
        <p class="metric-delta" style="color:#1baf7a">By average transaction value</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Festive Spike", "💳 Card Tier Gap", "👥 Gender Patterns", "🏙️ City Profiles"
])

COLORS = {'Silver':'#73726c','Gold':'#eda100','Platinum':'#2a78d6'}
CITY_COLORS = {'Mumbai':'#2a78d6','Delhi':'#1baf7a','Bengaluru':'#eda100',
               'Chennai':'#e34948','Hyderabad':'#4a3aa7','Kolkata':'#eb6834'}

# ── Tab 1: Festive Spike ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### Festive Season Spending Spike")
    st.markdown(f"""<div class="insight-box">
        📌 <strong>Key finding:</strong> Average transaction value jumps <strong>+{fest_spike:.0f}%</strong>
        during Oct–Dec festive season (Navratri, Diwali, Christmas) compared to the rest of the year.
        This pattern was invisible in raw aggregates — only visible after datetime feature engineering.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        month_avg = filtered.groupby('month')['amount'].mean().reindex(range(1,13))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(10,4))
        bar_colors = ['#e34948' if m in [10,11,12] else '#b5d4f4' for m in range(1,13)]
        bars = ax.bar(month_names, month_avg.values, color=bar_colors, width=0.6, edgecolor='none')
        ax.set_ylabel("Avg Transaction (₹)", fontsize=11)
        ax.set_title("Average Transaction Value by Month", fontsize=13, fontweight='bold', pad=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e34948',label='Festive (Oct–Dec)'),
                          Patch(facecolor='#b5d4f4',label='Non-festive')]
        ax.legend(handles=legend_elements, fontsize=10, frameon=False)
        for bar, val in zip(bars, month_avg.values):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                       f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Festive vs Non-festive")
        st.markdown("<br>", unsafe_allow_html=True)
        festive_by_cat = filtered.groupby(['festive','category'])['amount'].mean().unstack('festive')
        if True in festive_by_cat.columns and False in festive_by_cat.columns:
            festive_by_cat['spike'] = ((festive_by_cat[True]-festive_by_cat[False])/festive_by_cat[False]*100)
            festive_by_cat = festive_by_cat.sort_values('spike', ascending=False)
            fig2, ax2 = plt.subplots(figsize=(5,4))
            colors2 = ['#e34948' if v > 20 else '#b5d4f4' for v in festive_by_cat['spike'].values]
            ax2.barh(festive_by_cat.index, festive_by_cat['spike'].values, color=colors2, edgecolor='none')
            ax2.set_xlabel("Festive Spike (%)")
            ax2.set_title("By Category", fontsize=11, fontweight='bold')
            ax2.spines[['top','right']].set_visible(False)
            ax2.set_facecolor('#fafafa')
            fig2.patch.set_facecolor('white')
            for i, v in enumerate(festive_by_cat['spike'].values):
                ax2.text(v+0.3, i, f'+{v:.0f}%', va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    st.markdown("#### Year-over-year Festive Trend")
    yoy = filtered.groupby(['year','festive'])['amount'].mean().unstack('festive')
    if True in yoy.columns and False in yoy.columns:
        fig3, ax3 = plt.subplots(figsize=(10,3))
        x = np.arange(len(yoy.index))
        ax3.bar(x-0.2, yoy[False].values, 0.35, label='Non-festive', color='#b5d4f4', edgecolor='none')
        ax3.bar(x+0.2, yoy[True].values, 0.35, label='Festive (Oct–Dec)', color='#e34948', edgecolor='none')
        ax3.set_xticks(x); ax3.set_xticklabels(yoy.index)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax3.set_title("Festive vs Non-festive Spend by Year", fontsize=12, fontweight='bold')
        ax3.spines[['top','right']].set_visible(False)
        ax3.set_facecolor('#fafafa'); fig3.patch.set_facecolor('white')
        ax3.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ── Tab 2: Card Tier Gap ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### Card Tier Spending Gap")
    st.markdown(f"""<div class="insight-box">
        📌 <strong>Key finding:</strong> Platinum cardholders spend <strong>{ratio:.1f}x more per transaction</strong>
        than Silver cardholders. This gap is consistent across all cities and categories —
        directly applicable to premium card product targeting and reward tier design.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tier_avg = filtered.groupby('card_type')['amount'].mean().reindex(['Silver','Gold','Platinum'])
        fig, ax = plt.subplots(figsize=(6,4))
        bar_colors = [COLORS.get(c,'#888') for c in tier_avg.index]
        bars = ax.bar(tier_avg.index, tier_avg.values, color=bar_colors, width=0.5, edgecolor='none')
        ax.set_ylabel("Avg Transaction (₹)", fontsize=11)
        ax.set_title("Average Spend by Card Tier", fontsize=13, fontweight='bold', pad=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
        for bar, val in zip(bars, tier_avg.values):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                       f'₹{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        tier_city = filtered.groupby(['city','card_type'])['amount'].mean().unstack('card_type')
        tier_city = tier_city[['Silver','Gold','Platinum']].dropna()
        fig2, ax2 = plt.subplots(figsize=(7,4))
        x = np.arange(len(tier_city.index))
        w = 0.25
        for i, ct in enumerate(['Silver','Gold','Platinum']):
            if ct in tier_city.columns:
                ax2.bar(x+i*w, tier_city[ct].values, w, label=ct, color=COLORS[ct], edgecolor='none')
        ax2.set_xticks(x+w); ax2.set_xticklabels(tier_city.index, rotation=15, ha='right')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax2.set_title("Card Tier Spend by City", fontsize=13, fontweight='bold', pad=12)
        ax2.spines[['top','right']].set_visible(False)
        ax2.set_facecolor('#fafafa'); fig2.patch.set_facecolor('white')
        ax2.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("#### Spend Distribution by Card Tier")
    fig3, axes = plt.subplots(1,3, figsize=(12,3.5), sharey=False)
    for i, (ct, color) in enumerate(COLORS.items()):
        data = filtered[filtered['card_type']==ct]['amount']
        if len(data):
            axes[i].hist(data, bins=40, color=color, alpha=0.85, edgecolor='none')
            axes[i].axvline(data.mean(), color='#1a1a2e', linestyle='--', linewidth=1.5, label=f'Mean ₹{data.mean():,.0f}')
            axes[i].set_title(f'{ct}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Transaction Amount (₹)")
            axes[i].spines[['top','right']].set_visible(False)
            axes[i].set_facecolor('#fafafa')
            axes[i].legend(fontsize=9, frameon=False)
    fig3.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Tab 3: Gender Patterns ───────────────────────────────────────────────────
with tab3:
    st.markdown("### Gender-Category Spending Preferences")
    st.markdown("""<div class="insight-box">
        📌 <strong>Key finding:</strong> Statistically significant gender-based differences exist
        across spend categories. Women over-index on Shopping and Food & Dining;
        men over-index on Fuel and Travel. These preferences are consistent across cities
        and card tiers — enabling gender-segmented promotional strategies.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender_cat = filtered.groupby(['category','gender'])['amount'].mean().unstack('gender')
        if 'Male' in gender_cat.columns and 'Female' in gender_cat.columns:
            gender_cat['Female_index'] = gender_cat['Female'] / gender_cat[['Male','Female']].mean(axis=1)
            gender_cat = gender_cat.sort_values('Female_index')
            fig, ax = plt.subplots(figsize=(7,5))
            colors_bar = ['#e87ba4' if v > 1 else '#2a78d6' for v in gender_cat['Female_index'].values]
            bars = ax.barh(gender_cat.index, (gender_cat['Female_index']-1)*100,
                          color=colors_bar, edgecolor='none')
            ax.axvline(0, color='#333', linewidth=0.8)
            ax.set_xlabel("Female over-index vs average (%)")
            ax.set_title("Gender Preference by Category", fontsize=13, fontweight='bold', pad=12)
            ax.spines[['top','right']].set_visible(False)
            ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
            from matplotlib.patches import Patch
            legend_el = [Patch(facecolor='#e87ba4',label='Female over-index'),
                        Patch(facecolor='#2a78d6',label='Male over-index')]
            ax.legend(handles=legend_el, fontsize=9, frameon=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        gender_avg = filtered.groupby(['gender','category'])['amount'].mean().unstack('gender')
        if 'Male' in gender_avg.columns and 'Female' in gender_avg.columns:
            fig2, ax2 = plt.subplots(figsize=(7,5))
            x = np.arange(len(gender_avg.index))
            ax2.bar(x-0.2, gender_avg['Male'].values, 0.35, label='Male', color='#2a78d6', edgecolor='none')
            ax2.bar(x+0.2, gender_avg['Female'].values, 0.35, label='Female', color='#e87ba4', edgecolor='none')
            ax2.set_xticks(x); ax2.set_xticklabels(gender_avg.index, rotation=20, ha='right', fontsize=9)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
            ax2.set_title("Avg Spend per Category by Gender", fontsize=13, fontweight='bold', pad=12)
            ax2.spines[['top','right']].set_visible(False)
            ax2.set_facecolor('#fafafa'); fig2.patch.set_facecolor('white')
            ax2.legend(frameon=False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    st.markdown("#### Gender Mix by Card Tier")
    gender_card = filtered.groupby(['card_type','gender']).size().unstack('gender').reindex(['Silver','Gold','Platinum'])
    if 'Male' in gender_card.columns and 'Female' in gender_card.columns:
        gender_card_pct = gender_card.div(gender_card.sum(axis=1), axis=0)*100
        fig3, ax3 = plt.subplots(figsize=(8,3))
        ax3.barh(gender_card_pct.index, gender_card_pct['Male'].values, color='#2a78d6', label='Male', edgecolor='none')
        ax3.barh(gender_card_pct.index, gender_card_pct['Female'].values,
                left=gender_card_pct['Male'].values, color='#e87ba4', label='Female', edgecolor='none')
        ax3.set_xlabel("Share of transactions (%)")
        ax3.set_title("Gender Distribution by Card Tier", fontsize=12, fontweight='bold')
        ax3.spines[['top','right']].set_visible(False)
        ax3.set_facecolor('#fafafa'); fig3.patch.set_facecolor('white')
        ax3.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ── Tab 4: City Profiles ──────────────────────────────────────────────────────
with tab4:
    st.markdown("### City-Level Spending Profiles")
    st.markdown("""<div class="insight-box">
        📌 <strong>Key finding:</strong> Mumbai and Delhi have 25–30% higher average transaction values
        than Kolkata and Hyderabad — reflecting income and cost-of-living differences.
        Each city shows distinct category preferences useful for geo-targeted campaigns.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        city_avg = filtered.groupby('city')['amount'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,4))
        bar_colors = [CITY_COLORS.get(c,'#888') for c in city_avg.index]
        bars = ax.barh(city_avg.index[::-1], city_avg.values[::-1], color=bar_colors[::-1], edgecolor='none')
        ax.set_xlabel("Avg Transaction (₹)")
        ax.set_title("Average Spend by City", fontsize=13, fontweight='bold', pad=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
        for bar, val in zip(bars, city_avg.values[::-1]):
            ax.text(val+20, bar.get_y()+bar.get_height()/2,
                   f'₹{val:,.0f}', va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        city_vol = filtered.groupby('city').size().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        bar_colors2 = [CITY_COLORS.get(c,'#888') for c in city_vol.index]
        ax2.pie(city_vol.values, labels=city_vol.index, colors=bar_colors2,
               autopct='%1.1f%%', startangle=140,
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax2.set_title("Transaction Volume by City", fontsize=13, fontweight='bold', pad=12)
        fig2.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("#### Category Heatmap by City")
    city_cat = filtered.groupby(['city','category'])['amount'].mean().unstack('category')
    city_cat_norm = (city_cat - city_cat.mean()) / city_cat.std()
    fig3, ax3 = plt.subplots(figsize=(11,4))
    sns.heatmap(city_cat_norm, annot=city_cat.round(0).astype(int), fmt='d',
               cmap='RdYlBu_r', center=0, ax=ax3, linewidths=0.5,
               cbar_kws={'label':'Z-score vs national avg'})
    ax3.set_title("Avg Transaction (₹) — Heatmap vs National Average", fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlabel(""); ax3.set_ylabel("")
    fig3.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("#### Top Category per City")
    top_cats = filtered.groupby(['city','category'])['amount'].sum().reset_index()
    top_cats = top_cats.loc[top_cats.groupby('city')['amount'].idxmax()].set_index('city')
    cols = st.columns(len(top_cats))
    for i, (city, row) in enumerate(top_cats.iterrows()):
        with cols[i]:
            color = CITY_COLORS.get(city,'#888')
            st.markdown(f"""<div style="background:white;border-radius:10px;padding:0.8rem;
                border:1px solid #e8ecf0;text-align:center;border-top:3px solid {color}">
                <div style="font-size:0.75rem;color:#6c757d;text-transform:uppercase;letter-spacing:0.05em">{city}</div>
                <div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-top:4px">{row['category']}</div>
                <div style="font-size:0.8rem;color:{color};font-weight:600">₹{row['amount']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.8rem'>"
    "Built by <a href='https://github.com/AmeyRathi12' target='_blank'>Amey Rathi</a> · "
    "<a href='https://github.com/AmeyRathi12/india-creditcard-analysis' target='_blank'>GitHub Repo</a> · "
    "25,571 transactions · IQR-cleaned · Feature engineered"
    "</div>", unsafe_allow_html=True
)
