"""
Cohort Quiz Analysis — окрема сторінка для глибокого когортного аналізу
воронок mm-sq1-v1 та mf-sq1-v1.

Що тут є
--------
1. Drop-off per screen (скільки юзерів дійшли до кроку)
2. Розподіл відповідей на конкретному екрані (з опціональним split multi-select)
3. Cohort builder — фільтр по одній або декількох (screen, answer) парах
   → повна funnel-таблиця для когорти vs решта (з стат значущістю)
4. Cross-tab двох екранів: Sankey + Stacked bar
5. Multi-step сегментація (back-back-back filters)

Очікуваний CSV (long format) — формується запитом з bq_extract_guide.md секція 1.2 / 2.2:
    user_id, landingId, flowId, screen_id, screen_order, question_text, answer_value, event_at

Опційно: land_1.csv (той самий що в Home page) для повних бізнес-метрик.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cohort Quiz Analysis",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Cohort Quiz Analysis")
st.caption(
    "Глибокий когортний аналіз воронок **mm-sq1-v1** та **mf-sq1-v1**. "
    "Завантаж long-format quiz CSV (з bq_extract_guide.md, секції 1.2 / 2.2). "
    "Опційно — land_1.csv для розрахунку конверсій / ARPU по когортах."
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_LANDINGS = ['mm-sq1-v1', 'mf-sq1-v1']
LAND1_DATE_COLS = ['created_at', 'profile_created_at', 'reg_at', 'fo_at',
                   'landing_at', 'order_created_at']

# Хардкодимо порядок екранів з screen_catalog (на випадок якщо в long немає screen_order)
SCREEN_ORDER_MM = [
    'gender', 'last-date', 'get-start', 'target', 'marital-status',
    'interests', 'infographic', 'body', 'partner-age', 'birthdate',
    'attracts', 'name', 'table', 'value', 'taboo', 'ideal-date',
    'invest-time-dt', 'distance', 'loader', 'graph', 'email',
    'password', 'congrats', 'profile-photo', 'location', 'about', 'done'
]
SCREEN_ORDER_MF = [
    'decisions', 'lack', 'get-start', 'target', 'marital-status',
    'interests', 'infographic', 'body', 'partner-height', 'age-range',
    'birthdate', 'habits', 'pets', 'name', 'table', 'important-things',
    'relationship', 'ideal-date', 'loader', 'graph', 'email', 'password',
    'congrats', 'profile-photo', 'location', 'about', 'done'
]
DEFAULT_SCREEN_ORDER = {
    'mm-sq1-v1': SCREEN_ORDER_MM,
    'mf-sq1-v1': SCREEN_ORDER_MF,
}

# Які екрани вважаємо "interstitial" (не питання — там одна відповідь Continue)
# Корисно для дефолту — щоб у dropdown питань не показувати їх
INTERSTITIAL_SCREENS = {
    'gender', 'get-start', 'infographic', 'table', 'graph',
    'congrats', 'profile-photo', 'done'
}

# Multi-select екрани (csv_list / csv_list_mixed) — для них корисно сплітити по комі
MULTI_SELECT_SCREENS = {
    'mm-sq1-v1': {
        'target', 'interests', 'body', 'attracts', 'value',
        'taboo', 'location', 'about'
    },
    'mf-sq1-v1': {
        'target', 'interests', 'important-things', 'ideal-date',
        'location', 'about'
    },
}

# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_quiz_long(file_or_buf) -> pd.DataFrame:
    """Long quiz CSV: 1 row = user × screen × answer."""
    df = pd.read_csv(file_or_buf)
    required = {'user_id', 'landingId', 'screen_id', 'answer_value'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Приводимо типи
    df['user_id']      = df['user_id'].astype(str)
    df['landingId']    = df['landingId'].astype(str)
    df['screen_id']    = df['screen_id'].astype(str)
    df['answer_value'] = df['answer_value'].fillna('').astype(str).str.strip()

    if 'screen_order' in df.columns:
        df['screen_order'] = pd.to_numeric(df['screen_order'], errors='coerce')

    if 'event_at' in df.columns:
        df['event_at'] = pd.to_datetime(df['event_at'], errors='coerce')

    # Залишаємо тільки воронки які підтримуємо
    df = df[df['landingId'].isin(SUPPORTED_LANDINGS)].copy()

    # Defensive dedup: одна остання відповідь юзера на екран
    sort_col = 'event_at' if 'event_at' in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col)
    df = df.drop_duplicates(subset=['user_id', 'landingId', 'screen_id'], keep='last')

    return df


@st.cache_data(show_spinner=False)
def load_land1(file_or_buf) -> pd.DataFrame:
    """land_1.csv — той самий формат що в основній сторінці."""
    df = pd.read_csv(file_or_buf)
    for col in LAND1_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = df['amount'] / 100.0
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────
# Metric helpers (узгоджені з Home page)
# ─────────────────────────────────────────────────────────────────────
def calculate_metrics(df: pd.DataFrame) -> dict:
    """Повторює логіку Home page: visitors, reg, payers, ARPU 0d тощо."""
    if df is None or df.empty:
        return {
            'Visitors': 0, 'Onboarding Users': 0, 'Registered Users': 0,
            'Payers': 0, 'Payers (Day 0)': 0, 'Payers 0d (Landing)': 0,
            'Total Revenue': 0.0, 'Revenue 0d (Landing)': 0.0,
            'ARPU': 0.0, 'ARPPU': 0.0, 'ARPU 0d': 0.0, 'ARPPU 0d': 0.0,
        }

    visitors    = df['user_id'].nunique()
    onboarding  = df[df['profile_created_at'].notnull()]['user_id'].nunique() if 'profile_created_at' in df.columns else 0
    registered  = df[df['reg_at'].notnull()]['user_id'].nunique()              if 'reg_at' in df.columns              else 0
    payers      = df[df['fo_at'].notnull()]['user_id'].nunique()                if 'fo_at' in df.columns                else 0

    payers_d0 = 0
    if 'fo_at' in df.columns and 'reg_at' in df.columns:
        sub = df.dropna(subset=['fo_at', 'reg_at'])
        payers_d0 = sub[(sub['fo_at'] - sub['reg_at']) <= pd.Timedelta(hours=24)]['user_id'].nunique()

    payers_0d_land = 0
    if 'fo_at' in df.columns and 'landing_at' in df.columns:
        sub = df.dropna(subset=['fo_at', 'landing_at'])
        payers_0d_land = sub[(sub['fo_at'] - sub['landing_at']) <= pd.Timedelta(hours=24)]['user_id'].nunique()

    total_amount = df['amount'].sum() if 'amount' in df.columns else 0.0

    rev_0d_land = 0.0
    if {'amount', 'order_created_at', 'landing_at'}.issubset(df.columns):
        sub = df.dropna(subset=['order_created_at', 'landing_at', 'amount'])
        rev_0d_land = sub[(sub['order_created_at'] - sub['landing_at']) <= pd.Timedelta(hours=24)]['amount'].sum()

    arpu      = total_amount / visitors    if visitors    > 0 else 0
    arppu     = total_amount / payers      if payers      > 0 else 0
    arpu_0d   = rev_0d_land / registered   if registered  > 0 else 0
    arppu_0d  = rev_0d_land / payers_0d_land if payers_0d_land > 0 else 0

    return {
        'Visitors': visitors,
        'Onboarding Users': onboarding,
        'Registered Users': registered,
        'Payers': payers,
        'Payers (Day 0)': payers_d0,
        'Payers 0d (Landing)': payers_0d_land,
        'Total Revenue': float(total_amount),
        'Revenue 0d (Landing)': float(rev_0d_land),
        'ARPU': arpu, 'ARPPU': arppu,
        'ARPU 0d': arpu_0d, 'ARPPU 0d': arppu_0d,
    }


def conversion_rates(m: dict) -> dict:
    v = m['Visitors']; r = m['Registered Users']
    return {
        'Landing → Onboarding':   100 * m['Onboarding Users']    / v if v else 0,
        'Landing → Registration': 100 * r                         / v if v else 0,
        'Registration → Payer':   100 * m['Payers']               / r if r else 0,
        'Reg → Payer 0d':         100 * m['Payers 0d (Landing)'] / r if r else 0,
        'Landing → Payer (24h)':  100 * m['Payers 0d (Landing)'] / v if v else 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Statistical helpers (Bayesian + Frequentist)
# ─────────────────────────────────────────────────────────────────────
def bayes_proportion(succ_c, n_c, succ_v, n_v, n_samples=10000):
    if n_c == 0 or n_v == 0: return 0.5, 0.0
    succ_c, succ_v = min(succ_c, n_c), min(succ_v, n_v)
    pc = np.random.beta(1 + succ_c, 1 + (n_c - succ_c), n_samples)
    pv = np.random.beta(1 + succ_v, 1 + (n_v - succ_v), n_samples)
    prob = float(np.mean(pv > pc))
    mc   = float(np.mean(pc))
    uplift = (np.mean(pv) - mc) / mc * 100 if mc > 0 else 0.0
    return prob, uplift


def freq_proportion(succ_c, n_c, succ_v, n_v):
    if n_c == 0 or n_v == 0: return 1.0, 0.0
    try:
        _, pval = proportions_ztest([succ_v, succ_c], [n_v, n_c], alternative='two-sided')
        rc = succ_c / n_c; rv = succ_v / n_v
        uplift = (rv - rc) / rc * 100 if rc > 0 else 0.0
        return float(pval), float(uplift)
    except Exception:
        return 1.0, 0.0


def verdict(stat_val, uplift, method, n_c, n_v):
    if n_c < 100 or n_v < 100: return "⚠️ Insufficient data"
    if method.startswith('Bayes'):
        if stat_val > 0.95: return "🚀 Winner (>95%)"
        if stat_val < 0.05: return "📉 Loser (<5%)"
        return "⚖️ Inconclusive"
    else:
        if stat_val < 0.05:
            return "✅ Winner (sig)" if uplift > 0 else "❌ Loser (sig)"
        return "🤷 No diff"


# ─────────────────────────────────────────────────────────────────────
# Multi-select utilities
# ─────────────────────────────────────────────────────────────────────
def explode_multi(df_screen: pd.DataFrame, separator: str = ',') -> pd.DataFrame:
    """Розбиває answer_value 'a, b, c' на 3 рядки. Корисно для csv_list екранів."""
    out = df_screen.copy()
    out['answer_value'] = (
        out['answer_value']
        .fillna('')
        .astype(str)
        .str.split(separator)
    )
    out = out.explode('answer_value')
    out['answer_value'] = out['answer_value'].astype(str).str.strip()
    out = out[out['answer_value'] != '']
    return out


def is_multi_select(landing_id: str, screen_id: str) -> bool:
    return screen_id in MULTI_SELECT_SCREENS.get(landing_id, set())


# ─────────────────────────────────────────────────────────────────────
# Sidebar — data sources
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("📥 Data Sources")

st.sidebar.markdown("**Quiz events (long format)**")
quiz_file = st.sidebar.file_uploader(
    "Upload quiz_long.csv",
    type=['csv'],
    key='quiz_csv',
    help="Колонки: user_id, landingId, screen_id, answer_value, [screen_order], [event_at]"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**land_1.csv (опційно)**")
land1_file = st.sidebar.file_uploader(
    "Upload land_1.csv",
    type=['csv'],
    key='land1_csv',
    help="Той самий формат що на головній сторінці. Без нього — тільки drop-off і cross-tab без бізнес-метрик."
)

# Завантажуємо
df_quiz = None
df_land = None

if quiz_file is not None:
    try:
        df_quiz = load_quiz_long(quiz_file)
        st.sidebar.success(f"✅ Quiz: {len(df_quiz):,} rows / {df_quiz['user_id'].nunique():,} users")
    except Exception as e:
        st.sidebar.error(f"Quiz load error: {e}")

if land1_file is not None:
    try:
        df_land = load_land1(land1_file)
        st.sidebar.success(f"✅ land_1: {len(df_land):,} rows / {df_land['user_id'].nunique():,} users")
    except Exception as e:
        st.sidebar.error(f"land_1 load error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Early exit якщо нема quiz даних
# ─────────────────────────────────────────────────────────────────────
if df_quiz is None or df_quiz.empty:
    st.info(
        "👈 Завантаж long-format quiz CSV у сайдбарі.\n\n"
        "Очікувані колонки: `user_id`, `landingId`, `screen_id`, `answer_value`. "
        "Опційно: `screen_order`, `event_at`, `flowId`, `question_text`.\n\n"
        "SQL для extract'а — в `bq_extract_guide.md`, секції 1.2 (mm) та 2.2 (mf)."
    )
    with st.expander("ℹ️ Швидкий приклад SQL для extract"):
        st.code("""
WITH raw_quiz AS (
  SELECT
    user_id,
    created_at AS event_at,
    (SELECT value FROM UNNEST(params) WHERE key='landingId')   AS landingId,
    (SELECT value FROM UNNEST(params) WHERE key='pageName')    AS screen_id,
    SAFE_CAST((SELECT value FROM UNNEST(params) WHERE key='pageNumber') AS INT64) AS screen_order,
    (SELECT value FROM UNNEST(params) WHERE key='action')      AS answer_value
  FROM `gdx-prod-3caeb166.rawdata.analytics_events`
  WHERE DATE(created_at) BETWEEN '2026-04-01' AND '2026-04-27'
    AND name = 'user_action'
    AND (SELECT value FROM UNNEST(params) WHERE key='landingId') IN ('mm-sq1-v1','mf-sq1-v1')
    AND user_id IS NOT NULL AND user_id != ''
)
SELECT * FROM raw_quiz
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id, landingId, screen_id ORDER BY event_at DESC)=1
""", language='sql')
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Top-level controls — flow + global filters
# ─────────────────────────────────────────────────────────────────────
available_landings = [v for v in SUPPORTED_LANDINGS if v in df_quiz['landingId'].unique()]

ctop = st.columns([1, 2, 2])
with ctop[0]:
    flow = st.selectbox("Flow", available_landings, index=0)

screen_order = DEFAULT_SCREEN_ORDER[flow]
df_q = df_quiz[df_quiz['landingId'] == flow].copy()

# Додаємо/нормалізуємо screen_order на основі catalog якщо в long немає
if 'screen_order' not in df_q.columns or df_q['screen_order'].isna().all():
    order_map = {s: i + 1 for i, s in enumerate(screen_order)}
    df_q['screen_order'] = df_q['screen_id'].map(order_map)

# Фільтрація land_1 по тій самій воронці
df_l = None
if df_land is not None and 'landingId' in df_land.columns:
    df_l = df_land[df_land['landingId'] == flow].copy()

with ctop[1]:
    if df_l is not None and 'landing_at' in df_l.columns and df_l['landing_at'].notna().any():
        min_d = df_l['landing_at'].min().date()
        max_d = df_l['landing_at'].max().date()
        date_range = st.date_input(
            "Date range (за landing_at з land_1)",
            value=(min_d, max_d),
            min_value=min_d, max_value=max_d
        )
    else:
        date_range = None

with ctop[2]:
    apply_country_filter = False
    countries = []
    if df_l is not None and 'country' in df_l.columns:
        all_c = sorted(df_l['country'].dropna().unique())
        if all_c:
            sel_all = st.checkbox("Усі країни", value=True)
            if not sel_all:
                countries = st.multiselect("Countries", all_c)
                apply_country_filter = True

# Apply land_1 filters and shrink quiz cohort to those user_ids if land_1 is provided
if df_l is not None:
    if date_range is not None and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        sd, ed = date_range
        df_l = df_l[
            (df_l['landing_at'].dt.date >= sd) &
            (df_l['landing_at'].dt.date <= ed)
        ]
    if apply_country_filter and countries:
        df_l = df_l[df_l['country'].isin(countries)]

    keep_users = set(df_l['user_id'].unique())
    df_q = df_q[df_q['user_id'].isin(keep_users)]

st.caption(
    f"**{flow}** · quiz users: **{df_q['user_id'].nunique():,}** · quiz rows: **{len(df_q):,}**"
    + (f" · land_1 users: **{df_l['user_id'].nunique():,}**" if df_l is not None else "")
)

if df_q.empty:
    st.warning("Після фільтрів у quiz-данних не лишилось рядків.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sections (tabs)
# ─────────────────────────────────────────────────────────────────────
tab_dropoff, tab_dist, tab_cohort, tab_xtab = st.tabs([
    "📉 Drop-off",
    "📊 Answer distribution",
    "🎯 Cohort builder",
    "🔀 Cross-tab (A → B)"
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — Drop-off
# ═════════════════════════════════════════════════════════════════════
with tab_dropoff:
    st.subheader("Drop-off per screen")
    st.caption("Унікальні юзери що дійшли до кожного екрана. Базою (100%) є перший екран воронки.")

    drop = (df_q.groupby(['screen_order', 'screen_id'])['user_id']
                .nunique().reset_index(name='users_reached'))
    # Сортуємо по catalog порядку
    order_map = {s: i + 1 for i, s in enumerate(screen_order)}
    drop['screen_order'] = drop['screen_id'].map(order_map).fillna(drop['screen_order'])
    drop = drop.sort_values('screen_order')

    if not drop.empty:
        base = drop.iloc[0]['users_reached']
        drop['pct_of_start'] = drop['users_reached'] / base * 100
        drop['step_drop_pct'] = drop['users_reached'].pct_change().fillna(0) * 100  # відносно попереднього
        drop['label'] = drop.apply(
            lambda r: f"{int(r['users_reached']):,}<br>{r['pct_of_start']:.1f}%",
            axis=1
        )

        fig = px.bar(
            drop, x='screen_id', y='users_reached',
            text='label',
            title=f"Drop-off · {flow}",
            category_orders={'screen_id': screen_order},
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(
            xaxis_tickangle=-40,
            yaxis_title="Unique users reached",
            margin=dict(t=50, b=120),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Окрема діаграма step-drop
        st.markdown("##### Step-by-step drop (% від попереднього кроку)")
        drop['step_drop_label'] = drop['step_drop_pct'].apply(lambda v: f"{v:+.1f}%")
        fig2 = px.bar(
            drop, x='screen_id', y='step_drop_pct',
            text='step_drop_label',
            color='step_drop_pct',
            color_continuous_scale='RdYlGn',
            category_orders={'screen_id': screen_order},
            title="Step-over-step relative change",
        )
        fig2.update_traces(textposition='outside', cliponaxis=False)
        fig2.update_layout(xaxis_tickangle=-40, yaxis_ticksuffix='%',
                           coloraxis_showscale=False, height=400, margin=dict(t=50, b=120))
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Drop-off table"):
            show = drop.copy()
            show = show.rename(columns={
                'screen_order': '#',
                'pct_of_start': '% of start',
                'step_drop_pct': 'Δ vs prev (%)',
            })
            show['% of start']    = show['% of start'].map(lambda v: f"{v:.2f}%")
            show['Δ vs prev (%)'] = show['Δ vs prev (%)'].map(lambda v: f"{v:+.2f}%")
            show['users_reached'] = show['users_reached'].map(lambda v: f"{int(v):,}")
            st.dataframe(show[['#', 'screen_id', 'users_reached', '% of start', 'Δ vs prev (%)']],
                         hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — Answer distribution per screen
# ═════════════════════════════════════════════════════════════════════
with tab_dist:
    st.subheader("Розподіл відповідей на екрані")

    question_screens = [s for s in screen_order
                        if s in df_q['screen_id'].unique() and s not in INTERSTITIAL_SCREENS]
    interstitial_present = [s for s in screen_order
                            if s in df_q['screen_id'].unique() and s in INTERSTITIAL_SCREENS]
    show_interstitial = st.checkbox("Включити interstitial-екрани", value=False)
    options = question_screens + (interstitial_present if show_interstitial else [])

    if not options:
        st.info("Немає доступних екранів.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            screen_pick = st.selectbox("Екран", options, key='dist_screen')
        with c2:
            top_n = st.number_input("Top N answers", min_value=5, max_value=100, value=20, step=5)
        with c3:
            do_split = is_multi_select(flow, screen_pick)
            do_split = st.checkbox("Split multi-select", value=do_split,
                                   help="Розбити 'a, b, c' на 3 окремі відповіді")

        sub = df_q[df_q['screen_id'] == screen_pick]
        if do_split:
            sub = explode_multi(sub)

        total = sub['user_id'].nunique() if not do_split else len(sub)
        dist = (sub.groupby('answer_value')
                   .agg(users=('user_id', 'nunique'),
                        events=('user_id', 'count'))
                   .reset_index()
                   .sort_values('users', ascending=False)
                   .head(int(top_n)))
        dist['pct'] = dist['users'] / dist['users'].sum() * 100
        dist['label'] = dist.apply(lambda r: f"{int(r['users']):,} ({r['pct']:.1f}%)", axis=1)

        st.caption(f"Унікальні юзери на екрані: **{sub['user_id'].nunique():,}** · "
                   f"усього (зі сплітом=рядки): **{total:,}**")

        fig = px.bar(dist, x='users', y='answer_value', orientation='h',
                     text='label', title=f"{screen_pick} · top-{top_n} answers")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          height=max(360, 24 * len(dist) + 80),
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Distribution table"):
            show = dist.rename(columns={'answer_value': 'Answer', 'users': 'Users',
                                         'events': 'Events', 'pct': '%'})
            show['Users']  = show['Users'].map(lambda v: f"{int(v):,}")
            show['Events'] = show['Events'].map(lambda v: f"{int(v):,}")
            show['%']      = show['%'].map(lambda v: f"{v:.2f}%")
            st.dataframe(show[['Answer', 'Users', 'Events', '%']],
                         hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — Cohort builder
# ═════════════════════════════════════════════════════════════════════
with tab_cohort:
    st.subheader("Cohort builder · funnel метрики по сегменту відповіді")
    st.caption(
        "Збираєш фільтр (screen, answer) — система знаходить юзерів які так відповіли, "
        "і рахує всі funnel-метрики для цього сегмента vs решти юзерів воронки."
    )

    if df_l is None:
        st.warning(
            "⚠️ Без `land_1.csv` тут можна побачити тільки розмір когорти. "
            "Завантаж land_1.csv зліва щоб мати конверсії, ARPU 0d, payers тощо."
        )

    # ── Збір фільтрів
    question_screens = [s for s in screen_order
                        if s in df_q['screen_id'].unique() and s not in INTERSTITIAL_SCREENS]

    cf1, cf2, cf3 = st.columns([2, 3, 1])
    with cf1:
        n_filters = st.number_input("Кількість фільтрів (chain AND)", 1, 5, value=1)
    with cf3:
        treat_split = st.checkbox(
            "Match по split items",
            value=False,
            help="Якщо True — для multi-select екранів матч робиться по '{answer} є в csv list', "
                 "не по точному рядку."
        )

    filters = []
    for i in range(int(n_filters)):
        with st.container():
            cs1, cs2 = st.columns([1, 2])
            with cs1:
                sf = st.selectbox(
                    f"Screen #{i+1}", question_screens,
                    key=f'cohort_screen_{i}',
                    index=min(i, len(question_screens) - 1) if question_screens else 0
                )
            with cs2:
                sub_for_screen = df_q[df_q['screen_id'] == sf]
                if treat_split and is_multi_select(flow, sf):
                    sub_for_screen = explode_multi(sub_for_screen)
                answer_values = (
                    sub_for_screen['answer_value']
                    .value_counts()
                    .head(80)
                    .index.tolist()
                )
                af = st.multiselect(
                    f"Answers on {sf}", answer_values,
                    key=f'cohort_answer_{i}'
                )
            filters.append({'screen': sf, 'answers': af})

    # ── Computing cohort user_ids (intersection of filters)
    cohort_users = None
    for f in filters:
        if not f['answers']:
            continue
        sub = df_q[df_q['screen_id'] == f['screen']]
        if treat_split and is_multi_select(flow, f['screen']):
            sub_x = explode_multi(sub)
            uid = set(sub_x[sub_x['answer_value'].isin(f['answers'])]['user_id'].unique())
        else:
            uid = set(sub[sub['answer_value'].isin(f['answers'])]['user_id'].unique())
        cohort_users = uid if cohort_users is None else (cohort_users & uid)

    # ── Базова quiz population для воронки
    flow_users = set(df_q['user_id'].unique())

    if cohort_users is None or len(cohort_users) == 0:
        st.info("Збери хоча б один (screen, answer) фільтр щоб побачити когорту.")
    else:
        rest_users = flow_users - cohort_users

        st.markdown(f"##### Cohort size: **{len(cohort_users):,}** users · "
                    f"Rest: **{len(rest_users):,}** users · "
                    f"Total in flow: **{len(flow_users):,}**")

        # ── Якщо є land_1 — рахуємо всі funnel метрики
        if df_l is not None:
            method = st.radio("Stat method", ['Bayesian', 'Frequentist'], horizontal=True)

            df_cohort = df_l[df_l['user_id'].isin(cohort_users)]
            df_rest   = df_l[df_l['user_id'].isin(rest_users)]

            m_co  = calculate_metrics(df_cohort)
            m_re  = calculate_metrics(df_rest)
            c_co  = conversion_rates(m_co)
            c_re  = conversion_rates(m_re)

            metric_rows = []

            def add_prop_row(name, num_co, den_co, num_re, den_re):
                if method == 'Bayesian':
                    stat, lift = bayes_proportion(num_re, den_re, num_co, den_co)
                    stat_s = f"{stat:.1%}"
                else:
                    stat, lift = freq_proportion(num_re, den_re, num_co, den_co)
                    stat_s = f"p={stat:.4f}"
                vd = verdict(stat, lift, method, den_re, den_co)
                rate_co = (num_co / den_co * 100) if den_co else 0
                rate_re = (num_re / den_re * 100) if den_re else 0
                metric_rows.append({
                    "Metric": name,
                    "Cohort":  f"{rate_co:.2f}% ({int(num_co):,}/{int(den_co):,})",
                    "Rest":    f"{rate_re:.2f}% ({int(num_re):,}/{int(den_re):,})",
                    "Lift":    f"{lift:+.2f}%",
                    "Stat":    stat_s,
                    "Verdict": vd,
                })

            add_prop_row("Landing → Onboarding",
                         m_co['Onboarding Users'], m_co['Visitors'],
                         m_re['Onboarding Users'], m_re['Visitors'])
            add_prop_row("Landing → Registration",
                         m_co['Registered Users'], m_co['Visitors'],
                         m_re['Registered Users'], m_re['Visitors'])
            add_prop_row("Registration → Payer",
                         m_co['Payers'], m_co['Registered Users'],
                         m_re['Payers'], m_re['Registered Users'])
            add_prop_row("Reg → Payer 0d",
                         m_co['Payers 0d (Landing)'], m_co['Registered Users'],
                         m_re['Payers 0d (Landing)'], m_re['Registered Users'])
            add_prop_row("Landing → Payer (24h)",
                         m_co['Payers 0d (Landing)'], m_co['Visitors'],
                         m_re['Payers 0d (Landing)'], m_re['Visitors'])

            # Monetary (середні — без стат-теста, просто значення)
            metric_rows.append({
                "Metric":  "ARPU 0d ($)",
                "Cohort":  f"${m_co['ARPU 0d']:.2f}",
                "Rest":    f"${m_re['ARPU 0d']:.2f}",
                "Lift":    f"{m_co['ARPU 0d'] - m_re['ARPU 0d']:+.2f}",
                "Stat":    "",
                "Verdict": "",
            })
            metric_rows.append({
                "Metric":  "ARPPU 0d ($)",
                "Cohort":  f"${m_co['ARPPU 0d']:.2f}",
                "Rest":    f"${m_re['ARPPU 0d']:.2f}",
                "Lift":    f"{m_co['ARPPU 0d'] - m_re['ARPPU 0d']:+.2f}",
                "Stat":    "",
                "Verdict": "",
            })
            metric_rows.append({
                "Metric":  "Total Revenue ($)",
                "Cohort":  f"${m_co['Total Revenue']:,.0f}",
                "Rest":    f"${m_re['Total Revenue']:,.0f}",
                "Lift":    "",
                "Stat":    "",
                "Verdict": "",
            })

            df_metrics = pd.DataFrame(metric_rows)

            def color_row(row):
                v = row.get('Verdict', '')
                if 'Winner' in v: return ['background-color: #d1e7dd'] * len(row)
                if 'Loser'  in v: return ['background-color: #f8d7da'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_metrics.style.apply(color_row, axis=1),
                hide_index=True, use_container_width=True
            )

            # Funnel chart for cohort
            st.markdown("##### Funnel (absolute users)")
            funnel_stages = ['Visitors', 'Onboarding Users', 'Registered Users',
                             'Payers 0d (Landing)', 'Payers']
            funnel_data = []
            for stage in funnel_stages:
                funnel_data.append({'Group': 'Cohort', 'Stage': stage, 'Users': m_co[stage]})
                funnel_data.append({'Group': 'Rest',   'Stage': stage, 'Users': m_re[stage]})
            df_fun = pd.DataFrame(funnel_data)
            fig_fun = px.bar(df_fun, x='Stage', y='Users', color='Group', barmode='group',
                             text='Users')
            fig_fun.update_traces(texttemplate='%{text:,}', textposition='outside',
                                  cliponaxis=False)
            fig_fun.update_layout(height=400, margin=dict(t=30, b=30))
            st.plotly_chart(fig_fun, use_container_width=True)

        else:
            # Тільки розмір когорти
            st.info(
                f"Когорта містить **{len(cohort_users):,}** unique users "
                f"({len(cohort_users)/max(len(flow_users),1)*100:.1f}% від воронки). "
                "Для funnel-метрик завантаж land_1.csv."
            )


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — Cross-tab (Screen A → Screen B)
# ═════════════════════════════════════════════════════════════════════
with tab_xtab:
    st.subheader("Cross-tab: відповідь на екрані A → відповідь на екрані B")
    st.caption("Юзери що дійшли до обох екранів. Sankey показує абсолютні потоки, "
               "stacked bar показує % розподіл всередині кожної відповіді A.")

    question_screens = [s for s in screen_order
                        if s in df_q['screen_id'].unique() and s not in INTERSTITIAL_SCREENS]

    if len(question_screens) < 2:
        st.warning("Потрібно щонайменше 2 question-екрани в даних.")
    else:
        cx1, cx2, cx3 = st.columns([2, 2, 1])
        with cx1:
            scr_a = st.selectbox("Screen A", question_screens, index=0, key='xtab_a')
        with cx2:
            default_b_idx = 1 if len(question_screens) > 1 else 0
            scr_b = st.selectbox("Screen B", question_screens,
                                  index=default_b_idx, key='xtab_b')
        with cx3:
            top_a = st.number_input("Top A", 3, 50, value=8)
            top_b = st.number_input("Top B", 3, 50, value=8)

        cx4, cx5, cx6 = st.columns([1, 1, 2])
        with cx4:
            split_a = st.checkbox("Split A multi", value=is_multi_select(flow, scr_a),
                                  key='xtab_split_a')
        with cx5:
            split_b = st.checkbox("Split B multi", value=is_multi_select(flow, scr_b),
                                  key='xtab_split_b')
        with cx6:
            min_users = st.number_input("Hide cells with users <", 0, 1000, value=10,
                                         help="Прибирає шумові комбінації")

        if scr_a == scr_b:
            st.warning("A і B мають бути різними екранами.")
        else:
            sub_a = df_q[df_q['screen_id'] == scr_a][['user_id', 'answer_value']].rename(
                columns={'answer_value': 'A'}
            )
            sub_b = df_q[df_q['screen_id'] == scr_b][['user_id', 'answer_value']].rename(
                columns={'answer_value': 'B'}
            )
            if split_a:
                sub_a = sub_a.assign(A=sub_a['A'].astype(str).str.split(',')).explode('A')
                sub_a['A'] = sub_a['A'].str.strip()
                sub_a = sub_a[sub_a['A'] != '']
            if split_b:
                sub_b = sub_b.assign(B=sub_b['B'].astype(str).str.split(',')).explode('B')
                sub_b['B'] = sub_b['B'].str.strip()
                sub_b = sub_b[sub_b['B'] != '']

            # INNER join — тільки юзери що в обох екранах
            merged = sub_a.merge(sub_b, on='user_id', how='inner')

            if merged.empty:
                st.info("Немає юзерів які пройшли обидва екрани.")
            else:
                # Top-N для A та B (за к-стю user_id)
                top_a_vals = (merged.groupby('A')['user_id'].nunique()
                              .sort_values(ascending=False).head(int(top_a)).index.tolist())
                top_b_vals = (merged.groupby('B')['user_id'].nunique()
                              .sort_values(ascending=False).head(int(top_b)).index.tolist())
                m_filt = merged[merged['A'].isin(top_a_vals) & merged['B'].isin(top_b_vals)]

                xtab = (m_filt.groupby(['A', 'B'])['user_id']
                              .nunique().reset_index(name='users'))
                xtab = xtab[xtab['users'] >= int(min_users)]

                if xtab.empty:
                    st.info("Усі комбінації відфільтровано порогом 'Hide cells'.")
                else:
                    # % within A
                    totals_a = xtab.groupby('A')['users'].sum().rename('total_A')
                    xtab = xtab.merge(totals_a, on='A')
                    xtab['pct_within_A'] = xtab['users'] / xtab['total_A'] * 100

                    # ── Sankey
                    nodes_a = [f"A: {v}" for v in top_a_vals if v in xtab['A'].unique()]
                    nodes_b = [f"B: {v}" for v in top_b_vals if v in xtab['B'].unique()]
                    all_nodes = nodes_a + nodes_b
                    node_idx = {n: i for i, n in enumerate(all_nodes)}

                    sankey_src = [node_idx[f"A: {a}"] for a in xtab['A']]
                    sankey_tgt = [node_idx[f"B: {b}"] for b in xtab['B']]
                    sankey_val = xtab['users'].tolist()
                    sankey_lbl = [f"{r['users']:,} ({r['pct_within_A']:.1f}% of '{r['A']}')"
                                  for _, r in xtab.iterrows()]

                    fig_sankey = go.Figure(go.Sankey(
                        node=dict(
                            label=all_nodes,
                            pad=18, thickness=18,
                            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
                        ),
                        link=dict(
                            source=sankey_src,
                            target=sankey_tgt,
                            value=sankey_val,
                            customdata=sankey_lbl,
                            hovertemplate='%{customdata}<extra></extra>',
                        )
                    ))
                    fig_sankey.update_layout(
                        title=f"Sankey · {scr_a} → {scr_b}",
                        height=600, margin=dict(t=50, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig_sankey, use_container_width=True)

                    # ── Stacked bar (% within A)
                    st.markdown("##### Stacked bar — розподіл B всередині кожної відповіді A")
                    fig_stack = px.bar(
                        xtab, x='A', y='pct_within_A', color='B',
                        text=xtab['pct_within_A'].map(lambda v: f"{v:.0f}%"),
                        title=f"% B within A · {scr_a} → {scr_b}",
                        category_orders={'A': top_a_vals, 'B': top_b_vals},
                    )
                    fig_stack.update_traces(textposition='inside')
                    fig_stack.update_layout(
                        barmode='stack',
                        yaxis_ticksuffix='%',
                        xaxis_tickangle=-30,
                        height=520,
                        margin=dict(t=50, b=120),
                    )
                    st.plotly_chart(fig_stack, use_container_width=True)

                    # ── Heatmap-style table (text)
                    with st.expander("📋 Cross-tab table (raw + %)"):
                        pivot_users = xtab.pivot(index='A', columns='B', values='users').fillna(0).astype(int)
                        pivot_pct   = xtab.pivot(index='A', columns='B', values='pct_within_A').fillna(0)

                        # Строкова комбінована таблиця
                        combined = pivot_users.copy().astype(str)
                        for col in pivot_users.columns:
                            combined[col] = pivot_users[col].map(lambda v: f"{int(v):,}") + \
                                            "  (" + pivot_pct[col].map(lambda v: f"{v:.1f}%") + ")"
                        combined = combined.reindex(
                            [v for v in top_a_vals if v in combined.index]
                        )
                        combined = combined[[c for c in top_b_vals if c in combined.columns]]
                        st.dataframe(combined.reset_index(), hide_index=True,
                                     use_container_width=True)
