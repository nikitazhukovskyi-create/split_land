import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# --- Configuration ---
st.set_page_config(
    page_title="Universal Land Analysis",
    page_icon="🌍",
    layout="wide"
)

# --- Constants ---
DEFAULT_CSV_PATH = 'land_1.csv'
DATE_COLS = ['created_at', 'profile_created_at', 'reg_at', 'fo_at', 'landing_at', 'order_created_at']

# --- 1. Data Loading & Preprocessing ---

@st.cache_data
def load_data(file_path_or_buffer):
    """Loads and preprocesses data."""
    try:
        df = pd.read_csv(file_path_or_buffer)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Date Conversion (Added landing_at, order_created_at)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Currency Adjustment (Divide by 100)
    if 'amount' in df.columns:
        df['amount'] = df['amount'] / 100.0

    # Product Clustering Logic
    if 'id' in df.columns:
        def assign_cluster(pkg_id):
            if not isinstance(pkg_id, str): return 'other'
            pkg_id = pkg_id.lower()
            
            if pkg_id in ['credits-1550-taxes-v4', 'credits-2950-taxes-v4', 'credits-595-taxes-v4']:
                return 'credits'
            if pkg_id in ['pkg-premium-advanced-taxes-ntf3-6m-v1', 'pkg-premium-advanced-taxes-ntfm25-6m-v1', 'pkg-premium-advanced-taxes-ntfm25-smart-6m-v1']:
                return 'premium 6m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-3m':
                return 'gold 3m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-6m':
                return 'gold 6m'
            if pkg_id == 'pkg-premium-gold-taxes-ntf3-v2':
                return 'gold 1m'
            if pkg_id in ['pkg-premium-intermidiate-taxes-ntf3-3m-v1', 'pkg-premium-intermidiate-taxes-ntfm25-3m-v1', 'pkg-premium-intermidiate-taxes-ntfm25-smart-3m-v1']:
                return 'premium 3m'
            if pkg_id in ['pkg-premium-standard-taxes-ntf3-1m-v7', 'pkg-premium-standard-taxes-ntfm25-1m-v7', 'pkg-premium-standard-taxes-ntfm25-smart-1m-v7']:
                return 'premium 1m'
            
            return 'other'
        
        df['product_cluster'] = df['id'].apply(assign_cluster)
    
    return df

# --- 2. Metrics Definition ---

def calculate_metrics(df):
    """Calculates funnel and financial metrics including Day 0 specifics."""
    
    # 1. Base Counts
    visitors = df['user_id'].nunique()
    onboarding = df[df['profile_created_at'].notnull()]['user_id'].nunique() if 'profile_created_at' in df.columns else 0
    registered = df[df['reg_at'].notnull()]['user_id'].nunique() if 'reg_at' in df.columns else 0
    payers = df[df['fo_at'].notnull()]['user_id'].nunique() if 'fo_at' in df.columns else 0
    
    # Payers D0 (Legacy: fo_at - reg_at <= 24h)
    if 'fo_at' in df.columns and 'reg_at' in df.columns:
        df_payers = df.dropna(subset=['fo_at', 'reg_at'])
        df_payers_d0 = df_payers[(df_payers['fo_at'] - df_payers['reg_at']) <= pd.Timedelta(hours=24)]
        payers_d0 = df_payers_d0['user_id'].nunique()
    else:
        payers_d0 = 0

    # 2. Financials
    total_amount = df['amount'].sum() if 'amount' in df.columns else 0
    
    # ARPU (Revenue / Visitors) - Standard
    arpu = total_amount / visitors if visitors > 0 else 0
    
    # ARPPU (Revenue / Payers) - Standard
    arppu = total_amount / payers if payers > 0 else 0

    # 3. New Day 0 Metrics (Based on landing_at)
    # Reg -> Payer 0d (Landing): fo_at - landing_at <= 24h
    payers_0d_land_count = 0
    rev_0d_land = 0
    
    if 'fo_at' in df.columns and 'landing_at' in df.columns:
        df_pay_land = df.dropna(subset=['fo_at', 'landing_at'])
        # Unique users who matched condition
        payers_0d_land_count = df_pay_land[(df_pay_land['fo_at'] - df_pay_land['landing_at']) <= pd.Timedelta(hours=24)]['user_id'].nunique()
        
    if 'amount' in df.columns and 'order_created_at' in df.columns and 'landing_at' in df.columns:
         df_ord = df.dropna(subset=['order_created_at', 'landing_at', 'amount'])
         df_ord_0d = df_ord[(df_ord['order_created_at'] - df_ord['landing_at']) <= pd.Timedelta(hours=24)]
         rev_0d_land = df_ord_0d['amount'].sum()

    # ARPU 0d ($): Revenue 0d / Registered Users
    arpu_0d_val = rev_0d_land / registered if registered > 0 else 0
    
    # ARPPU 0d ($): Revenue 0d / Payers 0d (Landing)
    arppu_0d_val = rev_0d_land / payers_0d_land_count if payers_0d_land_count > 0 else 0

    # Landing -> Payer (24h): conversion from Landing to Payer within 24h of landing_at
    # Uses Visitors as denominator (not Registered Users)
    conv_land_payer_24h = (payers_0d_land_count / visitors * 100) if visitors > 0 else 0

    return {
        'Visitors': visitors,
        'Onboarding Users': onboarding,
        'Registered Users': registered,
        'Payers': payers,
        'Payers (Day 0)': payers_d0,
        'Total Revenue': total_amount,
        'ARPU': arpu,
        'ARPPU': arppu,
        # New Metrics Handlers
        'Payers 0d (Landing)': payers_0d_land_count,
        'Revenue 0d (Landing)': rev_0d_land,
        'ARPU 0d': arpu_0d_val,
        'ARPPU 0d': arppu_0d_val
    }

def get_conversion_rates(metrics):
    """Calculates conversion rates based on metrics dict."""
    visitors = metrics['Visitors']
    onboarding = metrics['Onboarding Users']
    registered = metrics['Registered Users']
    payers = metrics['Payers']
    
    conv_landing_onboarding = (onboarding / visitors * 100) if visitors > 0 else 0
    conv_landing_registration = (registered / visitors * 100) if visitors > 0 else 0
    conv_onboarding_registration = (registered / onboarding * 100) if onboarding > 0 else 0
    conv_registration_payer = (payers / registered * 100) if registered > 0 else 0
    conv_registration_payer_d0 = (metrics['Payers (Day 0)'] / registered * 100) if registered > 0 else 0
    
    # New: Reg -> Payer 0d (Landing)
    # Denom: Total Registered Users
    conv_reg_payer_0d_land = (metrics['Payers 0d (Landing)'] / registered * 100) if registered > 0 else 0
    
    # New: Landing -> Payer (24h)
    # Denom: Total Visitors (Landing)
    conv_land_payer_24h = (metrics['Payers 0d (Landing)'] / visitors * 100) if visitors > 0 else 0
    
    return {
        'Landing -> Onboarding': conv_landing_onboarding,
        'Landing -> Registration': conv_landing_registration,
        'Onboarding -> Registration': conv_onboarding_registration,
        'Registration -> Payer': conv_registration_payer,
        'Registration -> Payer D0': conv_registration_payer_d0,
        'Reg -> Payer 0d': conv_reg_payer_0d_land,
        'Landing -> Payer (24h)': conv_land_payer_24h,
    }

# --- 3. Statistical Engines ---

@st.cache_data
def run_bayesian_simulation_proportion(success_c, n_c, success_v, n_v, n_samples=10000):
    """Bayesian simulation for proportions (Beta)."""
    n_c, success_c = max(0, int(n_c)), max(0, int(success_c))
    n_v, success_v = max(0, int(n_v)), max(0, int(success_v))
    
    if n_c == 0 or n_v == 0: return 0.5, 0.0
    if success_c > n_c: success_c = n_c
    if success_v > n_v: success_v = n_v
        
    try:
        posterior_c = np.random.beta(1 + success_c, 1 + (n_c - success_c), n_samples)
        posterior_v = np.random.beta(1 + success_v, 1 + (n_v - success_v), n_samples)
        prob_beat = np.mean(posterior_v > posterior_c)
        mean_c = np.mean(posterior_c)
        uplift = (np.mean(posterior_v) - mean_c) / mean_c * 100 if mean_c > 0 else 0.0
        return prob_beat, uplift
    except Exception:
        return 0.5, 0.0

@st.cache_data
def run_bayesian_simulation_revenue(data_c, data_v, n_samples=10000):
    """Bayesian simulation for means (Normal)."""
    if len(data_c) < 2 or len(data_v) < 2: return 0.5, 0.0

    mu_c, std_c = np.mean(data_c), np.std(data_c, ddof=1)
    mu_v, std_v = np.mean(data_v), np.std(data_v, ddof=1)
    
    sem_c = std_c / np.sqrt(len(data_c))
    sem_v = std_v / np.sqrt(len(data_v))
    
    posterior_c = np.random.normal(mu_c, sem_c, n_samples)
    posterior_v = np.random.normal(mu_v, sem_v, n_samples)
    
    prob_beat = np.mean(posterior_v > posterior_c)
    uplift = np.mean(posterior_v) - np.mean(posterior_c)
    
    return prob_beat, uplift

def run_frequentist_tests(success_c, n_c, success_v, n_v):
    """Z-test for proportions."""
    if n_c == 0 or n_v == 0: return 1.0, 0.0
    count = np.array([success_v, success_c])
    nobs = np.array([n_v, n_c])
    try:
        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
        rate_c = success_c / n_c
        rate_v = success_v / n_v
        uplift = (rate_v - rate_c) / rate_c * 100 if rate_c > 0 else 0.0
        return pval, uplift
    except:
        return 1.0, 0.0

def run_frequentist_means(data_c, data_v):
    """T-test for means."""
    if len(data_c) < 2 or len(data_v) < 2: return 1.0, 0.0
    try:
        stat, pval = stats.ttest_ind(data_v, data_c, equal_var=False)
        uplift = np.mean(data_v) - np.mean(data_c)
        return pval, uplift
    except:
        return 1.0, 0.0

def run_statistics(df, df_c, control_variant, variants, method='Bayesian'):
    """Runs tests based on selected method, comparison against control df provided."""
    results = {}
    
    m_control = calculate_metrics(df_c)
    
    def get_rev_vec(dframe):
        reg = dframe[dframe['reg_at'].notnull()]['user_id'].unique()
        pay = dframe.groupby('user_id')['amount'].sum().reset_index()
        merged = pd.Series(reg, name='user_id').to_frame().merge(pay, on='user_id', how='left').fillna(0)
        return merged['amount'].values, dframe.dropna(subset=['fo_at'])['amount'].values

    # New Vectors for 0d Revenue Stats? 
    # For now, we apply stats to main metrics. 
    # User didn't strictly ask for stats on 0d metrics, just to add them to table. 
    # I will stick to main metrics for stats to avoid complexity explosion, or add if easy.
    # New metrics are "Reg -> Payer 0d", "ARPU 0d", "ARPPU 0d".
    # I will add Stats for them if possible.
    
    # Vectors for Control
    vec_arpu_c, vec_arppu_c = get_rev_vec(df_c)
    
    # Vectors for ARPU 0d?
    # ARPU 0d = Revenue 0d per Registered User.
    # We need a vector of 0d revenue per registered user.
    def get_rev_0d_vec(dframe):
        if 'landing_at' not in dframe.columns or 'order_created_at' not in dframe.columns:
            return np.array([]), np.array([])
            
        reg_users = dframe[dframe['reg_at'].notnull()]['user_id'].unique()
        
        # Filter orders in 0d
        df_ord = dframe.dropna(subset=['order_created_at', 'landing_at', 'amount'])
        df_ord_0d = df_ord[(df_ord['order_created_at'] - df_ord['landing_at']) <= pd.Timedelta(hours=24)]
        
        pay_0d = df_ord_0d.groupby('user_id')['amount'].sum().reset_index()
        
        # ARPU 0d Vector: All reg users, 0 if no pay
        merged_arpu = pd.Series(reg_users, name='user_id').to_frame().merge(pay_0d, on='user_id', how='left').fillna(0)
        
        # ARPPU 0d Vector: Only Payers 0d
        # Payers 0d defined as those who paid within 24h of landing
        # This is exactly pay_0d['amount']
        return merged_arpu['amount'].values, pay_0d['amount'].values

    vec_arpu_0d_c, vec_arppu_0d_c = get_rev_0d_vec(df_c)
    
    n_vis_c = m_control['Visitors']
    n_reg_c = m_control['Registered Users']

    for v in variants:
        df_v = df[df['landingId'] == v]
        m_v = calculate_metrics(df_v)
        vec_arpu_v, vec_arppu_v = get_rev_vec(df_v)
        vec_arpu_0d_v, vec_arppu_0d_v = get_rev_0d_vec(df_v)
        
        n_vis_v = m_v['Visitors']
        n_reg_v = m_v['Registered Users']
        
        tests = {}
        
        if method == 'Frequentist (Classical)':
            tests['Landing -> Onboarding'] = run_frequentist_tests(
                m_v['Onboarding Users'], n_vis_v, m_control['Onboarding Users'], n_vis_c)
            tests['Landing -> Registration'] = run_frequentist_tests(
                m_v['Registered Users'], n_vis_v, m_control['Registered Users'], n_vis_c)
            tests['Registration -> Payer'] = run_frequentist_tests(
                m_v['Payers'], n_reg_v, m_control['Payers'], n_reg_c)
            tests['Reg -> Payer 0d'] = run_frequentist_tests(
                m_v['Payers 0d (Landing)'], n_reg_v, m_control['Payers 0d (Landing)'], n_reg_c)
            tests['Landing -> Payer (24h)'] = run_frequentist_tests(
                m_v['Payers 0d (Landing)'], n_vis_v, m_control['Payers 0d (Landing)'], n_vis_c)
                
            tests['ARPU'] = run_frequentist_means(vec_arpu_c, vec_arpu_v)
            tests['ARPPU'] = run_frequentist_means(vec_arppu_c, vec_arppu_v)
            tests['ARPU 0d'] = run_frequentist_means(vec_arpu_0d_c, vec_arpu_0d_v)
            tests['ARPPU 0d'] = run_frequentist_means(vec_arppu_0d_c, vec_arppu_0d_v)
            
        else: # Bayesian
            tests['Landing -> Onboarding'] = run_bayesian_simulation_proportion(
                m_control['Onboarding Users'], n_vis_c, m_v['Onboarding Users'], n_vis_v)
            tests['Landing -> Registration'] = run_bayesian_simulation_proportion(
                m_control['Registered Users'], n_vis_c, m_v['Registered Users'], n_vis_v)
            tests['Registration -> Payer'] = run_bayesian_simulation_proportion(
                m_control['Payers'], n_reg_c, m_v['Payers'], n_reg_v)
            tests['Reg -> Payer 0d'] = run_bayesian_simulation_proportion(
                m_control['Payers 0d (Landing)'], n_reg_c, m_v['Payers 0d (Landing)'], n_reg_v)
            tests['Landing -> Payer (24h)'] = run_bayesian_simulation_proportion(
                m_control['Payers 0d (Landing)'], n_vis_c, m_v['Payers 0d (Landing)'], n_vis_v)
                
            tests['ARPU'] = run_bayesian_simulation_revenue(vec_arpu_c, vec_arpu_v)
            tests['ARPPU'] = run_bayesian_simulation_revenue(vec_arppu_c, vec_arppu_v)
            tests['ARPU 0d'] = run_bayesian_simulation_revenue(vec_arpu_0d_c, vec_arpu_0d_v)
            tests['ARPPU 0d'] = run_bayesian_simulation_revenue(vec_arppu_0d_c, vec_arppu_0d_v)
        
        results[v] = tests
        
    return results

    return results

def generate_comprehensive_summary(df, df_c, control_variant, variants, method, allowed_dims=None):
    """
    Scans dimensions (Global, 1-Level, 2-Level) for significant findings.
    Returns a list of structured findings.
    """
    findings = []
    
    # 1. Config & Dimensions
    metrics = [
        'Landing -> Onboarding', 'Landing -> Registration', 'Registration -> Payer',
        'Reg -> Payer 0d', 'Landing -> Payer (24h)', 'ARPU', 'ARPPU', 'ARPU 0d', 'ARPPU 0d'
    ]
    
    # Use allowed dims if provided, else auto-detect (though logic is now in render_dashboard)
    if allowed_dims is not None:
        cat_cols = allowed_dims
    else:
        exclude_cols = ['user_id', 'landingId', 'amount', 'id', 'product_cluster'] + DATE_COLS
        cat_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_string_dtype(df[c])]
    # Prioritize Country/Platform/OS if present?
    # HEURISTIC: Limit to top 5 values per dimension to prevent explosion
    
    # Global Scan
    stats_global = run_statistics(df, df_c, control_variant, variants, method=method)
    
    def parse_stats(res_stats, segment_name, counts):
        for v in variants:
            res = res_stats.get(v, {})
            for m in metrics:
                val, uplift = res.get(m, (0.5, 0.0) if method.startswith('Bayesian') else (1.0, 0.0))
                
                # Check Significance
                is_sig = False
                confidence_str = ""
                
                if method.startswith('Bayesian'):
                    if val > 0.95: 
                        is_sig = True
                        confidence_str = f"Prob {val:.1%}"
                        res_type = 'winner'
                    elif val < 0.05:
                        is_sig = True
                        confidence_str = f"Prob {val:.1%}"
                        res_type = 'loser'
                else: # Frequentist
                    if val < 0.05:
                        is_sig = True
                        confidence_str = f"p={val:.4f}"
                        res_type = 'winner' if uplift > 0 else 'loser'
                
                if is_sig:
                    # Score for ranking: Abs(Uplift) * Confidence? 
                    # Just Abs(Uplift) is fine for now, prioritizing winners.
                    score = abs(uplift)
                    findings.append({
                        'metric': m,
                        'segment': segment_name,
                        'variant': v,
                        'result': res_type,
                        'uplift': uplift,
                        'confidence': confidence_str,
                        'score': score,
                        'obs': counts.get(v, 0) # Store N obs
                    })

    # Run Global
    # Helper to clean getting n_obs
    def get_counts(dframe):
        return dframe['landingId'].value_counts().to_dict()

    parse_stats(stats_global, "Global (All Traffic)", get_counts(df))
    
    # 2. 1-Level Breakdown
    # Loop each col, top 5 values
    for col in cat_cols:
        top_vals = df[col].value_counts().nlargest(5).index.tolist()
        for val in top_vals:
            seg_name = f"{col}={val}"
            # Filter
            sub_df = df[df[col] == val]
            sub_df_c = df_c[df_c[col] == val]
            
            if not sub_df.empty and not sub_df_c.empty:
                # Run Stats
                s_res = run_statistics(sub_df, sub_df_c, control_variant, variants, method=method)
                parse_stats(s_res, seg_name, get_counts(sub_df))

    # 3. 2-Level Breakdown (Top 2 dims only)
    priority_cols = ['country', 'platform_name', 'device_model', 'os','source']
    selected_dims = [c for c in priority_cols if c in cat_cols]
    # Fill with others if not enough
    for c in cat_cols:
        if c not in selected_dims: selected_dims.append(c)
    
    dims_2 = selected_dims[:2]
    
    if len(dims_2) == 2:
        c1, c2 = dims_2[0], dims_2[1]
        top_vals_1 = df[c1].value_counts().nlargest(3).index.tolist() # Top 3 only
        top_vals_2 = df[c2].value_counts().nlargest(3).index.tolist()
        
        for v1 in top_vals_1:
            for v2 in top_vals_2:
                seg_name = f"{c1}={v1} & {c2}={v2}"
                sub_df = df[(df[c1] == v1) & (df[c2] == v2)]
                sub_df_c = df_c[(df_c[c1] == v1) & (df_c[c2] == v2)]
                
                if not sub_df.empty and not sub_df_c.empty:
                     s_res = run_statistics(sub_df, sub_df_c, control_variant, variants, method=method)
                     parse_stats(s_res, seg_name, get_counts(sub_df))

    # Sort and Format
    # Sort findings by Obs (Sample Size) desc as requested
    findings.sort(key=lambda x: x['obs'], reverse=True)
    
    return findings

def render_summary_widget(findings, control_var):
    st.markdown("### 🎯 Automated Insights (Significant Findings)")
    
    if not findings:
        st.info("ℹ️ No statistically significant findings detected across major segments.")
        return

    # A. Category Filter
    c_f_col, _ = st.columns([1, 2])
    cat_filter = c_f_col.radio("Metric Type", ['All', 'Monetary', 'Conversion'], horizontal=True)

    filtered_findings = []
    for f in findings:
        is_mon = 'Ar' in f['metric'] or 'Rev' in f['metric'] or 'AR' in f['metric']
        if cat_filter == 'Monetary' and not is_mon: continue
        if cat_filter == 'Conversion' and is_mon: continue
        filtered_findings.append(f)
        
    if not filtered_findings:
        st.info(f"No findings for category: {cat_filter}")
        return

    # Split Wins and Losses
    wins = [f for f in filtered_findings if f['result'] == 'winner']
    losses = [f for f in filtered_findings if f['result'] == 'loser']

    # Sort by Sample Size
    wins.sort(key=lambda x: x['obs'], reverse=True)
    losses.sort(key=lambda x: x['obs'], reverse=True)
    
    losses.sort(key=lambda x: x['obs'], reverse=True)
    
    # B. Short Name Helper
    def clean_name(s):
        if not isinstance(s, str): return str(s)
        return s.split('-')[-1] # Take last part

    def format_df(f_list):
        if not f_list: return pd.DataFrame()
        data = []
        for f in f_list:
            # Lift formatting
            lift_s = f"{f['uplift']:+.1f}%" if 'Conv' in f['metric'] or 'Reg' in f['metric'] else f"{f['uplift']:+.2f}"
            if 'Ar' in f['metric'] or 'Rev' in f['metric'] or 'AR' in f['metric']: # Financial
                 if '%' not in lift_s: lift_s = f"${f['uplift']:+.2f}"
            
            # Format Obs in k
            obs_s = f"{f['obs']/1000:.1f}k" if f['obs'] >= 1000 else str(f['obs'])
            
            # Format Obs in k (Just for tooltip/debug if we kept it, but user wants it gone from table)
            # We keep it in the dict construction? No, user said "Code drops the 'Obs' column".
            # Cleanest is to not append it to `data` list.

            # Append Comparison Context
            v_short = clean_name(f['variant'])
            c_short = clean_name(control_var)
            seg_s = f"{f['segment']} ({v_short} vs {c_short})"
            
            data.append({
                "Metric": f['metric'],
                "Segment": seg_s,
                "Lift": lift_s,
                "Stat": f['confidence'],
                # "Obs": obs_s # REMOVED per user request
            })
        return pd.DataFrame(data)

    c1, c2 = st.columns(2)
    
    with c1:
        st.error("📉 Significant Losses (V < Control)")
        if losses:
            df_loss = format_df(losses)
            # Display with color styling if possible, or just raw
            # Using st.dataframe with Pandas Styler for red text in Lift
            # Note: Streamlit dataframe styler support is good.
            st.dataframe(
                df_loss.style.map(lambda x: 'color: red', subset=['Lift']),
                width="stretch",
                hide_index=True
            )
        else:
            st.markdown("*No significant losses detected.*")
            
    with c2:
        st.success("🚀 Significant Wins (V > Control)")
        if wins:
            df_win = format_df(wins)
            st.dataframe(
                df_win.style.map(lambda x: 'color: green', subset=['Lift']),
                width="stretch",
                hide_index=True
            )
        else:
             st.markdown("*No significant wins detected.*")
    
    st.markdown("---")

# --- Verdict Logic ---
def get_stat_verdict(stat_val, uplift, method, n_c, n_v):
    # n_c and n_v should be sample sizes (e.g. Visitors or Registered Users)
    if n_c < 100 or n_v < 100:
        return "⚠️ Insufficient Data"
    
    is_bayesian = method.startswith('Bayesian')
    
    if is_bayesian:
        if stat_val > 0.95: return "🚀 High Confidence Winner (>95%)"
        if stat_val < 0.05: return "📉 High Confidence Loser (<5%)"
        return "⚖️ Inconclusive"
    else:
        # Frequentist
        if stat_val < 0.05:
            if uplift > 0: return "✅ Significant (Winner)"
            return "❌ Significant (Loser)"
        return "🤷 No Diff"

# --- 4. UI Layout ---

def render_dashboard():
    # Instructions Expander
    with st.expander("ℹ️ CSV Format Instructions (Click to Open)"):
        st.markdown('''
        **Required Columns:**
        - `user_id`: Unique Identifier
        - `landingId`: Variant Name
        - `landing_at`: Timestamp of landing visit
        - `order_created_at`: Timestamp of order creation
        - `reg_at`: Timestamp of registration
        - `fo_at`: Timestamp of first payment
        - `amount`: Revenue amount (cents or raw)
        - `id`: Package ID for granularity
        
        **Note:** Max file size 200MB.
        ''')

    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")
    stat_method = st.sidebar.radio(
        "Statistical Approach",
        ['Bayesian', 'Frequentist (Classical)'],
        help="Bayesian: Prob to Beat Control. Frequentist: P-value (Z-test/T-test)."
    )
    
    st.title(f"Universal Land Analysis ({stat_method.split()[0]}) 🌍")
    
    # Data Loading with Uploader Priority
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Custom CSV", type=['csv'], help="Upload your own data to analyze.")
    
    data = None
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
             st.sidebar.success("✅ Custom file loaded")
    else:
        # Fallback to default
        if DEFAULT_CSV_PATH:
             data = load_data(DEFAULT_CSV_PATH)
             if data is not None:
                 st.sidebar.info(f"Using default data: {DEFAULT_CSV_PATH}")

    if data is None:
        st.warning("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
        return
        
    # --- Universal Variant Selector ---
    all_variants = sorted(data['landingId'].dropna().astype(str).unique())
    selected_variants = st.multiselect(
        "Select Variants to Compare", 
        all_variants, 
        default=all_variants
    )
    
    if not selected_variants:
        st.warning("Please select at least one variant.")
        return

    # Filter by Variant Check
    df_main = data[data['landingId'].isin(selected_variants)]

    # --- Filters (Audience) ---
    with st.expander("🔍 Filter Audience", expanded=True):
        c1, c2 = st.columns(2)
        
        # Country Filter
        sel_all_c = True
        if 'country' in df_main.columns:
            all_c = sorted(df_main['country'].dropna().unique())
            sel_all_c = c1.checkbox("Select All Countries", value=True)
            if sel_all_c:
                countries = all_c
                c1.multiselect("Countries", all_c, default=all_c, disabled=True)
            else:
                countries = c1.multiselect("Countries", all_c, default=all_c)
            if not countries: countries = all_c
        else:
            countries = []
        
        # Dynamic Platform Filter (if exists)
        platforms = []
        sel_all_p = True
        if 'platform_name' in df_main.columns:
            all_p = sorted(df_main['platform_name'].dropna().unique())
            sel_all_p = c2.checkbox("Select All Platforms", value=True)
            if sel_all_p:
                platforms = all_p
                c2.multiselect("Platforms", all_p, default=all_p, disabled=True)
            else:
                platforms = c2.multiselect("Platforms", all_p, default=all_p)
        
        # Traffic Source Filter
        traffic_source = []
        sel_all_s = True
        if 'source' in df_main.columns:
            all_s = sorted(df_main['source'].dropna().unique())
            sel_all_s = c2.checkbox("Select All traffic sources", value=True)
            if sel_all_s:
                traffic_source = all_s
                c2.multiselect("traffic source", all_s, default=all_s, disabled=True)
            else:
                traffic_source = c2.multiselect("traffic source", all_s, default=all_s)

        # Apply Filters Dynamically
        mask = pd.Series(True, index=df_main.index)
        
        if 'country' in df_main.columns and not sel_all_c:
             mask &= df_main['country'].isin(countries)
             
        if 'platform_name' in df_main.columns and not sel_all_p:
             mask &= df_main['platform_name'].isin(platforms)

        if 'source' in df_main.columns and not sel_all_s:
             mask &= df_main['source'].isin(traffic_source)
             
        df_filtered = df_main[mask]
             
    st.markdown(f"**Data Points**: {len(df_filtered)} rows | **Traffic Share**: {len(df_filtered)/len(data):.1%}")

    if df_filtered.empty:
        st.warning("No data matches filters.")
        return

    # Determine Control
    # Default to first one or 'mm-nsp-v1' if present
    possible_controls = [v for v in selected_variants if 'v1' in v]
    control_variant = possible_controls[0] if possible_controls else selected_variants[0]
    
    # Run Stats
    # Separating Control Data for stats function
    df_control_stats = df_filtered[df_filtered['landingId'] == control_variant]
    test_variants = [v for v in selected_variants if v != control_variant]
    
    st.header("Executive Summary")
    
    # Identify Dimensions for Scanning
    exclude_cols = ['user_id', 'landingId', 'amount', 'id', 'product_cluster'] + DATE_COLS
    all_cat_cols = [c for c in df_filtered.columns if c not in exclude_cols and pd.api.types.is_string_dtype(df_filtered[c])]
    
    # Dimension Filter UI
    selected_dims = st.multiselect("Select Dimensions to Scan for Insights", all_cat_cols, default=all_cat_cols)
    
    with st.spinner("🤖 Scanning all data dimensions for insights..."):
        findings = generate_comprehensive_summary(df_filtered, df_control_stats, control_variant, test_variants, method=stat_method, allowed_dims=selected_dims)
        render_summary_widget(findings, control_variant)
        
    # Recalculate global stats for table view (cached anyway)
    stats_data = run_statistics(df_filtered, df_control_stats, control_variant, test_variants, method=stat_method)

    # --- Metrics Table ---
    st.subheader("Detailed Performance")
    
    metrics_list = [
        'Landing -> Onboarding', 
        'Landing -> Registration', 
        'Registration -> Payer',
        'Reg -> Payer 0d',
        'Landing -> Payer (24h)',
        'ARPU', 
        'ARPPU',
        'ARPU 0d',
        'ARPPU 0d'
    ]
    
    m_c = calculate_metrics(df_filtered[df_filtered['landingId'] == control_variant])
    c_c = get_conversion_rates(m_c)
    # Map Financials to c_c for uniform access
    c_c.update({'ARPU': m_c['ARPU'], 'ARPPU': m_c['ARPPU'], 'ARPU 0d': m_c['ARPU 0d'], 'ARPPU 0d': m_c['ARPPU 0d']})
    
    # Store rows to create df later
    # We display Control column + Variant columns
    # Actually, the previous format was Metric | Control | Var 1 | Stat | Diff
    # If multiple variants, we repeat rows? Or Pivot?
    # Previous code: For v in variants: Print distinct table for V vs Control.
    # We will stick to that as it's cleaner for stats.
    
    for v in test_variants:
        st.markdown(f"### {v} vs {control_variant} (Control)")
        m_v = calculate_metrics(df_filtered[df_filtered['landingId'] == v])
        c_v = get_conversion_rates(m_v)
        c_v.update({'ARPU': m_v['ARPU'], 'ARPPU': m_v['ARPPU'], 'ARPU 0d': m_v['ARPU 0d'], 'ARPPU 0d': m_v['ARPPU 0d']})
        
        res = stats_data.get(v, {})
        rows = []
        for metric in metrics_list:
            val, uplift = res.get(metric, (0.5, 0.0) if stat_method.startswith('Bayesian') else (1.0, 0.0))
            is_mon = 'ARP' in metric # ARPU, ARPPU, ARPU 0d, ARPPU 0d
            
            # Helper to get context counts
            def get_ctx(m_dict, met_name):
                if met_name == 'Landing -> Onboarding':
                    return int(m_dict['Onboarding Users']), int(m_dict['Visitors'])
                elif met_name == 'Landing -> Registration':
                    return int(m_dict['Registered Users']), int(m_dict['Visitors'])
                elif met_name == 'Registration -> Payer':
                    return int(m_dict['Payers']), int(m_dict['Registered Users'])
                elif met_name == 'Reg -> Payer 0d':
                    return int(m_dict['Payers 0d (Landing)']), int(m_dict['Registered Users'])
                elif met_name == 'Landing -> Payer (24h)':
                    return int(m_dict['Payers 0d (Landing)']), int(m_dict['Visitors'])
                elif met_name == 'ARPU':
                    return m_dict['Total Revenue'], int(m_dict['Visitors'])
                elif met_name == 'ARPPU':
                    return m_dict['Total Revenue'], int(m_dict['Payers'])
                elif met_name == 'ARPU 0d':
                    return m_dict['Revenue 0d (Landing)'], int(m_dict['Registered Users'])
                elif met_name == 'ARPPU 0d':
                    return m_dict['Revenue 0d (Landing)'], int(m_dict['Payers 0d (Landing)'])
                return 0, 0

            # Control Context
            num_c, den_c = get_ctx(m_c, metric)
            if is_mon:
                fmt_c = f"${c_c.get(metric,0):.2f} (${num_c:,.0f}/{den_c:,})"
            else:
                fmt_c = f"{c_c.get(metric,0):.2f}% ({num_c:,}/{den_c:,})"

            # Variant Context
            num_v, den_v = get_ctx(m_v, metric)
            if is_mon:
                fmt_v = f"${c_v.get(metric,0):.2f} (${num_v:,.0f}/{den_v:,})"
            else:
                fmt_v = f"{c_v.get(metric,0):.2f}% ({num_v:,}/{den_v:,})"
            
            upl_d = f"${uplift:+.2f}" if is_mon else f"{uplift:+.2f}%"
            
            # Context for verdict
            verdict = get_stat_verdict(val, uplift, stat_method, den_c, den_v)
            
            if stat_method.startswith('Bayesian'):
                stat_s = f"{val:.1%}" # Clean text
            else:
                stat_s = f"{val:.4f}" # Clean text
            
            # Determine Sig for Coloring
            sig_type = "neutral"
            if "✅" in verdict or "🚀" in verdict: sig_type = "winner"
            elif "❌" in verdict or "📉" in verdict: sig_type = "loser"

            rows.append({
                "Metric": metric, 
                "Control": fmt_c, 
                "Variant": fmt_v, 
                "Diff / Uplift": upl_d,
                "Stat": stat_s, # Renamed col
                "_status": sig_type
            })
        
        df_display = pd.DataFrame(rows)
        
        # Apply Styling
        def highlight_row(row):
            status = row['_status']
            if status == 'winner':
                return ['background-color: #d1e7dd'] * len(row) # Light Green
            elif status == 'loser':
                return ['background-color: #f8d7da'] * len(row) # Light Red
            return [''] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_row, axis=1),
            width="stretch",
            hide_index=True,
            column_config={'_status': None}
        )

    # --- Visual Analysis ---
    st.subheader("Visual Analysis")
    t1, t2, t3, t4 = st.tabs(["Funnel", "Revenue Dist", "Deep Dive: ARPPU Impact", "Audience Structure"])

    with t1:
        # Funnel with Tooltips
        funnel_data = []
        # Add 0d stats to funnel? "Payers 0d"
        funnel_metric_names = ['Visitors', 'Onboarding Users', 'Registered Users', 'Payers', 'Payers 0d (Landing)']
        
        # We need all variants including control
        for v in [control_variant] + test_variants:
            m = calculate_metrics(df_filtered[df_filtered['landingId'] == v])
            for stage in funnel_metric_names:
                funnel_data.append({'Variant': v, 'Stage': stage, 'Count': m[stage]})
        
        df_funnel = pd.DataFrame(funnel_data).drop_duplicates()
        if not df_funnel.empty:
            # Sort Funnel Data Descending by Count
            df_funnel = df_funnel.sort_values(by='Count', ascending=False)
            
            fig_funnel = px.bar(df_funnel, x='Stage', y='Count', color='Variant', barmode='group')
            fig_funnel.update_traces(hovertemplate='<b>%{x}</b><br>Variant: %{fullData.name}<br>Count: %{y}<extra></extra>')
            st.plotly_chart(fig_funnel, use_container_width=True)

    with t2:
         fig_box = px.box(df_filtered, x='landingId', y='amount', title="Revenue per User Distribution")
         st.plotly_chart(fig_box, use_container_width=True)

    # --- Deep Dive: ARPPU Drivers ---
    with t3:
        st.markdown("##### ARPPU Drivers & Package Impact Analysis")
        
        c3, c4 = st.columns(2)
        comp_control = c3.selectbox("Control Group", selected_variants, index=0)
        # Default test to second variant if exists
        def_test_idx = 1 if len(selected_variants) > 1 else 0
        comp_test = c4.selectbox("Test Group", selected_variants, index=def_test_idx)
        
        pkg_view = st.radio("View Packages By:", ['Product Clusters', 'Raw Package IDs'], horizontal=True)
        col_group = 'product_cluster' if pkg_view == 'Product Clusters' else 'id'
        
        if col_group in df_filtered.columns:
            # Filter for just these two if needed? OR calculate for all but show impact for these two.
            # We need contributions for both.
            
            # Contributors
            payers_per_variant = df_filtered[df_filtered['fo_at'].notnull()].groupby('landingId')['user_id'].nunique().to_dict()
            df_rev_pkg = df_filtered[df_filtered['fo_at'].notnull()].groupby(['landingId', col_group])['amount'].sum().reset_index()
            
            df_rev_pkg['total_payers'] = df_rev_pkg['landingId'].map(payers_per_variant)
            df_rev_pkg['contribution'] = df_rev_pkg['amount'] / df_rev_pkg['total_payers']
            
            # Viz 1: Stacked Bar (All Selected Variants)
            # Filter df_rev_pkg for selected variants only? No, Keep all for context is nice, or strict?
            # User said "Show ALL selected variants" in Summary, here let's show all selected in stacked bar.
            fig_stack = px.bar(df_rev_pkg, x='landingId', y='contribution', color=col_group,
                               title=f"ARPPU Composition by {pkg_view} ($)", 
                               labels={'contribution': 'Contribution to ARPPU ($)', col_group: pkg_view})
            fig_stack.update_layout(yaxis_tickformat='$.2f')
            fig_stack.update_traces(hovertemplate='<b>%{data.name}</b><br>Contrib: $%{y:.2f}<extra></extra>')
            st.plotly_chart(fig_stack, use_container_width=True)
            
            # Viz 2: Impact Analysis (Custom Comparison)
            df_pivot = df_rev_pkg.pivot(index=col_group, columns='landingId', values='contribution').fillna(0)
            
            if comp_control in df_pivot.columns and comp_test in df_pivot.columns:
                st.markdown(f"**Impact Dictionary: {comp_test} vs {comp_control}**")
                
                diff_col = 'Impact'
                df_pivot[diff_col] = df_pivot[comp_test] - df_pivot[comp_control]
                df_impact = df_pivot[[diff_col]].sort_values(diff_col, ascending=True).reset_index()
                
                fig_imp = px.bar(df_impact, y=col_group, x=diff_col, orientation='h',
                                 title=f"{pkg_view} Impact on ARPPU: {comp_test} vs {comp_control}",
                                 color=diff_col, color_continuous_scale='RdBu')
                fig_imp.update_layout(xaxis_tickformat='$.2f')
                fig_imp.update_traces(hovertemplate='<b>%{y}</b><br>Impact: $%{x:.2f}<extra></extra>')
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Insight
                best_pkg = df_impact.iloc[-1]
                worst_pkg = df_impact.iloc[0]
                insight_text = []
                if best_pkg[diff_col] > 0.05:
                    insight_text.append(f"Growth driven by **{best_pkg[col_group]}** (+${best_pkg[diff_col]:.2f}).")
                if worst_pkg[diff_col] < -0.05:
                    insight_text.append(f"Offset by drop in **{worst_pkg[col_group]}** (${worst_pkg[diff_col]:.2f}).")
                if insight_text:
                    st.info(" ".join(insight_text))
            
        else:
            st.warning(f"Column '{col_group}' unavailable.")

    # --- Audience Structure (Dynamic) ---
    with t4:
        # Detect Categorical Columns
        # Exclude known non-dims
        exclude_cols = ['user_id', 'landingId', 'amount', 'id', 'product_cluster'] + DATE_COLS
        # Also exclude numeric types if not desired, but "concept" might be numeric-ish.
        # Select object types + maybe 'concept'?
        cat_cols = [c for c in df_filtered.columns if c not in exclude_cols and pd.api.types.is_string_dtype(df_filtered[c])]
        
        # Add special knowns if checks fail
        if 'country' in df_filtered.columns and 'country' not in cat_cols: cat_cols.append('country')
        if 'platform_name' in df_filtered.columns and 'platform_name' not in cat_cols: cat_cols.append('platform_name')
        if 'platform_model' in df_filtered.columns and 'platform_model' not in cat_cols: cat_cols.append('platform_model')
        if 'source' in df_filtered.columns and 'source' not in cat_cols: cat_cols.append('source')
        cat_cols = sorted(list(set(cat_cols)))
        
        if cat_cols:
            dim_sel = st.selectbox("Choose Breakdown Dimension", cat_cols, index=0 if 'country' not in cat_cols else cat_cols.index('country'))
            
            # Generic Chart Logic
            # Top 15 values to avoid clutter
            top_vals = df_filtered[dim_sel].value_counts().nlargest(15).index
            df_sub = df_filtered[df_filtered[dim_sel].isin(top_vals)]
            
            df_g = df_sub.groupby(['landingId', dim_sel]).size().reset_index(name='count')
            df_g['share'] = df_g.groupby('landingId')['count'].transform(lambda x: x/x.sum())
            
            fig = px.bar(df_g, x='landingId', y='share', color=dim_sel, barmode='stack', title=f"Audience Split by {dim_sel}")
            fig.layout.yaxis.tickformat = '.0%'
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns found for Audience Structure.")

if __name__ == "__main__":
    render_dashboard()
