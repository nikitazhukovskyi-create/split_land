import sys
import importlib.util
from pathlib import Path
import io
import pandas as pd

# Stub streamlit + plotly + scipy + statsmodels (page imports them at top level)
class _Stub:
    def __getattr__(self, _name):
        return _Stub()
    def __call__(self, *a, **kw):
        return _Stub()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **kw):
        # works as both @cache_data and @cache_data(show_spinner=False)
        if a and callable(a[0]):
            return a[0]
        def deco(fn): return fn
        return deco

for mod in ['streamlit', 'plotly', 'plotly.express', 'plotly.graph_objects',
            'scipy', 'scipy.stats',
            'statsmodels', 'statsmodels.stats', 'statsmodels.stats.proportion']:
    sys.modules[mod] = _Stub()

# Import the page module
_candidates = [
    Path('/Users/user/Desktop/Cowork/pages/2_Cohort_Quiz_Analysis.py'),
    Path('/sessions/charming-kind-mendel/mnt/Cowork/pages/2_Cohort_Quiz_Analysis.py'),
    Path(__file__).resolve().parent / 'pages' / '2_Cohort_Quiz_Analysis.py',
]
page_path = next(p for p in _candidates if p.exists())
# The page module runs UI code at top level after the helpers; we only want
# the helper functions. Extract function defs / module constants up to first
# `st.title(`/`st.set_page_config(`/`st.sidebar` etc by exec'ing only the
# top-level function defs and assignments using ast walking.
import ast
src = page_path.read_text()
tree = ast.parse(src)
# Keep only: imports, FunctionDef, ClassDef, Assign at module level.
keep = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                         ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                         ast.AnnAssign)):
        keep.append(node)
new_tree = ast.Module(body=keep, type_ignores=[])
ast.fix_missing_locations(new_tree)
mod_globals = {'__name__': 'cohort_page_helpers', '__file__': str(page_path)}
# Exec node-by-node so we can stop right after load_land1 is defined.
needed = ['_assign_cluster', '_extract_plan_duration', 'load_land1']
for node in new_tree.body:
    snippet = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(snippet)
    try:
        exec(compile(snippet, str(page_path), 'exec'), mod_globals)
    except Exception as e:
        # Skip nodes that fail (e.g. assignments referencing UI state) —
        # we just need the helpers above.
        pass
    if all(k in mod_globals for k in needed):
        break

class _ModView:
    def __init__(self, g): self._g = g
    def __getattr__(self, n): return self._g[n]
mod = _ModView(mod_globals)

# ─────────────────────────────────────────────────────────────────────
# 1. Test _assign_cluster + _extract_plan_duration
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. _assign_cluster + _extract_plan_duration")
print("=" * 60)

cases = [
    ('pkg-premium-standard-taxes-ntf3-1m-v7',     'premium 1m', '1m'),
    ('pkg-premium-intermidiate-taxes-ntf3-3m-v1', 'premium 3m', '3m'),
    ('pkg-premium-advanced-taxes-ntf3-6m-v1',     'premium 6m', '6m'),
    ('pkg-premium-gold-taxes-ntf3-3m',            'gold 3m',    '3m'),
    ('pkg-premium-gold-taxes-ntf3-6m',            'gold 6m',    '6m'),
    ('pkg-premium-gold-taxes-ntf3-v2',            'gold 1m',    '1m'),  # default
    ('credits-1550-taxes-v4',                     'credits',    'credits'),
    ('some-unknown-pkg-2m',                       'other',      '2m'),
    (None,                                        None,         None),
    (123,                                         None,         None),
]
for pkg, exp_cluster, exp_dur in cases:
    got_cluster = mod._assign_cluster(pkg)
    got_dur     = mod._extract_plan_duration(pkg)
    ok_c = got_cluster == exp_cluster
    ok_d = got_dur == exp_dur
    print(f"  {str(pkg)[:45]:<45} cluster={got_cluster!r:<14} ({'OK' if ok_c else 'FAIL'}) "
          f"dur={got_dur!r:<10} ({'OK' if ok_d else 'FAIL'})")
    assert ok_c, f"cluster mismatch for {pkg}: got {got_cluster}, want {exp_cluster}"
    assert ok_d, f"duration mismatch for {pkg}: got {got_dur}, want {exp_dur}"

# ─────────────────────────────────────────────────────────────────────
# 2. Test load_land1 derived columns on synthetic CSV
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. load_land1 → order_seq / is_upsell / tt_bucket")
print("=" * 60)

# 5 users:
#   U1 — no orders (visitor only)
#   U2 — 1 order (premium 1m, 15 min after landing)
#   U3 — 2 orders (premium 3m + gold 1m upsell, 90 min later)
#   U4 — 3 orders (premium 6m + 2 credits upsells, days later)
#   U5 — 1 order, but landing_at is NaT (silent landing edge case)

csv = """user_id,landingId,landing_at,country,reg_at,fo_at,profile_created_at,order_id,order_created_at,id,amount,source
U1,mm-sq1-v1,2026-04-01 10:00:00,US,,,,,,,,fb
U2,mm-sq1-v1,2026-04-01 10:00:00,US,2026-04-01 10:05:00,2026-04-01 10:15:00,2026-04-01 10:02:00,O201,2026-04-01 10:15:00,pkg-premium-standard-taxes-ntf3-1m-v7,2999,fb
U3,mm-sq1-v1,2026-04-02 09:00:00,DE,2026-04-02 09:03:00,2026-04-02 09:30:00,2026-04-02 09:01:00,O301,2026-04-02 09:30:00,pkg-premium-intermidiate-taxes-ntf3-3m-v1,4999,google
U3,mm-sq1-v1,2026-04-02 09:00:00,DE,2026-04-02 09:03:00,2026-04-02 09:30:00,2026-04-02 09:01:00,O302,2026-04-02 11:00:00,pkg-premium-gold-taxes-ntf3-v2,1999,google
U4,mf-sq1-v1,2026-04-01 12:00:00,UK,2026-04-01 12:10:00,2026-04-01 12:30:00,2026-04-01 12:05:00,O401,2026-04-01 12:30:00,pkg-premium-advanced-taxes-ntf3-6m-v1,9999,fb
U4,mf-sq1-v1,2026-04-01 12:00:00,UK,2026-04-01 12:10:00,2026-04-01 12:30:00,2026-04-01 12:05:00,O402,2026-04-03 12:00:00,credits-1550-taxes-v4,1499,fb
U4,mf-sq1-v1,2026-04-01 12:00:00,UK,2026-04-01 12:10:00,2026-04-01 12:30:00,2026-04-01 12:05:00,O403,2026-04-08 12:00:00,credits-2950-taxes-v4,2499,fb
U5,mm-sq1-v1,,US,2026-04-04 14:00:00,2026-04-04 14:10:00,2026-04-04 14:01:00,O501,2026-04-04 14:10:00,pkg-premium-standard-taxes-ntf3-1m-v7,2999,direct
"""

df = mod.load_land1(io.StringIO(csv))
print(f"  Loaded {len(df)} rows, {df['user_id'].nunique()} unique users")
print()
print(df[['user_id', 'order_id', 'product_cluster', 'plan_duration',
         'order_seq', 'is_upsell', 'time_to_purchase_min', 'tt_bucket']].to_string(index=False))
print()

# Assertions
# U1 — no order_id, no order_seq
u1 = df[df['user_id'] == 'U1'].iloc[0]
assert pd.isna(u1['order_seq']), f"U1 should have NaN order_seq, got {u1['order_seq']}"
assert u1['is_upsell'] == False, f"U1 should not be upsell"

# U2 — single order, seq=1, not upsell, ~15 min → '<30min'
u2 = df[df['user_id'] == 'U2'].iloc[0]
assert int(u2['order_seq']) == 1, f"U2 order_seq should be 1, got {u2['order_seq']}"
assert u2['is_upsell'] == False
assert u2['tt_bucket'] == '<30min', f"U2 should be '<30min', got {u2['tt_bucket']}"
assert u2['plan_duration'] == '1m'
assert u2['product_cluster'] == 'premium 1m'
assert abs(u2['amount'] - 29.99) < 0.01, f"U2 amount should be 29.99 (cents/100), got {u2['amount']}"

# U3 — 2 orders, sorted by order_created_at: O301 then O302
u3 = df[df['user_id'] == 'U3'].sort_values('order_created_at')
seqs = [int(s) for s in u3['order_seq']]
assert seqs == [1, 2], f"U3 order_seqs should be [1,2], got {seqs}"
upsells = list(u3['is_upsell'])
assert upsells == [False, True], f"U3 is_upsell should be [F,T], got {upsells}"
buckets = list(u3['tt_bucket'])
# O301: landing 09:00, order 09:30 → exactly 30 min → '30min–2h'
# O302: landing 09:00, order 11:00 → 120 min → 'D1–7' bucket starts at 24*60=1440? wait, 120 min < 24*60 → '2–24h'
assert buckets == ['30min–2h', '2–24h'], f"U3 tt_buckets should be ['30min–2h','2–24h'], got {buckets}"

# U4 — 3 orders, seqs [1,2,3]
u4 = df[df['user_id'] == 'U4'].sort_values('order_created_at')
seqs = [int(s) for s in u4['order_seq']]
assert seqs == [1, 2, 3], f"U4 order_seqs should be [1,2,3], got {seqs}"
upsells = list(u4['is_upsell'])
assert upsells == [False, True, True]
# O401: 30 min after landing → '30min–2h'
# O402: 2 days later → 'D1–7'
# O403: 7 days later → 'D8+' (>= 7 days)
buckets = list(u4['tt_bucket'])
assert buckets == ['30min–2h', 'D1–7', 'D8+'], f"U4 buckets unexpected: {buckets}"
# product_clusters
clusters = list(u4['product_cluster'])
assert clusters == ['premium 6m', 'credits', 'credits']

# U5 — order but landing_at NaT → tt_bucket should be NaN
u5 = df[df['user_id'] == 'U5'].iloc[0]
assert pd.isna(u5['tt_bucket']), f"U5 tt_bucket should be NaN, got {u5['tt_bucket']}"
assert int(u5['order_seq']) == 1

print("  ✓ order_seq, is_upsell, tt_bucket — all correct")

# ─────────────────────────────────────────────────────────────────────
# 3. Mimic Deep Dive aggregations (key calcs)
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. Deep Dive aggregations (per-user)")
print("=" * 60)

cohort_users = {'U2', 'U3', 'U4'}  # 3 payers
df_cohort_dd = df[df['user_id'].isin(cohort_users)].copy()

user_agg = (df_cohort_dd.groupby('user_id', as_index=False)
            .agg(reg_at=('reg_at', 'min'),
                 fo_at=('fo_at', 'min'),
                 landing_at=('landing_at', 'min'),
                 total_amount=('amount', 'sum'),
                 orders_count=('order_id', 'nunique'),
                 country=('country', 'first')))
print(user_agg.to_string(index=False))
print()

n_pay = user_agg['fo_at'].notna().sum()
n_upsell = int((user_agg['orders_count'] > 1).sum())
total_rev = user_agg['total_amount'].sum()
print(f"  Cohort size: {len(cohort_users)}")
print(f"  Payers: {n_pay}")
print(f"  Upsell users: {n_upsell}  (expected 2: U3, U4)")
print(f"  Total revenue: ${total_rev:.2f}  (expected: 29.99+49.99+19.99+99.99+14.99+24.99 = 239.94)")
print(f"  LTV/payer: ${total_rev/n_pay:.2f}")

assert n_pay == 3
assert n_upsell == 2, f"Expected 2 upsell users (U3,U4), got {n_upsell}"
assert abs(total_rev - 239.94) < 0.01, f"Expected total $239.94, got ${total_rev}"

# First orders (for subscription-mix block)
df_orders = df_cohort_dd[df_cohort_dd['order_id'].notna()].copy()
first_orders = (df_orders.sort_values('order_created_at')
                .groupby('user_id', as_index=False).first())
print()
print("  First orders:")
print(first_orders[['user_id', 'product_cluster', 'plan_duration', 'amount']].to_string(index=False))

# Top upsell SKUs (is_upsell == True)
df_upsell = df_cohort_dd[df_cohort_dd['is_upsell'] == True]
print()
print(f"  Upsell orders: {len(df_upsell)} (expected 3: U3:O302, U4:O402, U4:O403)")
assert len(df_upsell) == 3

# Country breakdown
geo = (user_agg.groupby('country').agg(users=('user_id', 'nunique')).reset_index())
print()
print("  Geography:")
print(geo.to_string(index=False))

print()
print("=" * 60)
print("ALL DEEP DIVE SMOKE TESTS PASSED ✓")
print("=" * 60)
