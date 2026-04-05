"""
Higgsfield Retention Architect — Final Solution
GPU-accelerated: LightGBM (gpu) + XGBoost (cuda)
Feature cache: first run builds parquet, next runs load instantly
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy import stats
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')

t0 = time.time()
TRAIN = "Train Data"
TEST  = "Test Data"
CACHE = Path("feature_cache")
CACHE.mkdir(exist_ok=True)

TARGET_MAP = {'not_churned': 0, 'vol_churn': 1, 'invol_churn': 2}
TARGET_INV = {v: k for k, v in TARGET_MAP.items()}

def safe_mode(x):
    m = x.dropna().mode()
    return m.iloc[0] if len(m) > 0 else 'unknown'

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def feat_generation(gen_df, props_df):
    print("    generations ...", end=" ", flush=True); t = time.time()
    gen_df   = gen_df.copy()
    props_df = props_df[['user_id','subscription_start_date']].copy()
    props_df['sub_start'] = pd.to_datetime(props_df['subscription_start_date'], errors='coerce')

    gen_df['created_at'] = pd.to_datetime(gen_df['created_at'], errors='coerce')
    gen_df = gen_df.merge(props_df[['user_id','sub_start']], on='user_id', how='left')
    gen_df['day_offset'] = (gen_df['created_at'] - gen_df['sub_start']).dt.days.clip(0, 13)
    gen_df['gen_date']   = gen_df['created_at'].dt.date

    g = gen_df.groupby('user_id')

    f = {}
    f['gen_total']          = g.size()
    f['gen_completed']      = g['status'].apply(lambda x: (x=='completed').sum())
    f['gen_failed']         = g['status'].apply(lambda x: (x=='failed').sum())
    f['gen_nsfw']           = g['status'].apply(lambda x: (x=='nsfw').sum())
    f['gen_canceled']       = g['status'].apply(lambda x: (x=='canceled').sum())
    f['gen_credit_total']   = g['credit_cost'].sum()
    f['gen_credit_mean']    = g['credit_cost'].mean()
    f['gen_credit_max']     = g['credit_cost'].max()
    f['gen_credit_std']     = g['credit_cost'].std()
    f['gen_days_active']    = g['gen_date'].nunique()
    f['gen_types_unique']   = g['generation_type'].nunique()
    f['gen_max_in_day']     = gen_df.groupby(['user_id','gen_date']).size().groupby('user_id').max()
    f['gen_span_hours']     = (g['created_at'].max() - g['created_at'].min()).dt.total_seconds() / 3600

    tot = f['gen_total'].clip(lower=1)
    f['gen_completion_rate'] = f['gen_completed'] / tot
    f['gen_fail_rate']       = f['gen_failed'] / tot
    f['gen_nsfw_rate']       = f['gen_nsfw'] / tot
    f['gen_cancel_rate']     = f['gen_canceled'] / tot

    gen_df['is_video'] = gen_df['generation_type'].str.startswith('video', na=False)
    gen_df['is_image'] = gen_df['generation_type'].str.startswith('image', na=False)
    f['gen_video_count'] = gen_df.groupby('user_id')['is_video'].sum()
    f['gen_image_count'] = gen_df.groupby('user_id')['is_image'].sum()
    f['gen_video_ratio'] = f['gen_video_count'] / tot
    f['gen_per_active_day'] = f['gen_total'] / f['gen_days_active'].clip(lower=1)

    f['gen_duration_mean'] = g['duration'].mean()
    f['gen_duration_max']  = g['duration'].max()
    f['gen_duration_sum']  = g['duration'].sum()
    f['gen_duration_std']  = g['duration'].std()

    # 14-day arc: 4 quartile buckets
    for lo, hi, name in [(0,3,'q1'),(4,6,'q2'),(7,10,'q3'),(11,13,'q4')]:
        mask = (gen_df['day_offset'] >= lo) & (gen_df['day_offset'] <= hi)
        f[f'gen_{name}_count']   = gen_df[mask].groupby('user_id').size()
        f[f'gen_{name}_credits'] = gen_df[mask].groupby('user_id')['credit_cost'].sum()
        f[f'gen_{name}_compl']   = gen_df[mask].groupby('user_id')['status'].apply(
                                       lambda x: (x=='completed').sum() / max(len(x),1))

    # Credit depletion velocity
    early_cr = gen_df[gen_df['day_offset'] <= 6].groupby('user_id')['credit_cost'].sum()
    late_cr  = gen_df[gen_df['day_offset'] >= 7].groupby('user_id')['credit_cost'].sum()
    f['gen_credit_early']     = early_cr
    f['gen_credit_late']      = late_cr
    f['gen_credit_vel_ratio'] = early_cr / (late_cr.clip(lower=0.1))
    f['gen_early_count']      = gen_df[gen_df['day_offset'] <= 6].groupby('user_id').size()
    f['gen_late_count']       = gen_df[gen_df['day_offset'] >= 7].groupby('user_id').size()

    # Activity slope (linear trend over 14 days)
    daily = gen_df.groupby(['user_id','day_offset']).size().reset_index(name='cnt')

    def slope(s):
        if len(s) < 2: return 0.0
        r = stats.linregress(s['day_offset'], s['cnt'])
        return float(r.slope)

    f['gen_activity_slope'] = daily.groupby('user_id').apply(slope)

    # Peak day
    peak = daily.loc[daily.groupby('user_id')['cnt'].idxmax(), ['user_id','day_offset']]
    f['gen_peak_day'] = peak.set_index('user_id')['day_offset']

    # Activity cliff (max single-day drop)
    def cliff(s):
        if len(s) < 2: return 0.0
        return float(np.diff(s.sort_values('day_offset')['cnt'].values.astype(float)).min())

    f['gen_activity_cliff'] = daily.groupby('user_id').apply(cliff)

    # Last 3 days activity
    last3 = gen_df[gen_df['day_offset'] >= 11].groupby('user_id').size()
    f['gen_last3_count'] = last3
    f['gen_last3_frac']  = last3 / f['gen_total'].clip(lower=1)

    # First generation latency (hours after subscription start)
    f['gen_first_latency_h'] = (g['created_at'].min() - gen_df.groupby('user_id')['sub_start'].first()
                                ).dt.total_seconds() / 3600

    # Completion trend: early completion rate vs late
    early_compl = gen_df[gen_df['day_offset'] <= 6].groupby('user_id')['status'].apply(
        lambda x: (x=='completed').sum() / max(len(x),1))
    late_compl  = gen_df[gen_df['day_offset'] >= 7].groupby('user_id')['status'].apply(
        lambda x: (x=='completed').sum() / max(len(x),1))
    f['gen_compl_trend'] = late_compl - early_compl   # positive = improving, negative = degrading

    # Expensive model usage (video = expensive)
    # ratio of credits spent on video
    vid_credits = gen_df[gen_df['is_video']].groupby('user_id')['credit_cost'].sum()
    f['gen_video_credit_ratio'] = vid_credits / f['gen_credit_total'].clip(lower=0.1)

    res = pd.DataFrame(f).reset_index()
    print(f"{time.time()-t:.0f}s")
    return res


def feat_transactions(ta_df):
    print("    transactions ...", end=" ", flush=True); t = time.time()
    ta_df = ta_df.copy()
    ta_df['transaction_time'] = pd.to_datetime(ta_df['transaction_time'], errors='coerce')
    ta_df = ta_df.sort_values(['user_id','transaction_time'])

    ta_df['is_failed']  = ta_df['failure_code'].notna()
    ta_df['fail_decl']  = ta_df['failure_code'] == 'card_declined'
    ta_df['fail_cvc']   = ta_df['failure_code'].isin(['incorrect_cvc','invalid_cvc'])
    ta_df['fail_exp']   = ta_df['failure_code'] == 'expired_card'
    ta_df['fail_auth']  = ta_df['failure_code'] == 'authentication_required'

    for col in ['is_prepaid','is_virtual','is_business']:
        ta_df[col] = ta_df[col].map({True:1,False:0,'True':1,'False':0}).fillna(0)
    ta_df['is_3ds']      = ta_df['is_3d_secure'].astype(int)
    ta_df['is_3ds_auth'] = ta_df['is_3d_secure_authenticated'].astype(int)
    ta_df['ctry_mismatch'] = (
        ta_df['card_country'].fillna('') != ta_df['billing_address_country'].fillna('')
    ).astype(int)

    g = ta_df.groupby('user_id')
    f = {}
    f['ta_total']         = g.size()
    f['ta_failed_count']  = g['is_failed'].sum()
    f['ta_success_count'] = (~ta_df['is_failed']).groupby(ta_df['user_id']).sum()
    f['ta_fail_rate']     = g['is_failed'].mean()
    f['ta_card_declined'] = g['fail_decl'].sum()
    f['ta_cvc_fail']      = g['fail_cvc'].sum()
    f['ta_expired']       = g['fail_exp'].sum()
    f['ta_auth_req']      = g['fail_auth'].sum()
    f['ta_is_prepaid']    = g['is_prepaid'].max()
    f['ta_is_virtual']    = g['is_virtual'].max()
    f['ta_is_business']   = g['is_business'].max()
    f['ta_is_3ds']        = g['is_3ds'].max()
    f['ta_3ds_auth']      = g['is_3ds_auth'].max()
    f['ta_amount_total']  = g['amount_in_usd'].sum()
    f['ta_amount_mean']   = g['amount_in_usd'].mean()
    f['ta_amount_max']    = g['amount_in_usd'].max()
    f['ta_last_failed']   = g.last()['is_failed'].astype(float)
    f['ta_ctry_mismatch'] = g['ctry_mismatch'].max()

    def max_consec(x):
        s = b = 0
        for v in x:
            s = s+1 if v else 0; b = max(b,s)
        return b
    f['ta_max_consec_fail'] = g['is_failed'].apply(max_consec)

    ta_df['amt_r'] = ta_df['amount_in_usd'].round(2)
    f['ta_max_retries'] = ta_df.groupby(['user_id','amt_r']).size().groupby('user_id').max()
    f['ta_cvc_pass']    = g['cvc_check'].apply(lambda x: int((x=='pass').any()))

    f['ta_card_funding']   = g['card_funding'].agg(safe_mode)
    f['ta_card_brand']     = g['card_brand'].agg(safe_mode)
    f['ta_payment_method'] = g['payment_method_type'].agg(safe_mode)
    f['ta_bank_country']   = g['bank_country'].agg(safe_mode)
    f['ta_card_country']   = g['card_country'].agg(safe_mode)

    res = pd.DataFrame(f).reset_index()
    print(f"{time.time()-t:.0f}s")
    return res


def feat_purchases(df):
    print("    purchases ...", end=" ", flush=True); t = time.time()
    g = df.groupby('user_id')
    f = {}
    f['purch_count']       = g.size()
    f['purch_total']       = g['purchase_amount_dollars'].sum()
    f['purch_mean']        = g['purchase_amount_dollars'].mean()
    f['purch_types']       = g['purchase_type'].nunique()
    df['is_cp'] = df['purchase_type'].str.contains('credit', case=False, na=False)
    f['purch_credit_packs'] = df.groupby('user_id')['is_cp'].sum()
    res = pd.DataFrame(f).reset_index()
    print(f"{time.time()-t:.0f}s")
    return res


def feat_quiz(df):
    print("    quiz ...", end=" ", flush=True); t = time.time()
    df = df.drop_duplicates('user_id', keep='first').copy()
    df['qf_cost']    = df['frustration'].isin(['High cost of top models','high-cost']).astype(int)
    df['qf_incon']   = df['frustration'].isin(['inconsistent','Inconsistent results']).astype(int)
    df['qf_limited'] = df['frustration'].isin(['limited','Limited generations']).astype(int)
    df['qf_prompt']  = df['frustration'].isin(['hard-prompt','Hard to prompt']).astype(int)
    df['qf_other']   = df['frustration'].isin(['other','Other']).astype(int)
    cats = ['source','flow_type','team_size','experience','usage_plan','frustration','first_feature','role']
    df = df.rename(columns={c: f'q_{c}' for c in cats})
    cols = ['user_id'] + [f'q_{c}' for c in cats] + ['qf_cost','qf_incon','qf_limited','qf_prompt','qf_other']
    res = df[cols]
    print(f"{time.time()-t:.0f}s")
    return res


def feat_properties(df):
    print("    properties ...", end=" ", flush=True); t = time.time()
    df = df.copy()
    plan_tier = {'Higgsfield Basic':1,'Higgsfield Creator':2,'Higgsfield Pro':3,'Higgsfield Ultimate':4}
    df['prop_tier']   = df['subscription_plan'].map(plan_tier).fillna(0)
    df['sub_dt'] = pd.to_datetime(df['subscription_start_date'], errors='coerce')
    df['prop_dow']    = df['sub_dt'].dt.dayofweek
    df['prop_month']  = df['sub_dt'].dt.month
    df['prop_day']    = df['sub_dt'].dt.day
    cols = ['user_id','subscription_plan','country_code','prop_tier','prop_dow','prop_month','prop_day']
    res = df[cols].rename(columns={'subscription_plan':'prop_plan','country_code':'prop_country'})
    print(f"{time.time()-t:.0f}s")
    return res


def build_features(users_df, gen_df, props_df, purch_df, quiz_df, ta_df, label_col='churn_status'):
    df = users_df[['user_id']].copy()
    if label_col in users_df.columns:
        df[label_col] = users_df[label_col]

    df = df.merge(feat_properties(props_df), on='user_id', how='left')
    df = df.merge(feat_generation(gen_df, props_df), on='user_id', how='left')
    df = df.merge(feat_purchases(purch_df), on='user_id', how='left')
    df = df.merge(feat_quiz(quiz_df), on='user_id', how='left')
    df = df.merge(feat_transactions(ta_df), on='user_id', how='left')

    df['has_gen']   = df['gen_total'].notna().astype(int)
    df['has_tx']    = df['ta_total'].notna().astype(int)
    df['has_purch'] = df['purch_count'].notna().astype(int)

    zero_cols = [
        'gen_total','gen_completed','gen_failed','gen_nsfw','gen_canceled',
        'gen_credit_total','gen_video_count','gen_image_count','gen_days_active',
        'gen_early_count','gen_late_count','gen_completion_rate','gen_fail_rate',
        'gen_nsfw_rate','gen_cancel_rate','gen_video_ratio','gen_per_active_day','gen_max_in_day',
        'gen_q1_count','gen_q2_count','gen_q3_count','gen_q4_count',
        'gen_q1_credits','gen_q2_credits','gen_q3_credits','gen_q4_credits',
        'gen_q1_compl','gen_q2_compl','gen_q3_compl','gen_q4_compl',
        'gen_credit_early','gen_credit_late','gen_last3_count','gen_last3_frac',
        'ta_total','ta_failed_count','ta_success_count','ta_fail_rate',
        'ta_card_declined','ta_cvc_fail','ta_expired','ta_max_retries',
        'ta_max_consec_fail','ta_is_prepaid','ta_is_virtual','ta_ctry_mismatch',
        'purch_count','purch_total','purch_credit_packs',
        'qf_cost','qf_incon','qf_limited','qf_prompt','qf_other',
    ]
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Composite signals
    df['invol_risk'] = (
        df['ta_is_prepaid'].astype(float) * 3 +
        df['ta_fail_rate'].astype(float)  * 4 +
        df['ta_card_declined'].astype(float) * 1.5 +
        (df['ta_card_funding'] == 'prepaid').astype(float) * 3 +
        df['ta_last_failed'].fillna(0).astype(float) * 2 +
        df['ta_max_consec_fail'].astype(float) +
        df['ta_ctry_mismatch'].astype(float) * 1.5
    ).astype(float)

    df['vol_risk'] = (
        (1 - df['gen_completion_rate']) +
        (df['gen_total'] < 3).astype(int) * 3 +
        df['qf_cost'] * 2 + df['qf_incon'] * 2 + df['qf_limited'] * 1.5 +
        (df['gen_activity_slope'].fillna(0) < -1).astype(float) * 2 +
        (df['gen_peak_day'].fillna(7) < 3).astype(float) * 1.5
    ).astype(float)

    df['engagement'] = (
        np.log1p(df['gen_total']) * 2 +
        df['gen_days_active'] +
        np.log1p(df['gen_credit_total']) +
        df['purch_count'] * 0.5
    ).astype(float)

    # Interaction: plan tier × engagement
    df['plan_x_engagement'] = df['prop_tier'] * df['engagement']

    # Arc shape: front-heavy vs back-heavy
    df['gen_front_ratio'] = (df['gen_q1_count'] + df['gen_q2_count']) / df['gen_total'].clip(lower=1)
    df['gen_back_ratio']  = (df['gen_q3_count'] + df['gen_q4_count']) / df['gen_total'].clip(lower=1)

    return df


# ══════════════════════════════════════════════════════════════
# LOAD OR BUILD FEATURES (with cache)
# ══════════════════════════════════════════════════════════════
def get_features(split):
    """Load from cache if available, otherwise build and cache."""
    cache_file = CACHE / f"{split}_features.parquet"
    if cache_file.exists():
        print(f"  Loading {split} features from cache...")
        return pd.read_parquet(cache_file)

    print(f"  Building {split} features...")
    prefix = "train_users" if split == "train" else "test_users"
    folder = TRAIN if split == "train" else TEST

    users  = pd.read_csv(f"{folder}/{prefix}.csv")
    print("  Loading generations (large)...", end=" ", flush=True)
    t = time.time()
    gen    = pd.read_csv(f"{folder}/{prefix}_generations.csv",              low_memory=False)
    print(f"{time.time()-t:.0f}s")
    props  = pd.read_csv(f"{folder}/{prefix}_properties.csv")
    purch  = pd.read_csv(f"{folder}/{prefix}_purchases.csv")
    quiz   = pd.read_csv(f"{folder}/{prefix}_quizzes.csv",                  low_memory=False)
    ta     = pd.read_csv(f"{folder}/{prefix}_transaction_attempts_v1.csv",  low_memory=False)

    df = build_features(users, gen, props, purch, quiz, ta)
    df.to_parquet(cache_file, index=False)
    print(f"  Cached to {cache_file}")
    return df


print("=" * 60)
print("LOADING / BUILDING FEATURES...")
print("=" * 60)
train_df = get_features("train")
test_df  = get_features("test")
print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

# ══════════════════════════════════════════════════════════════
# TARGET ENCODING (leakage-free, CV-based)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TARGET ENCODING...")
print("=" * 60)

train_df['y']         = train_df['churn_status'].map(TARGET_MAP)
train_df['is_churned']= (train_df['y'] > 0).astype(int)
train_df['is_vol']    = (train_df['y'] == 1).astype(int)
train_df['is_invol']  = (train_df['y'] == 2).astype(int)

# More columns = more signal
TE_COLS = ['prop_country','prop_plan','q_source','q_frustration','q_role','q_usage_plan','q_first_feature']
TE_TGTS = ['is_churned','is_vol','is_invol']
skf_te  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

te_names = [f'te_{col}_{tgt}' for col in TE_COLS for tgt in TE_TGTS]
for n in te_names:
    train_df[n] = np.nan

global_means = {col: {tgt: train_df.groupby(col)[tgt].mean() for tgt in TE_TGTS} for col in TE_COLS}

for _, (tr_i, val_i) in enumerate(skf_te.split(train_df, train_df['y'])):
    for col in TE_COLS:
        for tgt in TE_TGTS:
            mean_map = train_df.iloc[tr_i].groupby(col)[tgt].mean()
            g_mean   = train_df[tgt].mean()
            name     = f'te_{col}_{tgt}'
            train_df.iloc[val_i, train_df.columns.get_loc(name)] = (
                train_df.iloc[val_i][col].map(mean_map).fillna(g_mean).values
            )

for col in TE_COLS:
    for tgt in TE_TGTS:
        name = f'te_{col}_{tgt}'
        g_mean = train_df[tgt].mean()
        test_df[name] = test_df[col].map(global_means[col][tgt]).fillna(g_mean)

print(f"  Created {len(te_names)} target-encoded features")

# ── Label encode remaining categoricals ──
CAT_COLS = [
    'prop_plan','prop_country',
    'q_source','q_flow_type','q_team_size','q_experience',
    'q_usage_plan','q_frustration','q_first_feature','q_role',
    'ta_card_funding','ta_card_brand','ta_payment_method',
    'ta_bank_country','ta_card_country'
]
for col in CAT_COLS:
    for df_ in [train_df, test_df]:
        if col in df_.columns:
            df_[col] = df_[col].fillna('unknown').astype(str)

for col in CAT_COLS:
    if col in train_df.columns:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]).unique())
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

# ══════════════════════════════════════════════════════════════
# FEATURE MATRIX
# ══════════════════════════════════════════════════════════════
EXCL = ['user_id','churn_status','y','is_churned','is_vol','is_invol']
FEAT = [c for c in train_df.columns if c not in EXCL]

X     = train_df[FEAT].fillna(-1).astype(np.float32)
y     = train_df['y']
X_tst = test_df[FEAT].fillna(-1).astype(np.float32)

print(f"\nFeatures: {len(FEAT)} | Train: {len(X):,} | Test: {len(X_tst):,}")
print(f"Class distribution:\n{y.value_counts().rename(TARGET_INV)}")

# ══════════════════════════════════════════════════════════════
# MODEL PARAMS (GPU)
# ══════════════════════════════════════════════════════════════
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

LGBM_S1 = dict(
    objective='binary', metric='binary_logloss',
    n_estimators=5000, learning_rate=0.02, num_leaves=255,
    min_child_samples=15, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=0.1,
    is_unbalance=True, device='gpu', random_state=42, n_jobs=-1, verbose=-1
)
LGBM_S2 = dict(
    objective='binary', metric='binary_logloss',
    n_estimators=5000, learning_rate=0.02, num_leaves=127,
    min_child_samples=15, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=0.1,
    device='gpu', random_state=42, n_jobs=-1, verbose=-1
)
LGBM_MC = dict(
    objective='multiclass', num_class=3, metric='multi_logloss',
    n_estimators=5000, learning_rate=0.02, num_leaves=127,
    min_child_samples=15, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=0.1,
    class_weight='balanced', device='gpu', random_state=42, n_jobs=-1, verbose=-1
)

# ══════════════════════════════════════════════════════════════
# STAGE 1: churn vs not-churn
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 1: Churn vs Not-Churn  (GPU LightGBM)")
print("=" * 60)
y_bin = (y > 0).astype(int)
s1_oof, s1_tst = np.zeros(len(X)), np.zeros(len(X_tst))

for fold, (tr, val) in enumerate(SKF.split(X, y_bin), 1):
    print(f"  Fold {fold}/5 ...", end=" ", flush=True); tf = time.time()
    m = lgb.LGBMClassifier(**LGBM_S1)
    m.fit(X.iloc[tr], y_bin.iloc[tr], eval_set=[(X.iloc[val], y_bin.iloc[val])],
          callbacks=[lgb.early_stopping(150,verbose=False), lgb.log_evaluation(9999)])
    s1_oof[val] = m.predict_proba(X.iloc[val])[:,1]
    s1_tst += m.predict_proba(X_tst)[:,1] / 5
    print(f"iter={m.best_iteration_}  F1={f1_score(y_bin.iloc[val],(s1_oof[val]>0.5).astype(int)):.4f}  {time.time()-tf:.0f}s")

# ══════════════════════════════════════════════════════════════
# STAGE 2: vol vs invol (churned users only)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2: Vol vs Invol  (GPU LightGBM)")
print("=" * 60)
churn_mask  = y_bin == 1
X_ch = X[churn_mask].reset_index(drop=True)
y_vi = (y[churn_mask] == 2).astype(int).reset_index(drop=True)  # 1=invol

s2_oof, s2_tst = np.zeros(len(X_ch)), np.zeros(len(X_tst))
SKF2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr, val) in enumerate(SKF2.split(X_ch, y_vi), 1):
    print(f"  Fold {fold}/5 ...", end=" ", flush=True); tf = time.time()
    m2 = lgb.LGBMClassifier(**LGBM_S2)
    m2.fit(X_ch.iloc[tr], y_vi.iloc[tr], eval_set=[(X_ch.iloc[val], y_vi.iloc[val])],
           callbacks=[lgb.early_stopping(150,verbose=False), lgb.log_evaluation(9999)])
    s2_oof[val] = m2.predict_proba(X_ch.iloc[val])[:,1]
    s2_tst += m2.predict_proba(X_tst)[:,1] / 5
    print(f"iter={m2.best_iteration_}  F1={f1_score(y_vi.iloc[val],(s2_oof[val]>0.5).astype(int)):.4f}  {time.time()-tf:.0f}s")

# ══════════════════════════════════════════════════════════════
# MULTICLASS: LightGBM
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MULTICLASS: LightGBM  (GPU)")
print("=" * 60)
lgbm_oof, lgbm_tst = np.zeros((len(X),3)), np.zeros((len(X_tst),3))
lgbm_fi = np.zeros(len(FEAT))

for fold, (tr, val) in enumerate(SKF.split(X, y), 1):
    print(f"  Fold {fold}/5 ...", end=" ", flush=True); tf = time.time()
    mc = lgb.LGBMClassifier(**LGBM_MC)
    mc.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[val], y.iloc[val])],
           callbacks=[lgb.early_stopping(150,verbose=False), lgb.log_evaluation(9999)])
    lgbm_oof[val] = mc.predict_proba(X.iloc[val])
    lgbm_tst += mc.predict_proba(X_tst) / 5
    lgbm_fi += mc.feature_importances_ / 5
    print(f"iter={mc.best_iteration_}  F1={f1_score(y.iloc[val],np.argmax(lgbm_oof[val],1),average='weighted'):.4f}  {time.time()-tf:.0f}s")

# ══════════════════════════════════════════════════════════════
# COMBINE: cascade + lgbm
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE...")
print("=" * 60)

lgbm_f1 = f1_score(y, np.argmax(lgbm_oof,1), average='weighted')
print(f"  LightGBM multiclass OOF F1: {lgbm_f1:.4f}")

def cascade_proba(p_churn, p_invol):
    p_not    = 1 - p_churn
    p_invol_ = p_churn * p_invol
    p_vol    = p_churn * (1 - p_invol)
    return np.stack([p_not, p_vol, p_invol_], axis=1)

# Stage 1 threshold sweep — find optimal churn/no-churn cutoff
print("  Sweeping Stage 1 threshold...")
best_thr_f1, best_thr = 0, 0.5
for thr in np.arange(0.20, 0.65, 0.02):
    s2_full = np.full(len(X), 0.5)
    s2_full[churn_mask.values] = s2_oof
    # Re-build cascade with this threshold applied to Stage 1
    p_churn_adj = (s1_oof >= thr).astype(float)  # hard threshold
    # Soft version: scale probabilities so threshold = 0.5
    p_churn_soft = s1_oof / (thr * 2)  # rescale so thr maps to 0.5
    p_churn_soft = np.clip(p_churn_soft, 0, 1)
    casc = cascade_proba(p_churn_soft, s2_full)
    # blend 80/20 with lgbm
    blend = 0.8 * casc + 0.2 * lgbm_oof
    f1 = f1_score(y, np.argmax(blend, 1), average='weighted')
    if f1 > best_thr_f1:
        best_thr_f1, best_thr = f1, thr

print(f"  Best Stage1 threshold={best_thr:.2f}  F1={best_thr_f1:.4f}")

# Rebuild cascade with best threshold
s2_full = np.full(len(X), 0.5)
s2_full[churn_mask.values] = s2_oof
p_churn_soft_oof = np.clip(s1_oof / (best_thr * 2), 0, 1)
p_churn_soft_tst = np.clip(s1_tst / (best_thr * 2), 0, 1)
casc_oof = cascade_proba(p_churn_soft_oof, s2_full)
casc_tst = cascade_proba(p_churn_soft_tst, s2_tst)

# Blend sweep: cascade vs lgbm
print("  Sweeping blend alpha...")
best_f1_blend, best_a = 0, 0.8
for a in np.arange(0.0, 1.01, 0.05):
    blend = a * casc_oof + (1 - a) * lgbm_oof
    f1 = f1_score(y, np.argmax(blend, 1), average='weighted')
    if f1 > best_f1_blend:
        best_f1_blend, best_a = f1, a

print(f"  Best alpha={best_a:.2f}  F1={best_f1_blend:.4f}")

ens_oof = best_a * casc_oof + (1 - best_a) * lgbm_oof
ens_tst = best_a * casc_tst + (1 - best_a) * lgbm_tst

# Vol scale sweep
best_vf, best_vs = best_f1_blend, 1.0
for vs in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
    sc = ens_oof.copy(); sc[:,1] *= vs
    sc /= sc.sum(axis=1, keepdims=True)
    f1 = f1_score(y, np.argmax(sc,1), average='weighted')
    if f1 > best_vf: best_vf, best_vs = f1, vs

if best_vs != 1.0:
    for arr in [ens_oof, ens_tst]:
        arr[:,1] *= best_vs
        arr /= arr.sum(axis=1, keepdims=True)
    print(f"  Vol scale={best_vs:.1f}  F1={best_vf:.4f}")

final_f1 = f1_score(y, np.argmax(ens_oof,1), average='weighted')
print(f"\n  Final OOF Weighted F1: {final_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y, np.argmax(ens_oof,1), target_names=['not_churned','vol_churn','invol_churn']))
print("Confusion Matrix:")
cm = confusion_matrix(y, np.argmax(ens_oof,1))
print(pd.DataFrame(cm, index=['not_churned','vol_churn','invol_churn'],
                       columns=['pred_not','pred_vol','pred_invol']))

# ══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
fi = pd.DataFrame({'feature': FEAT, 'importance': lgbm_fi}).sort_values('importance', ascending=False)
fi.to_csv('feature_importance.csv', index=False)
print("\nTop 20 features:")
print(fi.head(20).to_string(index=False))

# ══════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════
test_classes = [TARGET_INV[p] for p in np.argmax(ens_tst, 1)]
sub = pd.DataFrame({'user_id': test_df['user_id'], 'churn_status': test_classes})

print(f"\nTest distribution:\n{sub['churn_status'].value_counts()}")
sub.to_csv('RetentionArchitects_submission.csv', index=False)

proba = pd.DataFrame(ens_tst, columns=['prob_not_churned','prob_vol_churn','prob_invol_churn'])
proba['user_id']      = test_df['user_id'].values
proba['churn_status'] = test_classes
proba.to_csv('test_predictions_proba.csv', index=False)

print(f"\nSubmission saved: RetentionArchitects_submission.csv")
print(f"Total time: {time.time()-t0:.0f}s")
