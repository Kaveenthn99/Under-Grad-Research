import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, log_loss)
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import mutual_info_classif
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = f"{PATH}/train_processed.csv"
TEST_FILE = f"{PATH}/test_processed.csv"
NEGATIVES_FILE = f"{PATH}/negativecontrols_processed.csv"

RESULTS_DIR = f"{PATH}/results"
PLOTS_DIR = f"{PATH}/plots"

MAX_ITERATIONS = 500
CONVERGENCE_THRESHOLD = 1e-6
LEARNING_RATE = 0.05
INITIAL_WEIGHT = 1.0 / 3
TOP_CANDIDATES = 50
MIN_WEIGHT = 0.10
CLIP_EPS = 1e-15

SEED = 42
np.random.seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# SAFE LOG LOSS
# ============================================================
def safe_log_loss(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, CLIP_EPS, 1 - CLIP_EPS)
    return log_loss(y_true, y_pred_clipped)

# ============================================================
# RELIEF-F
# ============================================================
def relief_f(X, y, n_iterations=500):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    classes = np.unique(y)
    for _ in range(n_iterations):
        idx = np.random.randint(n_samples)
        sample = X[idx]
        sample_class = y[idx]
        same_class_mask = y == sample_class
        same_class_mask[idx] = False
        if same_class_mask.sum() == 0:
            continue
        same_class_dists = np.linalg.norm(X[same_class_mask] - sample, axis=1)
        nearest_hit = X[same_class_mask][np.argmin(same_class_dists)]
        for other_class in classes:
            if other_class == sample_class:
                continue
            other_mask = y == other_class
            other_dists = np.linalg.norm(X[other_mask] - sample, axis=1)
            nearest_miss = X[other_mask][np.argmin(other_dists)]
            prior_other = other_mask.sum() / n_samples
            weights += prior_other * (np.abs(sample - nearest_miss) - np.abs(sample - nearest_hit))
    weights = weights / n_iterations
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
    return weights

# ============================================================
# MUTUAL INFORMATION
# ============================================================
def mi_feature_selection(X, y, threshold=0.3):
    mi_scores = mutual_info_classif(X, y, random_state=SEED, n_neighbors=2)
    mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
    selected = np.where(mi_scores_norm >= threshold)[0]
    if len(selected) < 3:
        selected = np.argsort(mi_scores_norm)[::-1][:3]
    return sorted(selected), mi_scores_norm

# ============================================================
# STEP 0 — Load Data
# ============================================================
print("=" * 70)
print("PHASE 2 — 3-MODEL INDEPENDENT ENSEMBLE")
print("Random Forest + Gaussian Process + SVM")
print("Optimised with LOG LOSS (Binary Cross-Entropy)")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
negatives = pd.read_csv(NEGATIVES_FILE)

metadata_cols = ['ID']
descriptor_cols = [col for col in train.columns if col not in metadata_cols]

print("\n=== NaN Check ===")
train_nan_count = train[descriptor_cols].isna().any(axis=1).sum()
train_nan_ids = train.loc[train[descriptor_cols].isna().any(axis=1), 'ID'].tolist()
print(f"Training NaN: {train_nan_count}/{len(train)}")
train = train.dropna(subset=descriptor_cols).reset_index(drop=True)

test_nan_count = test[descriptor_cols].isna().any(axis=1).sum()
test_nan_ids = test.loc[test[descriptor_cols].isna().any(axis=1), 'ID'].tolist()
print(f"Test NaN: {test_nan_count}/{len(test)}")
test = test.dropna(subset=descriptor_cols).reset_index(drop=True)

neg_nan_count = negatives[descriptor_cols].isna().any(axis=1).sum()
neg_nan_ids = negatives.loc[negatives[descriptor_cols].isna().any(axis=1), 'ID'].tolist()
print(f"Negative NaN: {neg_nan_count}/{len(negatives)}")
negatives = negatives.dropna(subset=descriptor_cols).reset_index(drop=True)

print(f"\nAfter cleanup: Train={len(train)}, Test={len(test)}, Neg={len(negatives)}")

dropped_df = pd.DataFrame({
    'ID': train_nan_ids + test_nan_ids + neg_nan_ids,
    'Source': ['train'] * len(train_nan_ids) + ['test'] * len(test_nan_ids) + ['neg'] * len(neg_nan_ids)
})
dropped_df.to_csv(f"{RESULTS_DIR}/dropped_nan_ids.csv", index=False)

train['label'] = 1
negatives['label'] = 0
train_combined = pd.concat([train, negatives], ignore_index=True)

X_train = train_combined[descriptor_cols].values
y_train = train_combined['label'].values
X_test = test[descriptor_cols].values
X_neg = negatives[descriptor_cols].values
active_mask = y_train == 1

print(f"\nTraining: {X_train.shape[0]} ({active_mask.sum()} actives, {(~active_mask).sum()} neg)")
print(f"Test: {X_test.shape[0]}, Neg controls: {X_neg.shape[0]}")
print(f"Descriptors ({len(descriptor_cols)}): {descriptor_cols}")
assert not np.isnan(X_train).any() and not np.isnan(X_test).any() and not np.isnan(X_neg).any()

# ============================================================
# STEP 1 — AUTO FEATURE SELECTION
# ============================================================
print("\n=== Step 1: Automatic Feature Selection ===")

model_names = ['RandomForest', 'GP', 'SVM']

# Model 1: RF — Built-in importance
print("\n  [RF] Feature importance...")
rf_init = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=2,
                                  class_weight='balanced', random_state=SEED)
rf_init.fit(X_train, y_train)
rf_imp = rf_init.feature_importances_
rf_imp_norm = rf_imp / (rf_imp.max() + 1e-10)
rf_sel_idx = np.where(rf_imp_norm >= 0.15)[0].tolist()
if len(rf_sel_idx) < 4:
    rf_sel_idx = np.argsort(rf_imp_norm)[::-1][:5].tolist()
rf_sel_names = [descriptor_cols[i] for i in rf_sel_idx]
print(f"    Selected ({len(rf_sel_idx)}): {rf_sel_names}")

# Model 2: GP — Relief-F
print("\n  [GP] Relief-F...")
relief_weights = relief_f(X_train, y_train, n_iterations=500)
gp_sel_idx = np.where(relief_weights >= 0.25)[0].tolist()
if len(gp_sel_idx) < 4:
    gp_sel_idx = np.argsort(relief_weights)[::-1][:5].tolist()
gp_sel_names = [descriptor_cols[i] for i in gp_sel_idx]
print(f"    Selected ({len(gp_sel_idx)}): {gp_sel_names}")

# Model 3: SVM — Mutual Information
print("\n  [SVM] Mutual Information...")
svm_sel_idx, mi_scores = mi_feature_selection(X_train, y_train, threshold=0.2)
svm_sel_names = [descriptor_cols[i] for i in svm_sel_idx]
print(f"    Selected ({len(svm_sel_idx)}): {svm_sel_names}")

model_selected_indices = [rf_sel_idx, gp_sel_idx, svm_sel_idx]
model_selected_names_list = [rf_sel_names, gp_sel_names, svm_sel_names]

# ============================================================
# STEP 1b — DESCRIPTOR COVERAGE ENFORCEMENT
# ============================================================
print("\n=== Step 1b: Descriptor Coverage ===")

all_used = set()
for sel in model_selected_indices:
    all_used.update(sel)
unused = [i for i in range(len(descriptor_cols)) if i not in all_used]

if unused:
    print(f"  Unused ({len(unused)}): {[descriptor_cols[u] for u in unused]}")
    loo_cov = LeaveOneOut()
    gp_kernel_cov = ConstantKernel(1.0) * RBF(length_scale=1.0)

    for ud in unused:
        best_m = None
        best_imp = -np.inf
        fc_cur = [len(s) for s in model_selected_indices]
        max_fc = max(fc_cur)

        for m_idx in range(3):
            cand = model_selected_indices[m_idx] + [ud]
            lw = np.zeros(X_train.shape[0])
            lwo = np.zeros(X_train.shape[0])

            for ti, vi in loo_cov.split(X_train):
                Xt, Xv = X_train[ti], X_train[vi]
                yt = y_train[ti]

                if m_idx == 0:
                    m_w = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=2,
                                                  class_weight='balanced', random_state=SEED)
                    m_w.fit(Xt[:, cand], yt); lw[vi] = m_w.predict_proba(Xv[:, cand])[:, 1]
                    m_wo = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=2,
                                                   class_weight='balanced', random_state=SEED)
                    m_wo.fit(Xt[:, model_selected_indices[m_idx]], yt)
                    lwo[vi] = m_wo.predict_proba(Xv[:, model_selected_indices[m_idx]])[:, 1]
                elif m_idx == 1:
                    m_w = GaussianProcessClassifier(kernel=gp_kernel_cov, random_state=SEED, max_iter_predict=200)
                    m_w.fit(Xt[:, cand], yt); lw[vi] = m_w.predict_proba(Xv[:, cand])[:, 1]
                    m_wo = GaussianProcessClassifier(kernel=gp_kernel_cov, random_state=SEED, max_iter_predict=200)
                    m_wo.fit(Xt[:, model_selected_indices[m_idx]], yt)
                    lwo[vi] = m_wo.predict_proba(Xv[:, model_selected_indices[m_idx]])[:, 1]
                elif m_idx == 2:
                    m_w = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
                    m_w.fit(Xt[:, cand], yt); lw[vi] = m_w.predict_proba(Xv[:, cand])[:, 1]
                    m_wo = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
                    m_wo.fit(Xt[:, model_selected_indices[m_idx]], yt)
                    lwo[vi] = m_wo.predict_proba(Xv[:, model_selected_indices[m_idx]])[:, 1]

            ll_w = safe_log_loss(y_train, lw)
            ll_wo = safe_log_loss(y_train, lwo)
            imp = ll_wo - ll_w
            penalty = len(model_selected_indices[m_idx]) / max_fc
            adj = imp + 0.01 * (1 - penalty)
            if adj > best_imp:
                best_imp = adj
                best_m = m_idx

        model_selected_indices[best_m].append(ud)
        model_selected_names_list[best_m].append(descriptor_cols[ud])
        print(f"    {descriptor_cols[ud]:40s} -> {model_names[best_m]}")
else:
    print("  All descriptors covered.")

all_final = set()
for s in model_selected_indices:
    all_final.update(s)
assert len(all_final) == len(descriptor_cols)
print(f"\n  VERIFIED: All {len(descriptor_cols)} descriptors covered")

print("\n  Final:")
for mn, sn in zip(model_names, model_selected_names_list):
    print(f"    {mn:15s} ({len(sn)}): {sn}")

feature_selection_methods = {
    'RandomForest': {'method': 'RF Feature Importance', 'scores': dict(zip(descriptor_cols, np.round(rf_imp_norm, 4)))},
    'GP': {'method': 'Relief-F', 'scores': dict(zip(descriptor_cols, np.round(relief_weights, 4)))},
    'SVM': {'method': 'Mutual Information', 'scores': dict(zip(descriptor_cols, np.round(mi_scores, 4)))}
}

rf_sel_idx = model_selected_indices[0]
gp_sel_idx = model_selected_indices[1]
svm_sel_idx = model_selected_indices[2]

# ============================================================
# STEP 2 — Train Models
# ============================================================
print("\n=== Step 2: Training Models ===")

rf_model = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=2,
                                   class_weight='balanced', random_state=SEED)
rf_model.fit(X_train[:, rf_sel_idx], y_train)
rf_train = rf_model.predict_proba(X_train[:, rf_sel_idx])[:, 1]
print(f"  [1/3] RF — LogLoss: {safe_log_loss(y_train, rf_train):.6f}")

gp_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp_model = GaussianProcessClassifier(kernel=gp_kernel, random_state=SEED, max_iter_predict=200)
gp_model.fit(X_train[:, gp_sel_idx], y_train)
gp_train = gp_model.predict_proba(X_train[:, gp_sel_idx])[:, 1]
print(f"  [2/3] GP — LogLoss: {safe_log_loss(y_train, gp_train):.6f}")

svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=SEED)
svm_model.fit(X_train[:, svm_sel_idx], y_train)
svm_train = svm_model.predict_proba(X_train[:, svm_sel_idx])[:, 1]
print(f"  [3/3] SVM — LogLoss: {safe_log_loss(y_train, svm_train):.6f}")

# ============================================================
# STEP 3 — LOOCV
# ============================================================
print("\n=== Step 3: LOOCV ===")

loo = LeaveOneOut()
loo_pred = np.zeros((X_train.shape[0], 3))

for ti, vi in loo.split(X_train):
    Xt, Xv = X_train[ti], X_train[vi]
    yt = y_train[ti]

    rf_t = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=2,
                                   class_weight='balanced', random_state=SEED)
    rf_t.fit(Xt[:, rf_sel_idx], yt)
    loo_pred[vi, 0] = rf_t.predict_proba(Xv[:, rf_sel_idx])[:, 1]

    gp_t = GaussianProcessClassifier(kernel=gp_kernel, random_state=SEED, max_iter_predict=200)
    gp_t.fit(Xt[:, gp_sel_idx], yt)
    loo_pred[vi, 1] = gp_t.predict_proba(Xv[:, gp_sel_idx])[:, 1]

    svm_t = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
    svm_t.fit(Xt[:, svm_sel_idx], yt)
    loo_pred[vi, 2] = svm_t.predict_proba(Xv[:, svm_sel_idx])[:, 1]

loocv_ll = {}
for m_idx, mn in enumerate(model_names):
    loocv_ll[mn] = safe_log_loss(y_train, loo_pred[:, m_idx])
    print(f"  {mn:15s}: LOOCV LogLoss = {loocv_ll[mn]:.4f}")

# ============================================================
# STEP 4 — Scores
# ============================================================
print("\n=== Step 4: Generating Scores ===")

train_scores = np.column_stack([rf_train, gp_train, svm_train])
test_scores = np.column_stack([
    rf_model.predict_proba(X_test[:, rf_sel_idx])[:, 1],
    gp_model.predict_proba(X_test[:, gp_sel_idx])[:, 1],
    svm_model.predict_proba(X_test[:, svm_sel_idx])[:, 1]
])
neg_scores = np.column_stack([
    rf_model.predict_proba(X_neg[:, rf_sel_idx])[:, 1],
    gp_model.predict_proba(X_neg[:, gp_sel_idx])[:, 1],
    svm_model.predict_proba(X_neg[:, svm_sel_idx])[:, 1]
])
print("  Done.")

# ============================================================
# STEP 5 — Per-Descriptor Log Loss Analysis
# ============================================================
print("\n=== Step 5: Per-Descriptor Log Loss Analysis ===")

descriptor_ll = np.full((len(descriptor_cols), 4), np.nan)  # RF, GP, SVM, ENSEMBLE

for d_idx, desc in enumerate(descriptor_cols):
    single = [d_idx]
    loocv_single = np.full((X_train.shape[0], 3), np.nan)

    for ti, vi in loo.split(X_train):
        Xt, Xv = X_train[ti], X_train[vi]
        yt = y_train[ti]

        if d_idx in rf_sel_idx:
            try:
                rf_s = RandomForestClassifier(n_estimators=500, max_depth=2, min_samples_leaf=2,
                                               class_weight='balanced', random_state=SEED)
                rf_s.fit(Xt[:, single], yt)
                loocv_single[vi, 0] = rf_s.predict_proba(Xv[:, single])[:, 1]
            except:
                loocv_single[vi, 0] = 0.5

        if d_idx in gp_sel_idx:
            try:
                gp_s = GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(1.0),
                                                  random_state=SEED, max_iter_predict=200)
                gp_s.fit(Xt[:, single], yt)
                loocv_single[vi, 1] = gp_s.predict_proba(Xv[:, single])[:, 1]
            except:
                loocv_single[vi, 1] = 0.5

        if d_idx in svm_sel_idx:
            try:
                svm_s = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                            probability=True, random_state=SEED)
                svm_s.fit(Xt[:, single], yt)
                loocv_single[vi, 2] = svm_s.predict_proba(Xv[:, single])[:, 1]
            except:
                loocv_single[vi, 2] = 0.5

    for m_idx in range(3):
        if d_idx in model_selected_indices[m_idx]:
            descriptor_ll[d_idx, m_idx] = safe_log_loss(y_train, loocv_single[:, m_idx])

    available = [m for m in range(3) if d_idx in model_selected_indices[m] and not np.isnan(loocv_single[:, m]).any()]
    if len(available) > 0:
        ens_pred = np.mean([loocv_single[:, m] for m in available], axis=0)
        descriptor_ll[d_idx, 3] = safe_log_loss(y_train, ens_pred)

print(f"\n  {'Descriptor':40s} {'RF':>10s} {'GP':>10s} {'SVM':>10s} {'ENSEMBLE':>10s}")
print(f"  {'─' * 85}")
for d_idx, desc in enumerate(descriptor_cols):
    vs = [f"{descriptor_ll[d_idx,m]:>10.4f}" if not np.isnan(descriptor_ll[d_idx,m]) else f"{'---':>10s}" for m in range(4)]
    print(f"  {desc:40s} {vs[0]} {vs[1]} {vs[2]} {vs[3]}")

# ============================================================
# STEP 6 — Weight Optimisation using LOG LOSS
# ============================================================
print("\n=== Step 6: Weight Optimisation (Log Loss) ===")
print(f"  Start: {INITIAL_WEIGHT:.4f} each, Min: {MIN_WEIGHT}")

n_models = 3
weights = np.full(n_models, INITIAL_WEIGHT)
weight_history = [weights.copy()]
ll_history = []
best_ll = np.inf
best_weights = weights.copy()
best_iteration = 0

for iteration in range(MAX_ITERATIONS):
    ens = np.clip(sum(weights[m] * train_scores[:, m] for m in range(n_models)), CLIP_EPS, 1 - CLIP_EPS)
    cur_ll = safe_log_loss(y_train, ens)
    ll_history.append(cur_ll)

    if cur_ll < best_ll:
        best_ll = cur_ll
        best_weights = weights.copy()
        best_iteration = iteration

    grads = np.zeros(n_models)
    eps = 1e-5
    for m in range(n_models):
        wp = weights.copy(); wp[m] += eps; wp /= wp.sum()
        wm = weights.copy(); wm[m] -= eps; wm = np.maximum(wm, 0); wm /= wm.sum()
        pp = np.clip(sum(wp[i] * train_scores[:, i] for i in range(n_models)), CLIP_EPS, 1 - CLIP_EPS)
        pm = np.clip(sum(wm[i] * train_scores[:, i] for i in range(n_models)), CLIP_EPS, 1 - CLIP_EPS)
        grads[m] = (safe_log_loss(y_train, pp) - safe_log_loss(y_train, pm)) / (2 * eps)

    wn = weights - LEARNING_RATE * grads
    wn = np.maximum(wn, MIN_WEIGHT)
    wn /= wn.sum()
    wc = np.max(np.abs(wn - weights))
    weights = wn
    weight_history.append(weights.copy())

    if iteration % 50 == 0:
        print(f"  Iter {iteration:4d}: LogLoss={cur_ll:.6f} Change={wc:.8f}")
    if wc < CONVERGENCE_THRESHOLD and iteration > 10:
        print(f"\n  *** CONVERGED at {iteration} ***")
        break

weights = best_weights
fc = [len(s) for s in model_selected_indices]
print(f"\n  Best at iter {best_iteration}, LogLoss={best_ll:.6f}")
for i, mn in enumerate(model_names):
    print(f"    {mn:15s}: {weights[i]:.4f} ({weights[i]*100:.1f}%) [{fc[i]} desc]")

weight_history = np.array(weight_history)

# ============================================================
# STEP 7 — Performance
# ============================================================
print("\n=== Step 7: Performance ===")

ens_train = np.clip(sum(weights[m] * train_scores[:, m] for m in range(n_models)), CLIP_EPS, 1 - CLIP_EPS)
ens_ll_train = safe_log_loss(y_train, ens_train)

per_ll = {}
for m_idx, mn in enumerate(model_names):
    per_ll[mn] = safe_log_loss(y_train, train_scores[:, m_idx])
    print(f"  {mn:15s}: LogLoss={per_ll[mn]:.6f}")
print(f"  {'ENSEMBLE':15s}: LogLoss={ens_ll_train:.6f}")

ens_bin = (ens_train >= 0.5).astype(int)
t_acc = accuracy_score(y_train, ens_bin)
t_prec = precision_score(y_train, ens_bin, zero_division=0)
t_rec = recall_score(y_train, ens_bin, zero_division=0)
t_f1 = f1_score(y_train, ens_bin, zero_division=0)
t_cm = confusion_matrix(y_train, ens_bin)
print(f"\n  Ensemble: Acc={t_acc:.4f} Prec={t_prec:.4f} Rec={t_rec:.4f} F1={t_f1:.4f}")
print(f"  CM: TN={t_cm[0,0]} FP={t_cm[0,1]} FN={t_cm[1,0]} TP={t_cm[1,1]}")

m_acc = {}
for m_idx, mn in enumerate(model_names):
    m_acc[mn] = accuracy_score(y_train, (train_scores[:, m_idx] >= 0.5).astype(int))

# LOOCV
loo_ens = np.clip(sum(weights[m] * loo_pred[:, m] for m in range(n_models)), CLIP_EPS, 1 - CLIP_EPS)
lb = (loo_ens >= 0.5).astype(int)
l_acc = accuracy_score(y_train, lb)
l_prec = precision_score(y_train, lb, zero_division=0)
l_rec = recall_score(y_train, lb, zero_division=0)
l_f1 = f1_score(y_train, lb, zero_division=0)
l_ll = safe_log_loss(y_train, loo_ens)
print(f"  LOOCV: Acc={l_acc:.4f} P={l_prec:.4f} R={l_rec:.4f} F1={l_f1:.4f} LL={l_ll:.4f}")

# ============================================================
# STEP 8 — Rank
# ============================================================
print("\n=== Step 8: Ranking ===")

ens_test = np.clip(sum(weights[m] * test_scores[:, m] for m in range(n_models)), 0, 1)
ranking_df = pd.DataFrame({'ID': test['ID'].values, 'Ensemble_Score': ens_test})
for m_idx, mn in enumerate(model_names):
    ranking_df[f'{mn}_Score'] = test_scores[:, m_idx]
    ranking_df[f'{mn}_Rank'] = test_scores[:, m_idx].argsort()[::-1].argsort() + 1
ranking_df['Ensemble_Rank'] = ens_test.argsort()[::-1].argsort() + 1
ranking_df = ranking_df.sort_values('Ensemble_Rank').reset_index(drop=True)
for d in descriptor_cols:
    ranking_df[d] = test[d].values

print(f"  Top {TOP_CANDIDATES}:")
tc = ['Ensemble_Rank', 'ID', 'Ensemble_Score'] + [f'{m}_Rank' for m in model_names]
print(ranking_df[tc].head(TOP_CANDIDATES).to_string(index=False))

# ============================================================
# STEP 9 — Negative Validation
# ============================================================
print("\n=== Step 9: Negative Validation ===")

ens_neg = np.clip(sum(weights[m] * neg_scores[:, m] for m in range(n_models)), 0, 1)
all_combined = np.concatenate([ens_test, ens_neg])
neg_ranks = [(all_combined > ns).sum() + 1 for ns in ens_neg]
print(f"  Neg rank: {min(neg_ranks)}-{max(neg_ranks)} / {len(all_combined)}, Mean={np.mean(neg_ranks):.1f}")

# ============================================================
# STEP 10 — Plots (10 plots + data CSVs)
# ============================================================
print("\n=== Step 10: Plots ===")

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 300})
colors3 = ['#1f77b4', '#ff7f0e', '#2ca02c']
colors4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#333333']

# --- Plot 1: Log Loss Convergence ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ll_history, color='darkred', linewidth=1.5, label='Log Loss')
ax.axhline(y=best_ll, color='green', linestyle='--', alpha=0.7, label=f'Best={best_ll:.4f}')
ax.axvline(x=best_iteration, color='blue', linestyle='--', alpha=0.7, label=f'Iter={best_iteration}')
ax.scatter([best_iteration], [best_ll], color='green', s=150, zorder=5, marker='*')
ax.set_xlabel('Iteration'); ax.set_ylabel('Log Loss')
ax.set_title('Ensemble Log Loss Convergence — RF + GP + SVM')
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot01_convergence.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame({'Iteration': range(len(ll_history)), 'LogLoss': ll_history,
              'Best_LogLoss': best_ll, 'Best_Iteration': best_iteration}).to_csv(
    f"{PLOTS_DIR}/plot01_data.csv", index=False)
print("  [1/10] Convergence")

# --- Plot 2: Weight Evolution ---
fig, ax = plt.subplots(figsize=(12, 7))
for m_idx, mn in enumerate(model_names):
    ax.plot(weight_history[:, m_idx], label=f'{mn} ({fc[m_idx]} desc)', linewidth=2.5, color=colors3[m_idx])
    ax.scatter([best_iteration], [best_weights[m_idx]], color=colors3[m_idx],
               s=200, zorder=5, marker='D', edgecolors='black', linewidth=2)
    ax.annotate(f'{best_weights[m_idx]:.3f}', xy=(best_iteration, best_weights[m_idx]),
                xytext=(best_iteration + 15, best_weights[m_idx] + 0.02),
                fontsize=12, fontweight='bold', color=colors3[m_idx],
                arrowprops=dict(arrowstyle='->', color=colors3[m_idx], lw=2))
ax.axhline(y=INITIAL_WEIGHT, color='gray', linestyle=':', alpha=0.5, label=f'Initial ({INITIAL_WEIGHT:.3f})')
ax.axhline(y=MIN_WEIGHT, color='red', linestyle=':', alpha=0.5, label=f'Floor ({MIN_WEIGHT})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Weight')
ax.set_title('Weight Evolution — Log Loss Optimised')
ax.legend(loc='best'); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot02_weight_evolution.png", dpi=300, bbox_inches='tight'); plt.close()
wh_df = pd.DataFrame(weight_history[:len(ll_history)], columns=model_names)
wh_df.insert(0, 'Iteration', range(len(wh_df)))
wh_df['Best_Iteration'] = best_iteration
for i, mn in enumerate(model_names):
    wh_df[f'{mn}_Final'] = best_weights[i]; wh_df[f'{mn}_NumDesc'] = fc[i]
wh_df.to_csv(f"{PLOTS_DIR}/plot02_data.csv", index=False)
print("  [2/10] Weights")

# --- Plot 3: Log Loss per model ---
fig, ax = plt.subplots(figsize=(8, 6))
names4 = list(model_names) + ['ENSEMBLE']
ll_vals = [per_ll[mn] for mn in model_names] + [ens_ll_train]
bars = ax.bar(names4, ll_vals, color=colors4, edgecolor='black')
ax.set_ylabel('Log Loss'); ax.set_title('Log Loss per Model (lower = better)')
ax.axhline(y=0.693, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Random (ln2=0.693)')
for b, v in zip(bars, ll_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot03_logloss_per_model.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame({'Model': names4, 'LogLoss': ll_vals}).to_csv(f"{PLOTS_DIR}/plot03_data.csv", index=False)
print("  [3/10] Log loss per model")

# --- Plot 4: Accuracy ---
fig, ax = plt.subplots(figsize=(8, 6))
acc4 = list(m_acc.values()) + [t_acc]
bars = ax.bar(names4, acc4, color=colors4, edgecolor='black')
ax.set_ylabel('Accuracy'); ax.set_title('Training Accuracy'); ax.set_ylim(0, 1.15)
for b, v in zip(bars, acc4):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot04_accuracy.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame({'Model': names4, 'Accuracy': acc4}).to_csv(f"{PLOTS_DIR}/plot04_data.csv", index=False)
print("  [4/10] Accuracy")

# --- Plot 5: Training vs LOOCV ---
fig, ax = plt.subplots(figsize=(9, 6))
mn5 = ['Accuracy', 'Precision', 'Recall', 'F1', 'LogLoss']
tm5 = [t_acc, t_prec, t_rec, t_f1, ens_ll_train]
lm5 = [l_acc, l_prec, l_rec, l_f1, l_ll]
x5 = np.arange(len(mn5)); w5 = 0.35
b1 = ax.bar(x5 - w5/2, tm5, w5, label='Training', color='steelblue', edgecolor='black')
b2 = ax.bar(x5 + w5/2, lm5, w5, label='LOOCV', color='coral', edgecolor='black')
ax.set_xticks(x5); ax.set_xticklabels(mn5); ax.set_ylim(0, max(max(tm5), max(lm5)) * 1.25)
ax.legend(); ax.set_title('Training vs LOOCV')
for b, v in zip(b1, tm5): ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=9)
for b, v in zip(b2, lm5): ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot05_train_vs_loocv.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame({'Metric': mn5, 'Training': tm5, 'LOOCV': lm5}).to_csv(f"{PLOTS_DIR}/plot05_data.csv", index=False)
print("  [5/10] Train vs LOOCV")

# --- Plot 6: Score Distribution ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].hist(ens_test, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=np.mean(ens_neg), color='red', linestyle='--', linewidth=2, label=f'Mean neg={np.mean(ens_neg):.4f}')
axes[0].set_xlabel('Score'); axes[0].set_ylabel('Freq'); axes[0].set_title('Test Score Distribution'); axes[0].legend()
ts6 = ranking_df.head(TOP_CANDIDATES)['Ensemble_Score'].values
axes[1].bar(range(TOP_CANDIDATES), ts6, color='green', alpha=0.7)
axes[1].axhline(y=np.max(ens_neg), color='red', linestyle='--', linewidth=2, label=f'Max neg={np.max(ens_neg):.4f}')
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Score'); axes[1].set_title(f'Top {TOP_CANDIDATES}'); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot06_score_distribution.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame({'Test_Scores': ens_test}).to_csv(f"{PLOTS_DIR}/plot06_data_test.csv", index=False)
pd.DataFrame({'Neg_Scores': ens_neg}).to_csv(f"{PLOTS_DIR}/plot06_data_neg.csv", index=False)
pd.DataFrame({'Rank': range(1, TOP_CANDIDATES+1), 'ID': ranking_df['ID'].head(TOP_CANDIDATES).values,
              'Score': ts6}).to_csv(f"{PLOTS_DIR}/plot06_data_top.csv", index=False)
print("  [6/10] Scores")

# --- Plot 7: Rank Agreement ---
fig, ax = plt.subplots(figsize=(12, 14))
rc7 = [f'{m}_Rank' for m in model_names] + ['Ensemble_Rank']
tr7 = ranking_df[rc7].head(TOP_CANDIDATES).copy()
tr7.index = ranking_df['ID'].head(TOP_CANDIDATES).values
sns.heatmap(tr7.astype(float), annot=True, fmt='.0f', cmap='YlGnBu_r', linewidths=0.5, ax=ax)
ax.set_title(f'Rank Agreement — Top {TOP_CANDIDATES}')
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot07_rank_agreement.png", dpi=300, bbox_inches='tight'); plt.close()
ranking_df[['ID', 'Ensemble_Rank', 'Ensemble_Score'] + rc7].head(TOP_CANDIDATES).to_csv(
    f"{PLOTS_DIR}/plot07_data.csv", index=False)
print("  [7/10] Ranks")

# --- Plot 8: Feature Selection Heatmap ---
fig, ax = plt.subplots(figsize=(14, 5))
sm8 = np.zeros((3, len(descriptor_cols)))
for mi, sel in enumerate(model_selected_indices):
    for s in sel: sm8[mi, s] = 1.0
sa8 = [rf_imp_norm, relief_weights, mi_scores]
im8 = np.zeros((3, len(descriptor_cols)))
for mi in range(3):
    for di in model_selected_indices[mi]:
        im8[mi, di] = sa8[mi][di] if sa8[mi][di] > 0 else 0.5
sns.heatmap(im8, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=descriptor_cols,
            yticklabels=model_names, linewidths=0.5, ax=ax, mask=(sm8 == 0))
ax.set_title('Descriptor Selection (importance scores)')
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot08_feature_selection.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame(im8, columns=descriptor_cols, index=model_names).to_csv(f"{PLOTS_DIR}/plot08_data.csv")
pd.DataFrame(sm8, columns=descriptor_cols, index=model_names).to_csv(f"{PLOTS_DIR}/plot08_mask.csv")
print("  [8/10] Features")

# --- Plot 9: Per-Descriptor Log Loss by Model + Ensemble ---
fig, ax = plt.subplots(figsize=(18, 9))
bar_w = 0.2
x_pos = np.arange(len(descriptor_cols))
plot_labels = model_names + ['ENSEMBLE']

for m_idx in range(4):
    vals = descriptor_ll[:, m_idx]
    mask = ~np.isnan(vals)
    pv = np.where(np.isnan(vals), 0, vals)
    bars = ax.bar(x_pos[mask] + m_idx * bar_w, pv[mask], bar_w,
                  label=plot_labels[m_idx], color=colors4[m_idx], edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, pv[mask]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Descriptor', fontsize=14); ax.set_ylabel('LOOCV Log Loss (lower = better)', fontsize=14)
ax.set_title('Per-Descriptor Predictive Ability — Log Loss by Model + Ensemble', fontsize=16)
ax.set_xticks(x_pos + 1.5 * bar_w); ax.set_xticklabels(descriptor_cols, rotation=45, ha='right', fontsize=10)
ax.legend(loc='upper right', fontsize=12); ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0.693, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax.text(0.02, 0.95, 'Empty = not used | Lower = better | Red line = random baseline (ln2 = 0.693)',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/plot09_descriptor_logloss.png", dpi=300, bbox_inches='tight'); plt.close()
pd.DataFrame(descriptor_ll, columns=model_names + ['ENSEMBLE'], index=descriptor_cols).to_csv(
    f"{PLOTS_DIR}/plot09_data.csv")
print("  [9/10] Descriptor log loss")

# --- Plot 10: Ensemble-Only Descriptor Log Loss ---
fig, ax = plt.subplots(figsize=(14, 7))
ens_ll_d = descriptor_ll[:, 3]

bar_colors = []
for v in ens_ll_d:
    if np.isnan(v):
        bar_colors.append('#cccccc')
    elif v < 0.2:
        bar_colors.append('#2ca02c')
    elif v < 0.4:
        bar_colors.append('#1f77b4')
    elif v < 0.55:
        bar_colors.append('#ff7f0e')
    else:
        bar_colors.append('#d62728')

pv = np.where(np.isnan(ens_ll_d), 0, ens_ll_d)
bars = ax.bar(range(len(descriptor_cols)), pv, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.9)

for i, (bar, val) in enumerate(zip(bars, ens_ll_d)):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45)
        models_used = [model_names[m] for m in range(3) if i in model_selected_indices[m]]
        ax.text(bar.get_x() + bar.get_width()/2, -0.03,
                '+'.join([m[0] for m in models_used]),
                ha='center', va='top', fontsize=7, color='gray')

ax.set_xlabel('Descriptor', fontsize=14); ax.set_ylabel('Ensemble Log Loss', fontsize=14)
ax.set_title('Ensemble — Per-Descriptor Predictive Ability (Log Loss)', fontsize=16)
ax.set_xticks(range(len(descriptor_cols)))
ax.set_xticklabels(descriptor_cols, rotation=45, ha='right', fontsize=10)
ax.axhline(y=0.693, color='red', linestyle='--', alpha=0.3, label='Random (0.693)')
ax.grid(True, axis='y', alpha=0.2)
ax.set_ylim(-0.05, max(0.8, np.nanmax(ens_ll_d) + 0.1))

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ca02c', label='Excellent (<0.2)'),
                   Patch(facecolor='#1f77b4', label='Good (0.2-0.4)'),
                   Patch(facecolor='#ff7f0e', label='Moderate (0.4-0.55)'),
                   Patch(facecolor='#d62728', label='Poor (>0.55)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Log Loss Category')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot10_ensemble_descriptor_logloss.png", dpi=300, bbox_inches='tight'); plt.close()

p10_df = pd.DataFrame({
    'Descriptor': descriptor_cols,
    'Ensemble_LogLoss': ens_ll_d,
    'Category': ['Excellent' if v < 0.2 else 'Good' if v < 0.4 else 'Moderate' if v < 0.55 else 'Poor'
                 if not np.isnan(v) else 'N/A' for v in ens_ll_d],
    'Models_Used': [', '.join([model_names[m] for m in range(3) if d in model_selected_indices[m]])
                    for d in range(len(descriptor_cols))],
    'Num_Models': [sum(1 for m in range(3) if d in model_selected_indices[m])
                   for d in range(len(descriptor_cols))],
    'RF_LogLoss': descriptor_ll[:, 0], 'GP_LogLoss': descriptor_ll[:, 1], 'SVM_LogLoss': descriptor_ll[:, 2]
})
p10_df.to_csv(f"{PLOTS_DIR}/plot10_data.csv", index=False)
print("  [10/10] Ensemble descriptor log loss")

# ============================================================
# STEP 11 — Save All Results
# ============================================================
print("\n=== Step 11: Saving ===")

ranking_df.to_csv(f"{RESULTS_DIR}/test_ranked.csv", index=False)

cl = pd.DataFrame({'Iteration': range(len(ll_history)), 'LogLoss': ll_history})
for mi, mn in enumerate(model_names):
    cl[f'{mn}_Weight'] = weight_history[:len(ll_history), mi]
cl.to_csv(f"{RESULTS_DIR}/convergence_log.csv", index=False)

joblib.dump(rf_model, f"{RESULTS_DIR}/model_RandomForest.pkl")
joblib.dump(gp_model, f"{RESULTS_DIR}/model_GP.pkl")
joblib.dump(svm_model, f"{RESULTS_DIR}/model_SVM.pkl")
np.save(f"{RESULTS_DIR}/ensemble_weights.npy", weights)

desc_metrics = pd.DataFrame({
    'Descriptor': descriptor_cols,
    'RF_LogLoss': descriptor_ll[:, 0], 'GP_LogLoss': descriptor_ll[:, 1],
    'SVM_LogLoss': descriptor_ll[:, 2], 'Ensemble_LogLoss': descriptor_ll[:, 3]
})
desc_metrics.to_csv(f"{RESULTS_DIR}/descriptor_logloss.csv", index=False)

with open(f"{RESULTS_DIR}/final_weights.txt", 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("FINAL WEIGHTS — RF + GP + SVM (Log Loss Optimised)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Best iter: {best_iteration}, LogLoss: {best_ll:.6f}\n")
    f.write("=" * 60 + "\n\n")
    for i, mn in enumerate(model_names):
        f.write(f"{mn:15s}: {weights[i]:.6f} ({weights[i]*100:.2f}%) [{fc[i]} desc]\n")
    f.write(f"\nEnsemble Train LL: {ens_ll_train:.6f}\n")
    f.write(f"Ensemble LOOCV LL: {l_ll:.6f}\n")

with open(f"{RESULTS_DIR}/descriptor_selection_report.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DESCRIPTOR SELECTION & PREDICTIVE ABILITY REPORT (Log Loss)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")
    for mi, mn in enumerate(model_names):
        f.write(f"\n{'─' * 80}\nMODEL: {mn}\n{'─' * 80}\n")
        fs = feature_selection_methods[mn]
        f.write(f"  Method: {fs['method']}\n  Descriptors ({len(model_selected_names_list[mi])}):\n")
        for dn in model_selected_names_list[mi]:
            di = descriptor_cols.index(dn)
            ll_v = descriptor_ll[di, mi]
            ss = fs['scores'].get(dn, 'coverage')
            lls = f"{ll_v:.4f}" if not np.isnan(ll_v) else "N/A"
            f.write(f"    + {dn:40s} sel_score:{ss} LogLoss:{lls}\n")
    f.write(f"\n{'=' * 80}\nUSAGE SUMMARY\n{'=' * 80}\n\n")
    for di, d in enumerate(descriptor_cols):
        ub = [mn for mi, mn in enumerate(model_names) if di in model_selected_indices[mi]]
        f.write(f"  {d:40s}: {len(ub)}/3 -- {', '.join(ub)}\n")
    f.write(f"\n{'=' * 80}\nPER-DESCRIPTOR LOG LOSS (LOOCV)\n{'=' * 80}\n")
    f.write(f"\n  {'Descriptor':40s} {'RF':>10s} {'GP':>10s} {'SVM':>10s} {'ENSEMBLE':>10s}\n  {'─' * 75}\n")
    for di, d in enumerate(descriptor_cols):
        vs = [f"{descriptor_ll[di,m]:>10.4f}" if not np.isnan(descriptor_ll[di,m]) else f"{'---':>10s}" for m in range(4)]
        f.write(f"  {d:40s} {vs[0]} {vs[1]} {vs[2]} {vs[3]}\n")
    f.write(f"\n  Lower Log Loss = better predictive ability\n")
    f.write(f"  Random baseline = ln(2) = 0.6931\n")
    f.write(f"  '---' = descriptor not used by that model\n")

# Summary report
rpt = []
rpt.append("=" * 80)
rpt.append("3-MODEL ENSEMBLE — RF + GP + SVM — LOG LOSS ONLY")
rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt.append("=" * 80)
rpt.append(f"\n1. DATA: {active_mask.sum()}A + {(~active_mask).sum()}N = {X_train.shape[0]} train, {X_test.shape[0]} test, {X_neg.shape[0]} neg")
rpt.append(f"   {len(descriptor_cols)} descriptors, NaN: {train_nan_count}+{test_nan_count}+{neg_nan_count}")
rpt.append(f"\n2. MODELS:")
rpt.append(f"   RF:  n=1000, depth=3, balanced | Selection: RF Importance")
rpt.append(f"     Features: {model_selected_names_list[0]}")
rpt.append(f"   GP:  RBF kernel, probabilistic | Selection: Relief-F")
rpt.append(f"     Features: {model_selected_names_list[1]}")
rpt.append(f"   SVM: RBF C=10 balanced | Selection: Mutual Information")
rpt.append(f"     Features: {model_selected_names_list[2]}")
rpt.append(f"\n3. COVERAGE: All {len(descriptor_cols)} used" + (f", {len(unused)} auto-assigned" if unused else ""))
rpt.append(f"\n4. OPTIMISATION: Log Loss (Binary Cross-Entropy)")
rpt.append(f"   Equal start {INITIAL_WEIGHT:.3f}, floor {MIN_WEIGHT}, lr={LEARNING_RATE}")
rpt.append(f"   Converged at iter {best_iteration}, Best LogLoss={best_ll:.6f}")
rpt.append(f"\n5. WEIGHTS:")
for i, mn in enumerate(model_names):
    bar = "█" * int(weights[i] * 40)
    rpt.append(f"   {mn:15s}: {weights[i]:.4f} ({weights[i]*100:.1f}%) [{fc[i]}d] {bar}")
rpt.append(f"\n6. PERFORMANCE:")
rpt.append(f"   Train: Acc={t_acc:.4f} P={t_prec:.4f} R={t_rec:.4f} F1={t_f1:.4f} LL={ens_ll_train:.6f}")
rpt.append(f"   LOOCV: Acc={l_acc:.4f} P={l_prec:.4f} R={l_rec:.4f} F1={l_f1:.4f} LL={l_ll:.6f}")
rpt.append(f"   CM: TN={t_cm[0,0]} FP={t_cm[0,1]} FN={t_cm[1,0]} TP={t_cm[1,1]}")
rpt.append(f"\n7. MODEL LOG LOSS:")
rpt.append(f"   {'Model':15s} {'Train LL':>12s} {'LOOCV LL':>12s}")
for mn in model_names:
    rpt.append(f"   {mn:15s} {per_ll[mn]:>12.6f} {loocv_ll[mn]:>12.6f}")
rpt.append(f"   {'ENSEMBLE':15s} {ens_ll_train:>12.6f} {l_ll:>12.6f}")
rpt.append(f"\n8. NEG: Rank {min(neg_ranks)}-{max(neg_ranks)}/{len(all_combined)}, Mean={np.mean(neg_ranks):.1f}")
rpt.append(f"   Mean neg score: {np.mean(ens_neg):.4f}, Max: {np.max(ens_neg):.4f}")
rpt.append(f"\n9. TOP {TOP_CANDIDATES}:")
for _, r in ranking_df.head(TOP_CANDIDATES).iterrows():
    mr = [f"{m}:#{int(r[f'{m}_Rank'])}" for m in model_names]
    rpt.append(f"   #{int(r['Ensemble_Rank']):3d} {r['ID']:15s} {r['Ensemble_Score']:.4f} | {', '.join(mr)}")
rpt.append(f"\n10. ENRICHMENT:")
for p in [1, 5, 10, 20]:
    n = max(1, int(len(ranking_df) * p / 100))
    s = ranking_df.head(n)['Ensemble_Score'].values
    rpt.append(f"    Top {p:2d}% ({n:4d}): [{s.min():.4f}-{s.max():.4f}]")
rpt.append(f"\n11. PER-DESCRIPTOR LOG LOSS (LOOCV):")
rpt.append(f"    {'Descriptor':40s} {'RF':>10s} {'GP':>10s} {'SVM':>10s} {'ENSEMBLE':>10s}")
for di, d in enumerate(descriptor_cols):
    vs = [f"{descriptor_ll[di,m]:>10.4f}" if not np.isnan(descriptor_ll[di,m]) else f"{'---':>10s}" for m in range(4)]
    rpt.append(f"    {d:40s} {vs[0]} {vs[1]} {vs[2]} {vs[3]}")
rpt.append(f"\n12. FILES:")
rpt.append(f"    results/: test_ranked.csv, convergence_log.csv, final_weights.txt")
rpt.append(f"    results/: descriptor_selection_report.txt, descriptor_logloss.csv")
rpt.append(f"    results/: summary_report.txt, dropped_nan_ids.csv, ensemble_weights.npy")
rpt.append(f"    results/: model_RandomForest.pkl, model_GP.pkl, model_SVM.pkl")
rpt.append(f"    plots/:   plot01-10 (10 plots + data CSVs)")
rpt.append(f"\n13. METHOD: Independent ensemble, Log Loss optimised")
rpt.append(f"    RF=importance, GP=Relief-F, SVM=MI, coverage enforced, seed={SEED}")
rpt.append(f"    Log Loss heavily penalises confident wrong predictions")
rpt.append(f"    Random baseline LogLoss = ln(2) = 0.6931")
rpt.append("\n" + "=" * 80 + "\nEND\n" + "=" * 80)

rpt_text = "\n".join(rpt)
with open(f"{RESULTS_DIR}/summary_report.txt", 'w') as f:
    f.write(rpt_text)
print(rpt_text)
print(f"\n=== PHASE 2 COMPLETE ===")