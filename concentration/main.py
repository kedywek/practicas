import os
import glob
import mne
import numpy as np
import optuna
import matplotlib.pyplot as plt
from scipy.signal import welch

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.feature_selection import RFECV
from sklearn.base import clone

import joblib
from joblib import Parallel, delayed
from pyriemann.classification import MDM
from pyriemann.classification import FgMDM
from mne.decoding import CSP


import torch

from braindecode.models import EEGNet
from braindecode.classifier import EEGClassifier

from skorch.callbacks import EarlyStopping
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("pyriemann").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["LOKY_MAX_CPU_COUNT"] = str(2)
mne.set_log_level("ERROR")
def get_tree_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=1),
        "HistGradient": HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=1)
    }

def get_riemann_models():

    return {

        "MDM":
        make_pipeline(
            Covariances(estimator="oas"),
            MDM(metric="riemann")
        ),

        "FgMDM":
        make_pipeline(
            Covariances(estimator="oas"),
            FgMDM(metric="riemann")
        ),

        "Riemann + LogReg":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        ),

        "Riemann + LinSVM":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            SVC(
                kernel='linear',
                class_weight="balanced",
                random_state=42
            )
        ),
        "Riemann + RBF-SVM":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            SVC(
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )
        ),
        "Riemann + LinearSVC":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            LinearSVC(
                class_weight='balanced',
                random_state=42
            )
        ),
        "Riemann + HistGradient":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            HistGradientBoostingClassifier(
                max_iter=100,
                random_state=42
            )
        ),
        
        "Riemann + XGBoost":
        make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann'),
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
        )
    }
def get_csp_models():
    return {
        "CSP + LinSVM":
        make_pipeline(
            CSP(
                n_components=6,
                reg='oas',
                log=True,
                norm_trace=False
            ),
            SVC(
                kernel='linear',
                class_weight='balanced',
                random_state=42,
            )
        ),

        "CSP + LogReg":
        make_pipeline(
            CSP(
                n_components=6,
                reg='oas',
                log=True,
                norm_trace=False
            ),
            LogisticRegression(
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
            )
        )
    }

def get_eegnet():

   return EEGClassifier(

        module=EEGNet,

        module__n_chans=11,
        module__n_outputs=2,
        module__n_times=1000,

        cropped=False,

        optimizer=torch.optim.Adam,

        batch_size=64,
        lr=1e-3,
        max_epochs=100,

        callbacks=[
            EarlyStopping(patience=10)
        ],

        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def get_feature_names(ch_names):
    bands = ['Delta', 'Theta', 'Alpha_low', 'Alpha_high', 'Beta_low', 'Beta_high', 'Gamma']
    names = []
    for band in bands:
        # names.extend([f"{ch} {band} (Power)" for ch in ch_names])
        # names.extend([f"{ch} {band} (Log)" for ch in ch_names])
        names.extend([
            f"{ch} Rel {band}"
            for ch in ch_names
        ])
    names.extend([
        f"{ch} Theta/BetaLow"
        for ch in ch_names
    ])

    names.extend([
        f"{ch} Theta/BetaHigh"
        for ch in ch_names
    ])

    names.extend([
        f"{ch} AlphaLow/BetaLow"
        for ch in ch_names
    ])

    names.extend([
        f"{ch} AlphaLow/BetaHigh"
        for ch in ch_names
    ])

    names.extend([
        f"{ch} AlphaHigh/BetaLow"
        for ch in ch_names
    ])

    names.extend([
        f"{ch} AlphaHigh/BetaHigh"
        for ch in ch_names
    ])
    names.extend([
        f"{ch} EI"
        for ch in ch_names
    ])
    # names.extend([
    # f"{ch} Hjorth Activity"
    #     for ch in ch_names
    # ])

    # names.extend([
    #     f"{ch} Hjorth Mobility"
    #     for ch in ch_names
    # ])

    # names.extend([
    #     f"{ch} Hjorth Complexity"
    #     for ch in ch_names
    # ])

    names.extend([
        "F4-F3 Alpha Low Asymmetry",
        "F4-F3 Alpha High Asymmetry",
        "FP2-FP1 Alpha Low Asymmetry",
        "FP2-FP1 Alpha High Asymmetry"
    ])
    return np.array(names)

def extract_features_original(epoch_data: np.ndarray, sfreq: float, ch_names: list) -> np.ndarray:
    epoch_data = epoch_data - np.mean(epoch_data, axis=0, keepdims=True)
        
    n_channels = epoch_data.shape[0]
    nperseg = min(epoch_data.shape[1], int(sfreq * 2))
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
    bands = {
        'delta': (0.5, 4.0),

        'theta': (4.0, 8.0),

        'alpha_low': (8.0, 10.0),
        'alpha_high': (10.0, 13.0),

        'beta_low': (13.0, 20.0),
        'beta_high': (20.0, 30.0),

        'gamma': (30.0, 45.0)
    }
    features = []
    band_powers = {}
    
    for name, (f_min, f_max) in bands.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        p_band = np.trapezoid(psd[:, mask], freqs[mask], axis=-1) if np.any(mask) else np.zeros(n_channels)
        band_powers[name] = p_band
        # features.extend(p_band)
        # features.extend(0.5 * np.log2(2 * np.pi * np.e * (p_band + 1e-12)))
    total_power = (
        band_powers['delta'] +
        band_powers['theta'] +
        band_powers['alpha_low'] +
        band_powers['alpha_high'] +
        band_powers['beta_low'] +
        band_powers['beta_high'] +
        band_powers['gamma'] +
        1e-12
    )
    features.extend(band_powers['delta'] / total_power)

    features.extend(band_powers['theta'] / total_power)

    features.extend(band_powers['alpha_low'] / total_power)
    features.extend(band_powers['alpha_high'] / total_power)

    features.extend(band_powers['beta_low'] / total_power)
    features.extend(band_powers['beta_high'] / total_power)

    features.extend(band_powers['gamma'] / total_power)


    features.extend(band_powers['theta'] / (band_powers['beta_low'] + 1e-12))
    features.extend(band_powers['theta'] / (band_powers['beta_high'] + 1e-12))
    features.extend(band_powers['alpha_low'] / (band_powers['beta_low'] + 1e-12))
    features.extend(band_powers['alpha_low'] / (band_powers['beta_high'] + 1e-12))
    features.extend(band_powers['alpha_high'] / (band_powers['beta_low'] + 1e-12))
    features.extend(band_powers['alpha_high'] / (band_powers['beta_high'] + 1e-12))

    features.extend((band_powers['beta_high'] + band_powers['beta_low']) / (band_powers['theta'] + band_powers['alpha_low'] + band_powers['alpha_high'] + 1e-12))

    # for ch in range(epoch_data.shape[0]):
    #     activity, mobility, complexity = hjorth_parameters(epoch_data[ch])

    #     features.extend([
    #         activity,
    #         mobility,
    #         complexity
    #     ])

    pairs = [
        ("F3", "F4"),
        ("FP1", "FP2"),
    ]
    channel_map = {
        name: idx
        for idx, name in enumerate(ch_names)
    }
    for left, right in pairs:

        l = channel_map[left]
        r = channel_map[right]

        asym_low = np.log(band_powers['alpha_low'][r] + 1e-12) - np.log(band_powers['alpha_low'][l] + 1e-12)
        asym_high = np.log(band_powers['alpha_high'][r] + 1e-12) - np.log(band_powers['alpha_high'][l] + 1e-12)
        features.append(asym_low)
        features.append(asym_high)

    
    return np.array(features)

def hjorth_parameters(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)

    activity = np.var(signal)

    mobility = np.sqrt(
        np.var(diff1) /
        (activity + 1e-12)
    )

    complexity = np.sqrt(
        np.var(diff2) /
        (np.var(diff1) + 1e-12)
    ) / (mobility + 1e-12)

    return activity, mobility, complexity

def normalize_features(X: np.ndarray, groups: np.ndarray, method: str = "zscore") -> np.ndarray:
    X_out = np.copy(X)
    for g in np.unique(groups):
        mask = groups == g
        sub = X[mask]
        if method == "zscore":
            mean = np.mean(sub, axis=0)
            std = np.std(sub, axis=0) + 1e-12
            X_out[mask] = (sub - mean) / std
        elif method == "baseline_ratio":
            baseline = np.median(sub, axis=0) + 1e-12
            X_out[mask] = sub / baseline
    return X_out

def normalize_combined(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X_out = np.copy(X)
    for g in np.unique(groups):
        mask = groups == g
        sub = X[mask]
        baseline = np.median(sub, axis=0) + 1e-12
        sub_ratio = sub / baseline
        mean = np.mean(sub_ratio, axis=0)
        std = np.std(sub_ratio, axis=0) + 1e-12
        X_out[mask] = (sub_ratio - mean) / std
    return X_out

def normalize_baseline_strict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X_out = np.copy(X)
    for g in np.unique(groups):
        mask_all = groups == g
        mask_rest = (groups == g) & (y == 0)
        if not np.any(mask_rest): mask_rest = mask_all 
        sub_rest = X[mask_rest]
        mean_rest = np.mean(sub_rest, axis=0)
        std_rest = np.std(sub_rest, axis=0) + 1e-12
        X_out[mask_all] = (X[mask_all] - mean_rest) / std_rest
    return X_out


def load_eegmat_epochs(data_dir: str, target_channels: list = None):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "Subject*_*.edf"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "Subject*_*.edf")))
    map_ch = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8', 'Fp1': 'FP1', 'Fp2': 'FP2'}
    X_list, y_list, g_list = [], [], []
    ch_names_out, sfreq_out = None, 160.0
    
    print("Loading Data and Extracting Features (this takes a minute)...")
    for path in files:
        parts = os.path.basename(path).split('.')[0].split('_')
        if len(parts) < 2: continue
        try:
            sub_id = int(parts[0].replace('Subject', ''))
            cond = int(parts[1])
        except ValueError: continue
        y_val = 0 if cond == 1 else 1
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        raw.rename_channels({ch: map_ch.get(ch.replace('EEG ', '').replace('-REF', '').strip(), ch.replace('EEG ', '').replace('-REF', '').strip()) for ch in raw.ch_names})

        if target_channels:

            missing = [
                c for c in target_channels
                if c not in raw.ch_names
            ]

            if missing:
                print(
                    f"Missing channels in {os.path.basename(path)}: {missing}"
                )
                continue
            raw.pick(target_channels)
            raw.reorder_channels(target_channels)
            
        data = raw.get_data()
        sfreq_out = raw.info['sfreq']
        ch_names_out = raw.ch_names
        
        win, step = int(2.0 * sfreq_out), int(1.0 * sfreq_out)
        for start in range(0, data.shape[1] - win, step):
            X_list.append(data[:, start:start + win])
            y_list.append(y_val)
            g_list.append(sub_id)
            
    print(f"Loaded {len(X_list)} epochs.")
    return np.array(X_list), np.array(y_list), np.array(g_list), sfreq_out, ch_names_out

def process_fold(
    train_idx,
    test_idx,
    X,
    y,
    groups,
    model,
    rfecv_estimator=None
):
    Xt = X[train_idx]
    Xte = X[test_idx]

    yt = y[train_idx]
    yte = y[test_idx]

    train_groups = groups[train_idx]

    feature_count = None
    support = None

    if rfecv_estimator is not None:

        inner_splits = min(
            5,
            len(np.unique(train_groups))
        )

        rfecv = RFECV(
            estimator=rfecv_estimator,
            step=5,
            cv=GroupKFold(n_splits=inner_splits),
            scoring='balanced_accuracy',
            min_features_to_select=5,
            n_jobs=1   # WAŻNE
        )

        rfecv.fit(
            Xt,
            yt,
            groups=train_groups
        )

        Xt = rfecv.transform(Xt)
        Xte = rfecv.transform(Xte)

        feature_count = rfecv.n_features_
        support = rfecv.support_

    local_model = clone(model)

    local_model.fit(Xt, yt)

    pred = local_model.predict(Xte)

    return (
        yte,
        pred,
        feature_count,
        support
    )

def evaluate_loso(X, y, groups, model_dict, desc, rfecv_estimator=None):
    loso = LeaveOneGroupOut()

    print(f"\n--- {desc} (Leave-One-Subject-Out: {len(np.unique(groups))} Folds) ---")

    for name, model in model_dict.items():

        y_true = []
        y_pred = []
        feature_counts = []

        supports = []

        results = Parallel(
            n_jobs=5,
            verbose=0
        )(
            delayed(process_fold)(
                train_idx,
                test_idx,
                X,
                y,
                groups,
                model,
                rfecv_estimator
            )
            for train_idx, test_idx in
            loso.split(X, y, groups)
        )

        y_true = []
        y_pred = []

        feature_counts = []
        supports = []

        for (
            yte,
            pred,
            feature_count,
            support
        ) in results:

            y_true.extend(yte)
            y_pred.extend(pred)

            if feature_count is not None:
                feature_counts.append(
                    feature_count
                )

            if support is not None:
                supports.append(
                    support
                )
        print(" " * 80, end="\r")

        if len(y_true) > 0:

            print(
                f"{name:<20} | "
                f"Acc: {accuracy_score(y_true, y_pred)*100:.2f}% | "
                f"Bal Acc: {balanced_accuracy_score(y_true, y_pred)*100:.2f}% | "
                f"F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}"
            )

            if len(feature_counts) > 0:
                print(
                    f"{'':<20} | "
                    f"Avg Features: {np.mean(feature_counts):.1f}"
                )
        else:
            print(f"{name:<20} | Skipped")
        if len(supports):
            freq = np.mean(
                np.vstack(supports),
                axis=0
            )
            names = get_feature_names(ch_names)

            stable_idx = np.where(freq >= 0.8)[0]
            joblib.dump(
                stable_idx,
                "stable_features.joblib"
            )
            print("\nStable features:")

            for idx in stable_idx:
                print(
                    f"{freq[idx]:.2f}",
                    names[idx]
            )
            print(
                f"{'':<20} | Stable (>80%): "
                f"{np.sum(freq >= 0.8)}"
            )

def optimize_histgradient(trial, X, y, groups):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 60),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = HistGradientBoostingClassifier(**param)
    gkf = GroupKFold(n_splits=5)
    b_acc_scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2: continue
        model.fit(X[train_idx], y[train_idx])
        b_acc_scores.append(balanced_accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return np.mean(b_acc_scores)


if __name__ == "__main__":
    eegmat_dir = "data/eegmat"
    mne.set_log_level("ERROR")
    
    target_channels = [
        'F3', 'F4', 
        # 'Fz', 'Cz', 'Pz',
        'FP1','FP2'
    ]
   
    X_epochs, y, groups, sfreq, ch_names = load_eegmat_epochs(eegmat_dir, target_channels=target_channels)
    print(sfreq)
    print(X_epochs.shape)

    RUN_ML = False
    RUN_RIEMANN = False
    RUN_CSP = False
    RUN_FINAL_TRAINING = True
    RUN_EEGNET = False
    
    X_features_orig = np.array([
        extract_features_original(epoch, sfreq, ch_names)
        for epoch in X_epochs
    ])  

    print("Feature shape:", X_features_orig.shape)
    print("Expected features:", len(get_feature_names(ch_names)))

    assert X_features_orig.shape[1] == len(get_feature_names(ch_names))

    norm_datasets = {
        # "RAW (No Norm)": {"data": X_features_orig},
        "STRICT BASELINE (Theoretically Perfect)": {"data": normalize_baseline_strict(X_features_orig, y, groups)}
    }

    ml_models = {
        "LogReg": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),

        "LinSVM": SVC(
            kernel="linear",
            class_weight="balanced",
            random_state=42
        ),

        "rbf_SVM": SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42
        ),


        "HistGradient": HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=42
        ),

    }
    xgb_rfecv = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1
    )
    gkf = GroupKFold(n_splits=5)

    if RUN_ML:
        for desc, config in norm_datasets.items():
            print(f"\n>>>>> ML Pipeline: {desc} <<<<<")
            dataset = config["data"]

            evaluate_loso(
                dataset,
                y,
                groups,
                ml_models,
                f"ML LOSO Test ({desc})",
                rfecv_estimator=xgb_rfecv
            )

    if RUN_RIEMANN:
        riemann_models = get_riemann_models()
        evaluate_loso(X_epochs, y, groups, riemann_models, "Riemannian LOSO Test (Covariance + Tangent Space)")
    
    if RUN_CSP:
        csp_models = get_csp_models()

        evaluate_loso(
            X_epochs,
            y,
            groups,
            csp_models,
            "CSP LOSO Test"
        )

    if RUN_EEGNET:

        eeg_models = {
            "EEGNet": get_eegnet()
        }

        evaluate_loso(
            X_epochs.astype(np.float32),
            y,
            groups,
            eeg_models,
            "EEGNet LOSO Test"
        )

    if RUN_FINAL_TRAINING:

        X_target_norm = norm_datasets["STRICT BASELINE (Theoretically Perfect)"]["data"]

        rfecv_tune = RFECV(
            estimator=LogisticRegression(
                max_iter=3000,
                class_weight='balanced',
                random_state=42
            ),
            step=5,
            cv=gkf,
            scoring='balanced_accuracy',
            min_features_to_select=5,
            n_jobs=5
        )

        rfecv_tune.fit(
            X_target_norm,
            y,
            groups=groups
        )

        X_tune_optimal = X_target_norm[:, rfecv_tune.support_]
        
        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: optimize_histgradient(
                trial,
                X_tune_optimal,
                y,
                groups
            ),
            n_trials=30,
            show_progress_bar=True
        )

        print(f"\nFound Best Parameters: {study.best_params}")

        final_prod_model = SVCLogisticRegression(
                max_iter=3000,
                class_weight='balanced',
                random_state=42
            ),

        final_prod_model.fit(
            X_tune_optimal,
            y
        )

        joblib.dump(
            final_prod_model,
            'bci_shared_model.joblib'
        )

        joblib.dump(
            rfecv_tune.support_,
            'bci_shared_mask.joblib'
        )

        joblib.dump(
            ch_names,
            'bci_shared_channels.joblib'
        )

        print("Saved updated 'bci_shared_model.joblib' and feature masks.")