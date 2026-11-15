"""Comparações sem ajuste no teste; atualização supervisionada por bloco completo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


def _to_uv(X):
    return X*1e6


def make_model(feature, classifier, n_channels, cfg):
    """Cada .fit aprende CSP/espaço tangente/escala SÓ do treino recebido."""
    settings = cfg["classification"]
    spatial = (TangentSpace(metric="riemann", tsupdate=False) if feature == "TGSP" else
               CSP(nfilter=min(settings["csp_components"], n_channels-1), metric="euclid", log=True))
    if classifier == "LDA":
        head = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        head = SVC(kernel="linear" if classifier == "SVM-linear" else "rbf",
                   C=settings["svm_C"], gamma="scale", class_weight="balanced")
    return make_pipeline(FunctionTransformer(_to_uv, validate=False), Covariances(estimator="lwf"),
                         spatial, StandardScaler(), head)


def metrics(y, pred):
    n = len(y)
    accuracy = accuracy_score(y, pred)
    # IC binomial descritivo; não é um IC entre sujeitos nem corrige dependência temporal.
    z = 1.959963984540054
    center = (accuracy+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    both = len(np.unique(y)) == 2
    return dict(n_test=n, accuracy=accuracy,
                balanced_accuracy=balanced_accuracy_score(y, pred) if both else np.nan,
                kappa=cohen_kappa_score(y, pred) if both else np.nan,
                accuracy_ci_low=center-radius, accuracy_ci_high=center+radius)


def _score_row(block, experiment, pipeline, y, pred, n_train, train_blocks, fold="all"):
    return dict(**block.meta, experiment=experiment, pipeline=pipeline, fold=str(fold),
                n_train=n_train, train_blocks=train_blocks, status="ok", reason="", **metrics(y, pred))


def _prediction_rows(block, experiment, pipeline, indices, pred, fold="all"):
    frame = block.trials.iloc[indices][["trial_id", "trial", "label"]].copy()
    frame["predicted"] = pred
    frame["correct"] = frame.label.to_numpy() == pred
    return frame.assign(**block.meta, experiment=experiment, pipeline=pipeline, fold=str(fold))


def _skip(block, experiment, pipeline, reason):
    return dict(**block.meta, experiment=experiment, pipeline=pipeline, fold="all",
                status="skipped", reason=reason, n_test=len(block.X))


def _block_name(block):
    return f"s{block.meta['session']:02d}/{block.meta['block']}"


def cross_validate(block, cfg):
    settings = cfg["classification"]
    y = block.trials.label.to_numpy()
    counts = pd.Series(y).value_counts()
    scores, predictions = [], []
    for feature in ("TGSP", "CSP"):
        for classifier in ("LDA", "SVM-linear", "SVM-rbf"):
            name = f"{feature}+{classifier}"
            k = min(settings["cv_folds"], counts.min() if len(counts) == 2 else 0)
            if k < 2:
                scores.append(_skip(block, "offline_cv", name, "fewer_than_two_trials_per_class"))
                continue
            if settings["cv_scheme"] == "forward":
                splits = TimeSeriesSplit(n_splits=int(k)).split(block.X)
            else:
                splits = StratifiedKFold(int(k), shuffle=True, random_state=settings["seed"]).split(block.X, y)
            oof_indices, oof_pred, train_sizes = [], [], []
            for fold, (train, test) in enumerate(splits, 1):
                train_counts = pd.Series(y[train]).value_counts()
                invalid = ("training_fold_has_one_class" if len(train_counts) < 2 else
                           "fewer_than_two_training_trials_per_class" if classifier == "LDA" and train_counts.min() < 2 else "")
                if invalid:
                    scores.append(dict(_skip(block, "offline_cv", name, invalid),
                                       fold=str(fold)))
                    continue
                model = make_model(feature, classifier, len(block.channels), cfg)
                model.fit(block.X[train], y[train])
                pred = model.predict(block.X[test])
                scores.append(_score_row(block, "offline_cv", name, y[test], pred, len(train),
                                         _block_name(block), fold))
                predictions.append(_prediction_rows(block, "offline_cv", name, test, pred, fold))
                oof_indices.extend(test)
                oof_pred.extend(pred)
                train_sizes.append(len(train))
            if oof_indices:
                pooled = _score_row(block, "offline_cv", name, y[oof_indices], oof_pred,
                                    min(train_sizes), _block_name(block))
                pooled["coverage"] = len(oof_indices)/len(y)
                scores.append(pooled)
            else:
                scores.append(_skip(block, "offline_cv", name, "no_valid_training_folds"))
    return scores, predictions


def _transfer(train_blocks, test, experiment, cfg):
    classifier = "SVM-"+cfg["classification"]["online_svm_kernel"]
    name = "TGSP+"+classifier
    usable = [b for b in train_blocks if len(b.X)]
    if not usable or not len(test.X):
        return _skip(test, experiment, name, "no_accepted_train_or_test_trials"), None
    X = np.concatenate([b.X for b in usable])
    y = np.concatenate([b.trials.label.to_numpy() for b in usable])
    if len(np.unique(y)) < 2:
        return _skip(test, experiment, name, "training_has_one_class"), None
    # Auditável: nenhuma tentativa do bloco teste pode entrar no treino.
    train_ids = set().union(*(set(b.trials.trial_id) for b in usable))
    if train_ids & set(test.trials.trial_id):
        raise ValueError("Vazamento: tentativa aparece simultaneamente no treino e teste.")
    model = make_model("TGSP", classifier, len(test.channels), cfg)
    model.fit(X, y)
    pred = model.predict(test.X)
    score = _score_row(test, experiment, name, test.trials.label.to_numpy(), pred,
                       len(y), ";".join(_block_name(b) for b in usable))
    return score, _prediction_rows(test, experiment, name, np.arange(len(test.X)), pred)


def classify_subject(blocks, cfg):
    """Retorna métricas/predições. Blocos só são adicionados DEPOIS da avaliação."""
    scores, predictions = [], []
    if not cfg["classification"]["enabled"]:
        return pd.DataFrame(), pd.DataFrame()
    if len({tuple(b.channels) for b in blocks}) != 1:
        raise ValueError("Canais ou ordem variam entre blocos. Harmonize explicitamente antes de treinar.")
    if len({round(b.sfreq, 6) for b in blocks}) != 1:
        raise ValueError("Taxas de amostragem variam entre blocos; reamostragem explícita é necessária.")
    for block in blocks:
        if block.meta["block"] in {"me", "im"}:
            result, preds = cross_validate(block, cfg)
            scores.extend(result)
            predictions.extend(preds)
    by_key = {(b.meta["session"], b.meta["block"]): b for b in blocks}
    online = sorted([b for b in blocks if b.meta["block"].startswith("online_")],
                    key=lambda b: (b.meta["session"], b.meta["block"]))
    name = "TGSP+SVM-"+cfg["classification"]["online_svm_kernel"]
    for test in online:
        im = by_key.get((test.meta["session"], "im"))
        if im is None:
            scores.append(_skip(test, "session_im_frozen", name, "missing_same_session_im"))
        else:
            score, pred = _transfer([im], test, "session_im_frozen", cfg)
            scores.append(score)
            if pred is not None:
                predictions.append(pred)
    initial = by_key.get((1, "im"))
    if initial is None:
        scores.extend(_skip(b, "adaptive_protocol", name, "missing_session_1_im") for b in online)
    else:
        history = [initial]
        expected = [(s, f"online_{letter}") for s in range(1, 7) for letter in "abcd"]
        last = max((expected.index((b.meta["session"], b.meta["block"])) for b in online), default=-1)
        gap = ""
        for key in expected[:last+1]:
            test = by_key.get(key)
            if test is None:
                if not cfg["classification"]["allow_protocol_gaps"]:
                    gap = gap or f"missing_preceding_block_s{key[0]:02d}_{key[1]}"
                continue
            if gap:
                scores.append(_skip(test, "adaptive_protocol", name, gap))
                continue
            score, pred = _transfer(history, test, "adaptive_protocol", cfg)
            score["partial_sequence"] = cfg["classification"]["allow_protocol_gaps"]
            scores.append(score)
            if pred is not None:
                predictions.append(pred)
            # Atualização supervisionada cumulativa; nunca inclui im de sessões 2–6.
            history.append(test)
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
