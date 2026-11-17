"""Tentativas, filtros causais e medidas espectrais com baseline por tentativa."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import signal, stats

from eeg_io import Recording, code


class MissingAttentionError(ValueError):
    """Gravação legível, mas sem baseline individual identificável."""


@dataclass
class EpochBlock:
    meta: dict
    channels: list[str]
    sfreq: float
    X: np.ndarray                 # tentativas × canais × amostras (janela única de 2 s)
    trials: pd.DataFrame          # somente tentativas aceitas, na mesma ordem de X
    eda_trials: pd.DataFrame | None = None  # aceitas para a janela de atividade, independentemente de X


def event_codes(cfg):
    return {role: {code(x) for x in values} for role, values in cfg["events"].items()}


def amplitude_unit(cfg):
    return "u.a." if cfg["input"]["eeg_unit"] == "native" else "µV"


def validate_config(cfg):
    codes = event_codes(cfg)
    if cfg["input"]["eeg_unit"] not in {"V", "mV", "uV", "native"}:
        raise ValueError("Defina input.eeg_unit como V, mV, uV ou native (unidade desconhecida).")
    if not codes["attention"]:
        raise ValueError("Defina events.attention usando o código real encontrado em `inspect`.")
    if not codes["left"] or not codes["right"] or codes["left"] & codes["right"]:
        raise ValueError("Códigos left/right devem ser não vazios e distintos.")
    if codes["attention"] & (codes["left"] | codes["right"]):
        raise ValueError("Atenção não pode usar o mesmo código da classe.")
    epoch, spec, pre = cfg["epochs"], cfg["spectral"], cfg["preprocessing"]
    a, b = epoch["classification_s"]
    if a < 0 or not np.isclose(b-a, 2.):
        raise ValueError("classification_s deve definir exatamente 2 s, após o anchor.")
    a, b = epoch["activity_s"]
    if not 0 <= a < b:
        raise ValueError("activity_s inválido")
    if epoch["anchor"] not in {"cue", "task_start"}:
        raise ValueError("epochs.anchor deve ser cue ou task_start.")
    if epoch["anchor"] == "task_start" and not codes["task_start"]:
        raise ValueError("O anchor task_start exige events.task_start.")
    if epoch["baseline_start_s"] < 0 or epoch["baseline_end_before_cue_s"] < 0:
        raise ValueError("A baseline deve estar inteiramente no período de atenção pré-cue.")
    if epoch["min_baseline_s"] < max(spec["welch_window_s"], spec["tfr_window_s"]):
        raise ValueError("min_baseline_s deve comportar uma janela Welch e uma janela TFR.")
    if epoch["activity_s"][1]-epoch["activity_s"][0] < spec["welch_window_s"]:
        raise ValueError("activity_s precisa comportar uma janela Welch.")
    if not 0 < spec["tfr_step_s"] <= spec["tfr_window_s"] or not 0 <= spec["overlap"] < 1:
        raise ValueError("Passo/overlap espectral inválido.")
    lo, hi = pre["bandpass_hz"]
    flo, fhi = cfg["classification"]["frequency_hz"]
    if not 0 < lo < hi or not lo <= flo < fhi <= hi:
        raise ValueError("Faixa de classificação deve estar contida no bandpass.")
    for band in spec["bands"].values():
        if not lo <= band[0] < band[1] <= hi:
            raise ValueError("Bandas espectrais devem estar contidas no bandpass.")
    if pre["reference"] not in {"average", "original"}:
        raise ValueError("reference deve ser average ou original.")
    if cfg["figures"]["selection"] not in {"first_per_kind", "all"}:
        raise ValueError("figures.selection inválido.")
    if cfg["classification"]["cv_scheme"] not in {"stratified", "forward"}:
        raise ValueError("cv_scheme inválido.")
    if cfg["classification"]["online_svm_kernel"] not in {"linear", "rbf"}:
        raise ValueError("online_svm_kernel inválido.")
    interval = cfg["classification"].get("adaptive_update_blocks", 2)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        raise ValueError("adaptive_update_blocks deve ser um inteiro positivo.")
    figures = cfg["figures"]
    if not 0 < figures.get("tfr_alpha", .05) < 1:
        raise ValueError("tfr_alpha deve estar entre 0 e 1.")
    for name, minimum in (("tfr_permutations", 32), ("tfr_min_trials", 2)):
        value = figures.get(name, 512 if name == "tfr_permutations" else 8)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} deve ser um inteiro >= {minimum}.")


def _causal_filter(x, sos):
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[None, :, 0, None]
    return signal.sosfilt(sos, x, axis=-1, zi=zi)[0]


def preprocess(rec, cfg):
    pre = cfg["preprocessing"]
    bad = set(pre["bad_channels"])
    original = cfg["input"].get("channels") or rec.channels
    keep = [i for i, name in enumerate(rec.channels) if name not in bad and original[i] not in bad]
    if len(keep) < 3:
        raise ValueError("São necessários pelo menos três canais EEG válidos.")
    # Escala interna: volts se conhecidos; native × 1e-6 se unidade desconhecida.
    # O fator é puramente numérico neste último caso, não uma calibração em volts.
    factor = {"V": 1., "mV": 1e-3, "uV": 1e-6, "native": 1e-6}[cfg["input"]["eeg_unit"]]
    raw = rec.signal[keep].astype(float) * factor
    names = [rec.channels[i] for i in keep]
    hi = pre["bandpass_hz"][1]
    if hi >= rec.sfreq / 2:
        raise ValueError("Bandpass acima da frequência de Nyquist.")
    filtered = raw.copy()
    if pre["notch_hz"]:
        if not 0 < pre["notch_hz"] < rec.sfreq / 2:
            raise ValueError("Notch fora da faixa válida.")
        b, a = signal.iirnotch(pre["notch_hz"], Q=30, fs=rec.sfreq)
        filtered = _causal_filter(filtered, signal.tf2sos(b, a))
    sos = signal.butter(pre["order"], pre["bandpass_hz"], btype="bandpass",
                        fs=rec.sfreq, output="sos")
    filtered = _causal_filter(filtered, sos)
    if pre["reference"] == "average":
        filtered -= filtered.mean(axis=0, keepdims=True)
    class_sos = signal.butter(pre["order"], cfg["classification"]["frequency_hz"],
                              btype="bandpass", fs=rec.sfreq, output="sos")
    class_signal = _causal_filter(filtered, class_sos)
    qc = pd.DataFrame({"channel": names, "raw_std": raw.std(axis=1)*1e6,
                       "filtered_std": filtered.std(axis=1)*1e6,
                       "raw_ptp": np.ptp(raw, axis=1)*1e6,
                       "filtered_ptp": np.ptp(filtered, axis=1)*1e6,
                       "amplitude_unit": amplitude_unit(cfg)})
    return raw, filtered, class_signal, names, qc


def trial_table(rec, cfg):
    codes, ep = event_codes(cfg), cfg["epochs"]
    ev = rec.events
    attention = np.unique(ev.loc[ev.code.isin(codes["attention"]), "time_s"].to_numpy())
    if not len(attention):
        counts = {str(int(k)): int(v) for k, v in ev.code.value_counts().sort_index().items()}
        raise MissingAttentionError(f"Nenhuma atenção encontrada em {rec.path}; "
                                    f"events.attention={sorted(codes['attention'])}; "
                                    f"códigos presentes (código: contagem)={counts}. "
                                    "Confira os marcadores/exportação; baseline geral e cue não substituem atenção.")
    rows = []
    for i, start in enumerate(attention):
        end = attention[i+1] if i+1 < len(attention) else rec.times[-1]+1/rec.sfreq
        markers = ev.loc[(ev.time_s >= start) & (ev.time_s < end)]
        cues = markers.loc[markers.code.isin(codes["left"] | codes["right"])]
        row = dict(trial=i+1, attention_s=start, cue_s=np.nan, task_start_s=np.nan,
                   rest_s=np.nan, end_s=end, period_end_s=end,
                   next_attention_s=attention[i+1] if i+1 < len(attention) else np.nan,
                   cue_code=np.nan, class_source="cue", label="", status="excluded", reason="")
        if len(cues) != 1:
            row["reason"] = f"expected_one_cue_found_{len(cues)}"
            rows.append(row)
            continue
        cue = cues.iloc[0]
        row["cue_s"] = float(cue.time_s)
        row["cue_code"] = int(cue.code)
        row["label"] = "left" if cue.code in codes["left"] else "right"
        task = markers.loc[markers.code.isin(codes["task_start"]) & (markers.time_s >= cue.time_s)]
        rest = markers.loc[markers.code.isin(codes["rest_start"]) & (markers.time_s > cue.time_s)]
        stops = markers.loc[markers.code.isin(codes["trial_stop"]) & (markers.time_s > start)]
        if len(task):
            row["task_start_s"] = float(task.time_s.iloc[0])
        if len(rest):
            row["rest_s"] = float(rest.time_s.iloc[0])
            row["end_s"] = min(row["end_s"], row["rest_s"])
        if len(stops):
            row["end_s"] = min(row["end_s"], float(stops.time_s.iloc[0]))
        anchor = cue.time_s if ep["anchor"] == "cue" else row["task_start_s"]
        row.update(anchor_s=anchor, baseline_start_s=start+ep["baseline_start_s"],
                   baseline_end_s=cue.time_s-ep["baseline_end_before_cue_s"])
        if not np.isfinite(anchor):
            row["reason"] = "missing_task_start"
        elif anchor < cue.time_s or anchor >= row["end_s"]:
            row["reason"] = "invalid_event_order"
        elif row["baseline_end_s"]-row["baseline_start_s"] < ep["min_baseline_s"]-1e-8:
            row["reason"] = "short_attention_baseline"
        elif row["baseline_start_s"] < rec.times[0]+cfg["preprocessing"]["settling_s"]:
            row["reason"] = "filter_initial_transient"
        elif anchor + ep["classification_s"][1] > row["end_s"]+1e-8:
            row["reason"] = "classification_overlaps_rest_or_next_trial"
        else:
            row["status"] = "candidate"
        rows.append(row)
    for row in rows:
        common = row["reason"] if row["reason"] != "classification_overlaps_rest_or_next_trial" else ""
        reason = common
        if not reason and row["anchor_s"] + ep["activity_s"][1] > row["end_s"]+1e-8:
            reason = "activity_overlaps_rest_or_next_trial"
        row.update(eda_status="excluded" if reason else "candidate", eda_reason=reason)
    return pd.DataFrame(rows)


def segment(x, rec, start, stop):
    first = int(np.rint((start-rec.times[0])*rec.sfreq))
    n = int(np.rint((stop-start)*rec.sfreq))
    if first < 0 or n < 1 or first+n > x.shape[-1]:
        raise ValueError("Janela fora dos limites do EEG.")
    return x[:, first:first+n]


def prepare_block(rec, cfg, meta):
    raw, filtered, cls, names, qc = preprocess(rec, cfg)
    trials = trial_table(rec, cfg)
    trials["trial_id"] = [f"{meta['subject']}/s{meta['session']:02d}/{meta['block']}/{i:04d}"
                          for i in trials.trial]
    ep, pre = cfg["epochs"], cfg["preprocessing"]
    def artifact_reason(windows):
        known_unit = cfg["input"]["eeg_unit"] != "native"
        if known_unit and any(np.ptp(x, axis=1).max()*1e6 > pre["reject_peak_to_peak_uv"] for x in windows):
            return "high_peak_to_peak"
        if any(x.std(axis=1).min()*1e6 <= (pre["reject_flat_std_uv"] if known_unit else 0.) for x in windows):
            return "flat_channel"
        return ""

    epochs = []
    for idx, row in trials.iterrows():
        for status, reason_column, window in (("status", "reason", "classification_s"),
                                               ("eda_status", "eda_reason", "activity_s")):
            if row[status] != "candidate":
                continue
            baseline = segment(filtered, rec, row.baseline_start_s, row.baseline_end_s)
            activity = segment(filtered, rec, row.anchor_s+ep[window][0], row.anchor_s+ep[window][1])
            reason = artifact_reason((baseline, activity))
            trials.at[idx, reason_column] = reason
            trials.at[idx, status] = "excluded" if reason else "accepted"
            if status == "status" and not reason:
                epochs.append(segment(cls, rec, row.anchor_s+ep[window][0], row.anchor_s+ep[window][1]))
    trials["physical_amplitude_qc_applied"] = cfg["input"]["eeg_unit"] != "native"
    accepted = trials.loc[trials.status == "accepted"].reset_index(drop=True)
    eda_accepted = trials.loc[trials.eda_status == "accepted"].reset_index(drop=True)
    n_samples = int(round(2*rec.sfreq))
    X = np.stack(epochs) if epochs else np.empty((0, len(names), n_samples))
    block = EpochBlock(meta, names, rec.sfreq, X, accepted, eda_accepted)
    return block, trials, raw, filtered, qc


def spectral_summary(rec, filtered, block, cfg, with_tfr=False):
    if block.eda_trials is not None:
        block = replace(block, trials=block.eda_trials)
    spec, ep = cfg["spectral"], cfg["epochs"]
    nperseg = int(round(spec["welch_window_s"]*rec.sfreq))
    noverlap = int(round(nperseg*spec["overlap"]))
    psd_base, psd_task, powers = [], [], []
    for trial in block.trials.itertuples():
        baseline = segment(filtered, rec, trial.baseline_start_s, trial.baseline_end_s)*1e6
        task = segment(filtered, rec, trial.anchor_s+ep["activity_s"][0],
                       trial.anchor_s+ep["activity_s"][1])*1e6
        f, p0 = signal.welch(baseline, rec.sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
        _, p1 = signal.welch(task, rec.sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
        psd_base.append(p0)
        psd_task.append(p1)
        for name, (lo, hi) in spec["bands"].items():
            mask = (f >= lo) & (f < hi) # bandas [lo, hi), sem duplicar o bin de 13 Hz
            if mask.sum() < 2:
                raise ValueError(f"Resolução espectral insuficiente para a banda {name}.")
            pbase = p0[:, mask].sum(axis=-1)*(f[1]-f[0])
            ptask = p1[:, mask].sum(axis=-1)*(f[1]-f[0])
            for ch, b, a in zip(block.channels, pbase, ptask):
                powers.append(dict(trial_id=trial.trial_id, trial=trial.trial, label=trial.label,
                                   channel=ch, band=name, baseline_power=b, activity_power=a,
                                   power_unit=amplitude_unit(cfg)+"²",
                                   erd_percent=100*(a/b-1)))
    if not len(psd_base):
        return None, pd.DataFrame(columns=["trial_id", "trial", "label", "channel", "band",
                                          "baseline_power", "activity_power", "power_unit", "erd_percent"])
    result = dict(f=f, baseline=np.stack(psd_base), activity=np.stack(psd_task))
    if with_tfr:
        result.update(time_frequency(rec, filtered, block, cfg))
        result.update(tfr_significance(result, block, cfg))
    return result, pd.DataFrame(powers)


def tfr_significance(summary, block, cfg):
    """Clusters bilaterais de log-potência/baseline por tentativa, não por amostra EEG."""
    from mne.stats import combine_adjacency, permutation_cluster_1samp_test

    options = cfg["figures"]
    channels = [c for c in cfg["spectral"]["channels_of_interest"] if c in block.channels]
    frequency = ((summary["tf_f"] >= cfg["spectral"]["display_hz"][0]) &
                 (summary["tf_f"] <= cfg["spectral"]["display_hz"][1]))
    shape = (int(frequency.sum()), len(summary["tf_t"]))
    masks, records = {}, []
    family = max(1, len(channels)*2)
    alpha = options.get("tfr_alpha", .05)

    def t_stat(x):
        sd = x.std(axis=0, ddof=1)
        return np.divide(x.mean(axis=0)*np.sqrt(len(x)), sd, out=np.zeros_like(sd), where=sd > 0)

    for channel in channels:
        for label in ("left", "right"):
            selected = block.trials.label.to_numpy() == label
            n = int(selected.sum())
            mask = np.zeros(shape, dtype=bool)
            record = dict(channel=channel, label=label, n_trials=n, method="two_sided_sign_flip_log_ratio_clusters",
                          family_panels=family, alpha=alpha, p_corrected=[], n_significant_bins=0,
                          seed=cfg["classification"]["seed"], cluster_forming_alpha=.05,
                          n_permutations=options.get("tfr_permutations", 512))
            if not options.get("tfr_significance", True):
                record["status"] = "disabled"
            elif n < options.get("tfr_min_trials", 8):
                record["status"] = "insufficient_trials"
            else:
                index = block.channels.index(channel)
                absolute = summary["tf_absolute"][selected, index][:, frequency]
                baseline = summary["tf_baseline"][selected, index][:, frequency, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = 10*np.log10(absolute/baseline)
                valid = np.isfinite(values).all(axis=0)
                valid &= summary["tf_t"][None, :] >= 0  # só pós-cue; baseline não é testada contra si mesma
                values = np.where(np.isfinite(values), values, 0.)
                valid &= values.std(axis=0) > 0
                if not valid.any():
                    record["status"] = "no_valid_bins"
                else:
                    _, clusters, p_values, null = permutation_cluster_1samp_test(
                        values.reshape(n, -1), threshold=stats.t.ppf(1-.05/2, n-1), tail=0,
                        stat_fun=t_stat, adjacency=combine_adjacency(*shape), exclude=~valid.ravel(),
                        n_permutations=options.get("tfr_permutations", 512),
                        rng=cfg["classification"]["seed"], n_jobs=1, out_type="indices", verbose=False)
                    corrected = np.minimum(1., p_values*family)
                    for cluster, p in zip(clusters, corrected):
                        if p <= alpha:
                            mask.ravel()[cluster[0]] = True
                    record.update(status="tested", p_corrected=corrected.tolist(),
                                  n_significant_bins=int(mask.sum()), effective_permutations=len(null))
            masks[(channel, label)] = mask
            records.append(record)
    return dict(tfr_masks=masks, tfr_tests=records)


def time_frequency(rec, filtered, block, cfg):
    spec = cfg["spectral"]
    n = int(round(spec["tfr_window_s"]*rec.sfreq))
    step = max(1, int(round(spec["tfr_step_s"]*rec.sfreq)))
    tlo, thi = spec["tfr_range_s"]
    centers = np.arange(tlo+n/rec.sfreq/2, thi-n/rec.sfreq/2+1e-8, step/rec.sfreq)
    absolute, relative, baselines = [], [], []
    for tr in block.trials.itertuples():
        start = max(tr.attention_s, tr.cue_s+tlo, rec.times[0])
        stop = min(tr.end_s, tr.cue_s+thi, rec.times[-1]+1/rec.sfreq)
        data = segment(filtered, rec, start, stop)*1e6
        base = segment(filtered, rec, tr.baseline_start_s, tr.baseline_end_s)*1e6
        f, p0 = signal.welch(base, rec.sfreq, nperseg=n, noverlap=n-step, axis=-1)
        _, t, p = signal.spectrogram(data, rec.sfreq, nperseg=n, noverlap=n-step,
                                     window="hann", detrend="constant", scaling="density", axis=-1)
        t = t+start-tr.cue_s
        interp = np.empty((*p.shape[:-1], len(centers)))
        for ch in range(p.shape[0]):
            for freq in range(p.shape[1]):
                interp[ch, freq] = np.interp(centers, t, p[ch, freq], left=np.nan, right=np.nan)
        absolute.append(interp)
        baselines.append(p0)
        relative.append(100*(interp/np.maximum(p0[..., None], np.finfo(float).tiny)-1))
    return dict(tf_f=f, tf_t=centers, tf_absolute=np.stack(absolute), tf_erd=np.stack(relative),
                tf_baseline=np.stack(baselines))


def event_table(rec, cfg, meta, trials):
    table = rec.events.copy()
    roles = event_codes(cfg)
    table["event_name"] = table.code.map(
        lambda x: "/".join(k for k, v in roles.items() if x in v) or f"unmapped_0x{x:04X}")
    if trials.empty:
        return table.assign(**meta, trial_id="", from_attention_s=np.nan,
                            from_cue_s=np.nan, from_task_start_s=np.nan)
    index = np.searchsorted(trials.attention_s.to_numpy(), table.time_s, side="right")-1
    # Um TrialStart antes da atenção pertence à tentativa que vai começar.
    for j, row in enumerate(table.itertuples()):
        if row.code in roles["trial_start"] and row.code not in roles["attention"]:
            following = np.searchsorted(trials.attention_s.to_numpy(), row.time_s, side="left")
            if following < len(trials):
                index[j] = following
    table["trial_id"] = [trials.iloc[i].trial_id if i >= 0 else "" for i in index]
    table["from_attention_s"] = [t-trials.iloc[i].attention_s if i >= 0 else np.nan
                                  for t, i in zip(table.time_s, index)]
    table["from_cue_s"] = [t-trials.iloc[i].cue_s if i >= 0 else np.nan
                            for t, i in zip(table.time_s, index)]
    table["from_task_start_s"] = [t-trials.iloc[i].task_start_s if i >= 0 else np.nan
                                   for t, i in zip(table.time_s, index)]
    return table.assign(**meta)
