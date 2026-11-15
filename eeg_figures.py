"""Figuras vetoriais e relatório HTML independente de serviços externos."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
from eeg_processing import amplitude_unit


COLORS = {"left": "#0072B2", "right": "#D55E00"}
LABELS = {"left": "Mão esquerda", "right": "Mão direita"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


class Report:
    def __init__(self, output, subject, cfg):
        self.output, self.cfg = output, cfg
        self.pdf = PdfPages(output / "figures.pdf")
        self.parts = [f"<h1>EDA de EEG — {escape(subject)}</h1>"]
        if cfg.get("_synthetic"):
            self.text("DEMONSTRAÇÃO COM DADOS SINTÉTICOS. Não contém resultados do sujeito EG102.")
        if cfg["input"]["eeg_unit"] == "native":
            self.text("UNIDADE FÍSICA NÃO CONFIRMADA: amplitudes e potências estão em unidades arbitrárias. "
                      "Limiares físicos de artefato em µV não são aplicados. Confirmar ganhos/unidades da aquisição "
                      "antes de comparar potência absoluta entre sujeitos.")
        if not cfg["figures"]["montage_confirmed"]:
            self.text("POSIÇÕES ARTIFICIAIS: nomes anatômicos, mapas e lateralidade são ilustrativos. "
                      "Substitua a lista de canais e confirme a montagem antes de interpretação científica.")

    def text(self, text):
        self.parts.append(f"<p>{escape(str(text))}</p>")

    def table(self, title, table):
        self.parts.append(f"<h2>{escape(title)}</h2>")
        self.parts.append(table.to_html(index=False, float_format=lambda x: f"{x:.4g}", escape=True))

    def figure(self, fig, title, caption):
        if self.cfg.get("_synthetic"):
            title = "DEMONSTRAÇÃO SINTÉTICA · "+title
        fig.suptitle(title, fontsize=12)
        if getattr(fig, "_is_topomap", False):
            fig.subplots_adjust(top=.88, bottom=.10, hspace=.45, wspace=.40)
        else:
            fig.tight_layout(rect=(0, 0, 1, .95))
        self.pdf.savefig(fig)
        image = BytesIO()
        fig.savefig(image, format="png", dpi=self.cfg["figures"]["dpi"])
        encoded = base64.b64encode(image.getvalue()).decode("ascii")
        self.parts.append(f'<h2>{escape(title)}</h2><figure><img alt="{escape(title)}" '
                          f'src="data:image/png;base64,{encoded}"><figcaption>{escape(caption)}</figcaption></figure>')
        plt.close(fig)

    def close(self):
        self.pdf.close()
        html = ("<!doctype html><html lang='pt-BR'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>EDA de EEG</title><style>body{max-width:1100px;margin:2em auto;"
                "font:16px/1.5 system-ui;padding:0 1em;color:#222}img{max-width:100%}"
                "figure{margin:1em 0}figcaption{color:#444}table{border-collapse:collapse;"
                "font-size:12px;display:block;overflow-x:auto}td,th{padding:.4em;border:1px solid #ddd}"
                "h2{margin-top:2em}p{white-space:pre-wrap}</style>" + "".join(self.parts) + "</html>")
        (self.output / "report.html").write_text(html, encoding="utf-8")


def _roi(block, cfg):
    requested = cfg["spectral"]["channels_of_interest"]
    available = [c for c in requested if c in block.channels]
    if available:
        suffix = " (provisório)" if not cfg["figures"]["montage_confirmed"] else ""
        return [(c+suffix, [block.channels.index(c)]) for c in available]
    return [("Média de todos os canais (ROI solicitada ausente)", list(range(len(block.channels))))]


def _bootstrap(values, cfg):
    """Reamostra tentativas, nunca canais como se fossem observações independentes."""
    rng = np.random.default_rng(cfg["classification"]["seed"])
    index = rng.integers(0, len(values), (cfg["figures"]["bootstrap_samples"], len(values)))
    # Algumas bordas da TFR podem ficar fora da atenção de todas as tentativas.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.nanmean(values[index], axis=1)
        return np.nanmean(values, axis=0), *np.nanpercentile(dist, [2.5, 97.5], axis=0)


def raw_plot(rec, raw, filtered, block, trials, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    first = trials.loc[trials.status == "accepted"]
    attention = first.attention_s.iloc[0] if len(first) else trials.attention_s.iloc[0]
    start = max(rec.times[0], attention-.5)
    mask = (rec.times >= start) & (rec.times < start+cfg["figures"]["raw_seconds"])
    n = min(len(block.channels), cfg["figures"]["raw_channels"])
    before, after = raw[:n, mask]*1e6, filtered[:n, mask]*1e6
    before -= np.median(before, axis=1, keepdims=True)
    after -= np.median(after, axis=1, keepdims=True)
    spacing = max(30., np.percentile(np.abs(before), 99)*2.5,
                  np.percentile(np.abs(after), 99)*2.5)
    offsets = np.arange(n)[::-1]*spacing
    for ax, data, title in zip(axes, (before, after), ("Bruto", "Pré-processado (causal)")):
        for ch in range(n):
            ax.plot(rec.times[mask]-attention, data[ch]+offsets[ch], lw=.55, color="#263238")
        ax.set_yticks(offsets, block.channels[:n])
        ax.set_xlabel("Tempo desde a atenção (s)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        for tr in trials.itertuples():
            if attention-.5 <= tr.attention_s <= start+cfg["figures"]["raw_seconds"]:
                ax.axvline(tr.attention_s-attention, color="gray", ls=":", lw=.8)
                if np.isfinite(tr.cue_s):
                    ax.axvline(tr.cue_s-attention, color=COLORS.get(tr.label, "gray"), lw=.9)
        ax.plot([.1, .1], [offsets[0]+spacing*.2, offsets[0]+spacing*.2+20], color="k", lw=2)
        ax.text(.2, offsets[0]+spacing*.2, f"20 {amplitude_unit(cfg)}", va="bottom")
    return fig


def timing_plot(trials, events):
    """Uma referência com t=0 no início motor; identidade da classe vem da cue."""
    good = trials.loc[trials.cue_s.notna() & trials.task_start_s.notna()]
    accepted = good.loc[good.status == "accepted"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={"height_ratios": [1.4, 1.6]})
    if good.empty:
        axes[0].text(.5, .5, "Nenhuma sequência com cue e início motor identificáveis.\n"
                     "Configure events.task_start para definir o início da execução/imagética como t = 0.",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].axis("off")
        axes[1].set_visible(False)
        return fig
    tr = (accepted if len(accepted) else good).iloc[0]
    ax = axes[0]
    origin, cue, end = tr.task_start_s, tr.cue_s, tr.period_end_s
    task, rest = tr.task_start_s, tr.rest_s
    class_color = COLORS[tr.label]
    def phase(start, stop, color, text):
        if np.isfinite(start) and np.isfinite(stop) and stop > start:
            ax.axvspan(start-origin, stop-origin, color=color, alpha=.22, lw=0)
            ax.text((start+stop)/2-origin, .55, text, ha="center", va="center", fontsize=10)
    phase(tr.attention_s, cue, "#E6AB02", "Atenção")
    motor_end = rest if np.isfinite(rest) else end
    phase(cue, task, class_color, "Cue")
    phase(task, motor_end, "#009E73", "Execução / imagética")
    phase(rest, end, "#999999", "Pausa")
    if np.isfinite(tr.get("baseline_start_s", np.nan)):
        bs, be = tr.baseline_start_s-origin, tr.baseline_end_s-origin
        ax.axvspan(bs, be, ymin=0, ymax=.25, color="gray", alpha=.25, hatch="///", lw=0)
        ax.text((bs+be)/2, .12, "Baseline", ha="center", fontsize=8)
        baseline_text = f"Baseline: [{bs:.10f}, {be:.10f}] s desde o início da execução/imagética."
    else:
        baseline_text = "Baseline não definida nesta sequência."
    rows = events.loc[events.trial_id == tr.trial_id].sort_values("time_s")
    text_rows = []
    translations = {"attention": "Atenção", "left": "Cue — mão esquerda", "right": "Cue — mão direita",
                    "task_start": "Início motor (sem identidade de classe)", "rest_start": "Início da pausa",
                    "trial_stop": "Fim da tentativa", "trial_start": "Início da tentativa"}
    xmin = tr.attention_s-origin
    for ev in rows.itertuples():
        delta = ev.time_s-origin
        xmin = min(xmin, delta)
        color = class_color if ev.event_name in {"left", "right"} else "#444444"
        width = 2. if ev.time_s == origin else 1.4 if ev.event_name in {"left", "right"} else .9
        ax.axvline(delta, color=color, lw=width, ls="-")
        text_rows.append([translations.get(ev.event_name, ev.event_name), f"{int(ev.code)} / 0x{int(ev.code):04X}",
                          f"{delta:.10f}", f"{ev.time_s:.10f}"])
    ax.axvline(end-origin, color="#444444", lw=.9, ls=":")
    end_name = "Próxima atenção" if np.isfinite(tr.next_attention_s) else "Fim do registro (limite)"
    text_rows.append([end_name, "—", f"{end-origin:.10f}", f"{end:.10f}"])
    ax.set_xlim(xmin-.1, end-origin+.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Tempo desde o início da execução / imagética (s)")
    ax.set_title(f"Uma janela de referência · tentativa {int(tr.trial)} · início motor = 0 s")
    ax.text(cue-origin+.06, .90, f"Cue: {LABELS[tr.label]}", color=class_color, fontsize=9, weight="bold")
    ax.grid(axis="x", alpha=.15)
    axes[1].axis("off")
    table = axes[1].table(cellText=text_rows, colLabels=["Marcador", "Código decimal / hex",
                            "Δ início motor (s)", "Tempo no arquivo (s)"], colWidths=[.37, .23, .20, .20],
                         cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#EEEEEE")
    fig.text(.5, .01, baseline_text, ha="center", fontsize=9)
    return fig


def spectral_plot(summary, block, cfg):
    roi = _roi(block, cfg)
    fig, axes = plt.subplots(len(roi), 2, figsize=(10, 3*len(roi)), squeeze=False, sharex=True, sharey=True)
    f = summary["f"]
    for row, (name, indices) in enumerate(roi):
        for col, label in enumerate(COLORS):
            ax = axes[row, col]
            mask = block.trials.label.to_numpy() == label
            if mask.any():
                for phase, style, text in (("baseline", "--", "Atenção/baseline"), ("activity", "-", "Atividade")):
                    values = summary[phase][mask][:, indices].mean(axis=1)
                    avg, low, high = _bootstrap(values, cfg)
                    db = lambda x: 10*np.log10(np.maximum(x, np.finfo(float).tiny))
                    ax.plot(f, db(avg), style, color=COLORS[label], label=text)
                    ax.fill_between(f, db(low), db(high), color=COLORS[label], alpha=.10)
            ax.set_title(f"{LABELS[label]} · {name} · n={mask.sum()}")
            ax.set_xlim(*cfg["spectral"]["display_hz"])
            ax.set_xlabel("Frequência (Hz)")
            ax.set_ylabel(f"PSD (dB re 1 {amplitude_unit(cfg)}²/Hz)")
            ax.grid(alpha=.2)
            for lo, hi in cfg["spectral"]["bands"].values():
                ax.axvspan(lo, hi, color="gray", alpha=.06)
    axes[0, 0].legend(fontsize=8)
    return fig


def tfr_plot(summary, block, cfg, relative=False):
    roi = _roi(block, cfg)
    fig, axes = plt.subplots(len(roi), 2, figsize=(10, 3*len(roi)), squeeze=False, sharex=True, sharey=True)
    freq_mask = ((summary["tf_f"] >= cfg["spectral"]["display_hz"][0]) &
                 (summary["tf_f"] <= cfg["spectral"]["display_hz"][1]))
    source = summary["tf_erd"] if relative else 10*np.log10(np.maximum(summary["tf_absolute"], 1e-30))
    if relative:
        limits, cmap = (-100, 100), "RdBu_r"
    else:
        limits, cmap = np.nanpercentile(source[:, :, freq_mask], [5, 95]), "viridis"
    for row, (name, indices) in enumerate(roi):
        for col, label in enumerate(COLORS):
            ax = axes[row, col]
            mask = block.trials.label.to_numpy() == label
            if mask.any():
                # Média de razões por tentativa para ERD; dB médio para espectrograma absoluto.
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    z = np.nanmean(source[mask][:, indices].mean(axis=1), axis=0)[freq_mask]
                im = ax.pcolormesh(summary["tf_t"], summary["tf_f"][freq_mask], z,
                                   shading="auto", cmap=cmap, vmin=limits[0], vmax=limits[1], rasterized=True)
                fig.colorbar(im, ax=ax, label="ERD/ERS (%)" if relative else f"dB re 1 {amplitude_unit(cfg)}²/Hz",
                             extend="both")
            ax.axvline(0, color="k", ls="--", lw=.8)
            ax.set_title(f"{LABELS[label]} · {name} · n={mask.sum()}")
            ax.set_xlabel("Tempo desde a cue (s)")
            ax.set_ylabel("Frequência (Hz)")
    return fig


def erd_curve_plot(summary, block, cfg):
    roi = _roi(block, cfg)
    bands = cfg["spectral"]["bands"]
    fig, axes = plt.subplots(len(bands), len(roi), figsize=(5*len(roi), 3*len(bands)), squeeze=False)
    for row, (band, (lo, hi)) in enumerate(bands.items()):
        maskf = (summary["tf_f"] >= lo) & (summary["tf_f"] < hi)
        for col, (name, indices) in enumerate(roi):
            ax = axes[row, col]
            # Integrar frequências ANTES de dividir pela baseline, em cada tentativa.
            task = summary["tf_absolute"][:, indices][:, :, maskf].sum(axis=2).mean(axis=1)
            base = summary["tf_baseline"][:, indices][:, :, maskf].sum(axis=2).mean(axis=1)
            values = 100*(task/base[:, None]-1)
            for label, color in COLORS.items():
                mask = block.trials.label.to_numpy() == label
                if mask.any():
                    avg, low, high = _bootstrap(values[mask], cfg)
                    ax.plot(summary["tf_t"], avg, color=color, label=f"{LABELS[label]} (n={mask.sum()})")
                    ax.fill_between(summary["tf_t"], low, high, color=color, alpha=.15)
            ax.axhline(0, color="gray", lw=.8)
            ax.axvline(0, color="k", ls="--", lw=.8)
            baseline = block.trials[["baseline_start_s", "baseline_end_s"]].sub(block.trials.cue_s, axis=0).median()
            ax.axvspan(baseline.iloc[0], baseline.iloc[1], color="gray", alpha=.12)
            ax.set_title(f"{band} ({lo:g}–{hi:g} Hz) · {name}")
            ax.set_xlabel("Tempo desde a cue (s)")
            ax.set_ylabel("ERD/ERS (%)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    return fig


def topomap_plot(powers, block, cfg, relative=False):
    montage = mne.channels.make_standard_montage(cfg["figures"]["montage"])
    if not set(block.channels) <= set(montage.ch_names):
        return None
    info = mne.create_info(block.channels, block.sfreq, ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    bands = list(cfg["spectral"]["bands"])
    fig, axes = plt.subplots(len(bands), 2, figsize=(9, 3.7*len(bands)), squeeze=False)
    fig._is_topomap = True
    for row, band in enumerate(bands):
        table = powers.loc[powers.band == band]
        field = "erd_percent" if relative else "activity_power"
        means = table.groupby(["label", "channel"])[field].mean()
        vals = means.to_numpy() if relative else 10*np.log10(means.to_numpy())
        if relative:
            scale = max(1., np.max(np.abs(vals)))
            vlim = (-scale, scale)
        else:
            vlim = (float(vals.min()), float(vals.max()))
            if np.isclose(*vlim):
                vlim = (vlim[0]-1, vlim[1]+1)
        for col, label in enumerate(COLORS):
            ax = axes[row, col]
            if label not in means.index.get_level_values("label"):
                ax.set_visible(False)
                continue
            values = means.loc[label].reindex(block.channels).to_numpy()
            if not relative:
                values = 10*np.log10(values)
            image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                            names=block.channels, vlim=vlim,
                                            cmap="RdBu_r" if relative else "viridis", contours=0)
            fig.colorbar(image, ax=ax, shrink=.65, label="ERD/ERS (%)" if relative else f"dB re 1 {amplitude_unit(cfg)}²")
            ax.set_title(f"{LABELS[label]} · {band}")
    if not cfg["figures"]["montage_confirmed"]:
        fig.text(.5, .015, "POSIÇÕES ARTIFICIAIS — MAPA ILUSTRATIVO, SEM VALIDADE ANATÔMICA",
                 color="#B00020", ha="center", fontsize=10, weight="bold")
    return fig


def classification_figures(scores, predictions, report):
    if scores.empty:
        return
    pooled = scores.loc[(scores.status == "ok") & (scores.fold == "all")]
    offline = pooled.loc[pooled.experiment == "offline_cv"]
    if len(offline):
        names = [f"{f}+{c}" for f in ("TGSP", "CSP") for c in ("LDA", "SVM-linear", "SVM-rbf")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, block in zip(axes, ("me", "im")):
            data = offline.loc[offline.block == block]
            for session, rows in data.groupby("session"):
                values = rows.set_index("pipeline").balanced_accuracy.reindex(names)*100
                ax.plot(range(6), values, "o-", lw=.8, label=f"Sessão {session}")
            ax.set_title("Execução motora" if block == "me" else "Imagética motora")
            ax.set_xticks(range(6), names, rotation=35, ha="right")
            ax.set_ylim(0, 103)
            ax.axhline(50, ls="--", color="gray", lw=.8)
            ax.set_ylabel("Acurácia balanceada OOF (%)")
            if len(data):
                ax.legend(fontsize=7)
        report.figure(fig, "Comparação dos seis pipelines nos blocos offline",
                      "Uma janela de 2 s por tentativa. CSP, referência tangente e escala são ajustados "
                      "dentro de cada fold. Pontos são resultados OOF de cada sessão, sem teste inferencial entre folds.")
    online = pooled.loc[pooled.experiment.isin(["session_im_frozen", "adaptive_protocol"])]
    if len(online):
        keys = sorted(set(zip(online.session, online.block)))
        lookup = {key: i for i, key in enumerate(keys)}
        fig, ax = plt.subplots(figsize=(11, 4))
        for experiment, color, label in (("session_im_frozen", "#777777", "IM da mesma sessão → online (congelado)"),
                                         ("adaptive_protocol", "#0072B2", "IM1 + online anteriores → próximo bloco")):
            rows = online.loc[online.experiment == experiment].sort_values(["session", "block"])
            if rows.empty:
                continue
            x = [lookup[(r.session, r.block)] for r in rows.itertuples()]
            ax.plot(x, rows.accuracy*100, "o-", color=color, label=label)
            ax.fill_between(x, rows.accuracy_ci_low*100, rows.accuracy_ci_high*100, color=color, alpha=.12)
        ax.set_xticks(range(len(keys)), [f"s{s}/{b[-1]}" for s, b in keys], rotation=45)
        ax.set_ylim(0, 103)
        ax.set_ylabel("Acurácia no bloco teste (%)")
        ax.set_xlabel("Ordem cronológica dos blocos online")
        ax.legend(fontsize=8)
        report.figure(fig, "Decodificação online e pseudo-coadaptação supervisionada",
                      "Predição antes de incorporar os rótulos do bloco teste. Faixas: IC de Wilson de 95% "
                      "descritivo entre tentativas, sem corrigir dependência temporal. Não são ICs entre sujeitos. "
                      "Não equivale a reproduzir decisões/feedback do modelo original em tempo real.")
    if not predictions.empty:
        from sklearn.metrics import confusion_matrix
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, experiment in zip(axes, ("session_im_frozen", "adaptive_protocol")):
            rows = predictions.loc[predictions.experiment == experiment]
            if rows.empty:
                ax.set_visible(False)
                continue
            matrix = confusion_matrix(rows.label, rows.predicted, labels=["left", "right"])
            norm = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i,j]}\n{norm[i,j]:.0%}", ha="center", va="center",
                            color="white" if norm[i,j] > .6 else "black")
            ax.set_xticks([0, 1], ["Esquerda", "Direita"])
            ax.set_yticks([0, 1], ["Esquerda", "Direita"])
            ax.set_xlabel("Predição")
            ax.set_ylabel("Classe real")
            ax.set_title(experiment)
        report.figure(fig, "Matrizes de confusão dos testes online", "Contagens e proporções por classe verdadeira.")
