"""Relatório conjunto: protocolos comparáveis e sujeitos como unidade inferencial."""
from __future__ import annotations

from itertools import product
from html import escape
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from eeg_figures import Report
from eeg_io import BLOCKS


PROTOCOLS = {"me_cv": "CV ME", "mi_cv": "CV MI", "frozen": "MI → online", "adaptive": "Adaptativo por pares"}
BANDS = {"mu": "Alfa/mu (8–13 Hz)", "beta": "Beta (13–30 Hz)"}
KINDS = {"me": "ME", "im": "MI", "online": "Online"}


def _bootstrap(values, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def subject_test(values, name, seed=42):
    """Teste bilateral de troca de sinais: diferenças ou inclinações por sujeito."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    low, high = _bootstrap(values, seed)
    result = dict(test=name, n_subjects=n, effect_mean=float(values.mean()) if n else np.nan,
                  ci_low=low, ci_high=high, p_raw=np.nan, p_holm=np.nan,
                  status="insufficient_subjects" if n < 3 else "tested", method="subject_sign_flip_two_sided")
    if n < 3:
        return result
    observed = abs(values.mean())
    if n <= 15:
        signs = np.array(list(product((-1., 1.), repeat=n)))
        null = np.abs((signs*values).mean(axis=1))
        p = np.mean(null >= observed-1e-12)
        result["resampling"] = f"exact_{len(null)}"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1., 1.], size=(10000, n))
        null = np.abs((signs*values).mean(axis=1))
        p = (1+np.sum(null >= observed-1e-12))/(len(null)+1)
        result["resampling"] = "monte_carlo_10000"
    result["p_raw"] = float(p)
    return result


def correct_tests(rows):
    """Holm sobre a família planejada inteira, inclusive contrastes indisponíveis."""
    order = sorted((i for i, r in enumerate(rows) if np.isfinite(r["p_raw"])), key=lambda i: rows[i]["p_raw"])
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(rows)-rank)*rows[i]["p_raw"]))
        rows[i]["p_holm"] = previous
    return pd.DataFrame(rows)


def accuracy_tables(scores):
    columns = ["subject", "session", "block", "protocol", "accuracy", "balanced_accuracy"]
    if scores.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["subject", "protocol", "accuracy", "balanced_accuracy", "n_blocks"]), pd.DataFrame(columns=["subject", "comparison", "difference_pp", "n_matched"])
    rows = scores.loc[(scores.pipeline == "TGSP+LR") & (scores.status == "ok") &
                      (scores.fold.astype(str) == "all")].copy()
    rows["protocol"] = rows.experiment.map({"session_im_frozen": "frozen", "adaptive_protocol": "adaptive"})
    offline = rows.experiment == "offline_cv"
    rows.loc[offline, "protocol"] = rows.loc[offline, "block"].map({"me": "me_cv", "im": "mi_cv"})
    rows = rows.dropna(subset=["protocol"])
    if rows.duplicated(["subject", "session", "block", "protocol"]).any():
        raise ValueError("Resultados duplicados no resumo; folds não podem ser contados duas vezes.")
    if "partial_sequence" in rows:
        rows = rows.loc[~rows.partial_sequence.astype(str).str.lower().eq("true")].copy()
    if "coverage" in rows:
        # CV forward incompleta permanece nos relatórios individuais; não equivale à OOF completa.
        rows = rows.loc[~rows.protocol.isin(["me_cv", "mi_cv"]) | rows.coverage.eq(1)].copy()
    by_subject = rows.groupby(["subject", "protocol"], as_index=False).agg(
        accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), n_blocks=("block", "size"))
    pairs = []
    for a, b, keys in (("me_cv", "mi_cv", ["subject", "session"]),
                       ("frozen", "adaptive", ["subject", "session", "block"])):
        left = rows.loc[rows.protocol == a, keys+["accuracy"]]
        right = rows.loc[rows.protocol == b, keys+["accuracy"]]
        matched = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        matched["difference_pp"] = 100*(matched.accuracy_b-matched.accuracy_a)
        for subject, data in matched.groupby("subject"):
            pairs.append(dict(subject=subject, comparison=f"{b} minus {a}",
                              difference_pp=data.difference_pp.mean(), n_matched=len(data)))
    return rows, by_subject, pd.DataFrame(pairs, columns=["subject", "comparison", "difference_pp", "n_matched"])


def physiology_tables(power, trials, roi):
    columns = ["subject", "session", "block", "trial_id", "trial", "label", "kind", "band", "erd_percent", "observed_trial", "progress", "n_channels"]
    if power.empty or trials.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["subject", "kind", "band", "slope_pp", "n_trials", "status"])
    values = power.loc[power.band.isin(BANDS) & power.channel.isin(roi)].copy()
    if values.duplicated(["subject", "trial_id", "band", "channel"]).any():
        raise ValueError("Potências duplicadas por tentativa/canal/banda no resumo.")
    order = trials.copy()
    order["kind"] = np.where(order.block.str.startswith("online_"), "online", order.block)
    order["block_order"] = order.block.map(dict(zip(BLOCKS, range(len(BLOCKS)))))
    order = order.sort_values(["subject", "session", "block_order", "trial"])
    if order.duplicated(["subject", "trial_id"]).any():
        raise ValueError("Tentativas duplicadas no resumo.")
    order["observed_trial"] = order.groupby(["subject", "kind"]).cumcount()+1
    count = order.groupby(["subject", "kind"])["trial_id"].transform("size")
    order["progress"] = (order.observed_trial-1)/np.maximum(count-1, 1)
    values = values.groupby(["subject", "trial_id", "band"], as_index=False).agg(
        erd_percent=("erd_percent", "mean"), n_channels=("channel", "nunique"))
    values = values.merge(order[["subject", "trial_id", "session", "block", "trial", "label", "kind", "observed_trial", "progress", "eda_status"]],
                          on=["subject", "trial_id"], how="left", validate="many_to_one", indicator=True)
    if values._merge.eq("left_only").any():
        raise ValueError("Potência sem tentativa correspondente no resumo.")
    values = values.drop(columns="_merge")
    values = values.loc[(values.eda_status == "accepted") & np.isfinite(values.erd_percent)].copy()
    slopes = []
    for (subject, kind, band), data in values.groupby(["subject", "kind", "band"]):
        counts = data.label.value_counts()
        row = dict(subject=subject, kind=kind, band=band, slope_pp=np.nan, n_trials=len(data), status="insufficient_trials")
        if len(data) >= 8 and all(counts.get(label, 0) >= 3 for label in ("left", "right")):
            design = np.c_[np.ones(len(data)), data.progress, data.label.eq("right").astype(float)]
            if np.linalg.matrix_rank(design) == 3:
                slope = np.linalg.lstsq(design, data.erd_percent, rcond=None)[0][1]
                row.update(slope_pp=float(slope), status="estimated")
        slopes.append(row)
    return values, pd.DataFrame(slopes, columns=["subject", "kind", "band", "slope_pp", "n_trials", "status"])


def matched_online(blocks):
    keys = ["subject", "session", "block"]
    return blocks.loc[blocks.protocol == "frozen", keys+["accuracy"]].merge(
        blocks.loc[blocks.protocol == "adaptive", keys+["accuracy"]], on=keys,
        suffixes=("_frozen", "_adaptive"), validate="one_to_one")


def accuracy_trends(blocks):
    rows = []
    for subject, data in matched_online(blocks).groupby("subject"):
        x = ((data.session-1)*4+data.block.str[-1].map(dict(zip("abcd", range(1, 5))))).to_numpy()
        for protocol in ("frozen", "adaptive"):
            row = dict(subject=subject, protocol=protocol, n_blocks=len(data), slope_pp=np.nan,
                       first_block=int(x.min()), last_block=int(x.max()), status="insufficient_blocks")
            if len(data) >= 4 and np.ptp(x) > 0:
                progress = (x-x.min())/np.ptp(x)
                row.update(slope_pp=float(np.polyfit(progress, data["accuracy_"+protocol]*100, 1)[0]), status="estimated")
            rows.append(row)
    return pd.DataFrame(rows, columns=["subject", "protocol", "n_blocks", "slope_pp", "first_block", "last_block", "status"])


def summary_statistics(paired, slopes, seed=42, acc_slopes=None):
    rows = []
    for comparison in ("mi_cv minus me_cv", "adaptive minus frozen"):
        values = paired.loc[paired.comparison == comparison, "difference_pp"]
        rows.append(subject_test(values, comparison, seed))
    for protocol in ("frozen", "adaptive"):
        values = [] if acc_slopes is None else acc_slopes.loc[acc_slopes.protocol == protocol, "slope_pp"]
        rows.append(subject_test(values, f"Accuracy slope {protocol}", seed))
    for kind in ("me", "im", "online"):
        for band in BANDS:
            values = slopes.loc[(slopes.kind == kind) & (slopes.band == band), "slope_pp"]
            rows.append(subject_test(values, f"ERD slope {kind}/{band}", seed))
    return correct_tests(rows)


def _subject_figure(subject, accuracy, physiology):
    fig = plt.figure(figsize=(11, 10))
    grid = fig.add_gridspec(4, 2)
    acc = fig.add_subplot(grid[0, :])
    for protocol, color in (("frozen", "#777777"), ("adaptive", "#0072B2")):
        data = accuracy.loc[(accuracy.subject == subject) & (accuracy.protocol == protocol)].sort_values(["session", "block"])
        if len(data):
            x = (data.session-1)*4+data.block.str[-1].map(dict(zip("abcd", range(1, 5))))
            # Reindexar mantém lacunas visíveis, sem ligar pontos sobre blocos ausentes.
            series = pd.Series(data.accuracy.to_numpy()*100, index=x).reindex(range(1, 25))
            acc.plot(series.index, series, "o-", ms=3, label=PROTOCOLS[protocol], color=color)
    acc.set(xlim=(.5, 24.5), ylim=(0, 103), ylabel="Acurácia (%)", xlabel="Bloco online (1–24; sessões 1–6)")
    if acc.lines:
        acc.legend(fontsize=8)
    else:
        acc.text(.5, .5, "Sem classificação online comparável", ha="center", transform=acc.transAxes)
    for row, kind in enumerate(("me", "im", "online"), 1):
        for col, band in enumerate(BANDS):
            ax = fig.add_subplot(grid[row, col])
            data = physiology.loc[(physiology.subject == subject) & (physiology.kind == kind) & (physiology.band == band)]
            if data.empty:
                ax.text(.5, .5, "Sem tentativas EDA válidas", ha="center", transform=ax.transAxes, fontsize=8)
            for label, color in (("left", "#0072B2"), ("right", "#D55E00")):
                part = data.loc[data.label == label].sort_values("observed_trial")
                if len(part):
                    ax.scatter(part.observed_trial, part.erd_percent, s=7, alpha=.25, color=color)
                    smooth = part.erd_percent.rolling(10, min_periods=1).mean()
                    ax.plot(part.observed_trial, smooth, lw=1, color=color,
                            label="Esquerda" if label == "left" else "Direita")
            ax.axhline(0, color="gray", lw=.7)
            for session, part in data.groupby("session"):
                if len(part):
                    start = part.observed_trial.min()
                    ax.axvline(start, color="gray", ls=":", alpha=.4, lw=.6)
                    ax.text(start, .96, f"s{int(session)}", transform=ax.get_xaxis_transform(), fontsize=6, va="top")
            ax.set(title=f"{KINDS[kind]} · {BANDS[band]} · N={len(data)}", ylabel="ERD/ERS (%)", xlabel="Tentativa observada (ordem cronológica)")
            ax.grid(alpha=.15)
            if row == 1 and col == 0 and len(data):
                ax.legend(fontsize=7)
            if data.empty:
                ax.set_axis_off()
    return fig


def write_summary(output, subjects, cfg, overwrite=False):
    """Somente sujeitos explícitos desta execução; nunca mistura pastas antigas por acaso."""
    output = Path(output)
    if not subjects or len(set(subjects)) != len(subjects) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in subjects):
        raise ValueError("Informe uma lista não vazia de sujeitos distintos para resumir.")
    if (output / "report.html").exists() and not overwrite:
        raise ValueError(f"Resumo {output / 'report.html'} já existe; use --overwrite ou outra saída.")
    loaded = {name: [] for name in ("scores", "trials", "bandpower")}
    coverage, signatures, synthetic, sources = [], [], [], []
    for subject in subjects:
        folder = output / subject
        info = json.loads((folder / "run-info.json").read_text())
        config = yaml.safe_load((folder / "config-used.yaml").read_text())
        with (folder / "tables.zip").open("rb") as stream:
            sources.append(dict(subject=subject, tables_sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
        scientific = {k: config[k] for k in ("input", "epochs", "classification", "spectral")}
        scientific["montage"] = {k: config["figures"][k] for k in ("montage", "montage_confirmed")}
        signatures.append(json.dumps(scientific, sort_keys=True))
        synthetic.append(bool(info.get("synthetic")))
        if len(signatures) == 1:
            summary_cfg = config.copy()
            summary_cfg["_synthetic"] = synthetic[-1]
        with ZipFile(folder / "tables.zip") as archive:
            inventory = pd.read_csv(archive.open("inventory.csv"))
            for name in loaded:
                wanted = {"subject", "session", "block", "trial_id", "trial", "label", "eda_status"} if name == "trials" else (
                    {"subject", "trial_id", "band", "channel", "erd_percent"} if name == "bandpower" else None)
                table = pd.read_csv(archive.open(name+".csv"), dtype={"subject": str},
                                    usecols=(lambda col: col in wanted) if wanted is not None else None)
                if len(table) and set(table.subject) != {subject}:
                    raise ValueError(f"Sujeito inconsistente em {folder}/{name}")
                loaded[name].append(table)
            coverage.append(dict(subject=subject, partial=bool(info.get("partial")),
                                 available=int(inventory.status.eq("available").sum()),
                                 omitted=int(inventory.status.str.startswith("skipped_").sum()),
                                 missing=int(inventory.status.eq("missing").sum())))
    if len(set(signatures)) > 1 or len(set(synthetic)) > 1:
        raise ValueError("Resumo exige configurações comparáveis e não mistura sinais reais e sintéticos.")
    cfg = summary_cfg
    if cfg["spectral"]["bands"].get("mu") != [8., 13.] or cfg["spectral"]["bands"].get("beta") != [13., 30.]:
        raise ValueError("Resumo alfa/beta requer mu=[8,13] e beta=[13,30] Hz, sem mudar bandas silenciosamente.")
    data = {name: pd.concat(tables, ignore_index=True) for name, tables in loaded.items()}
    if len(data["scores"]) and not data["scores"].pipeline.eq("TGSP+LR").all():
        raise ValueError("Resumo conjunto usa TGSP+LR. Reexecute com classification.pipelines: tgsp_lr.")
    blocks, accuracy, paired = accuracy_tables(data["scores"])
    physiology, slopes = physiology_tables(data["bandpower"], data["trials"], cfg["spectral"]["channels_of_interest"])
    acc_slopes = accuracy_trends(blocks)
    tests = summary_statistics(paired, slopes, cfg["classification"]["seed"], acc_slopes)
    coverage = pd.DataFrame(coverage)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".summary-", dir=output))
    report = Report(work, "Todos os sujeitos · TGSP+LR", cfg)
    try:
        report.text(f"{len(subjects)} sujeitos nesta execução. Modelo fixo: TGSP + regressão logística. "
                    "Compara protocolos, não escolhe o melhor classificador com base no teste. "
                    "Médias de acurácia dão o mesmo peso a cada bloco dentro do sujeito e a cada sujeito no grupo.")
        if coverage.partial.any():
            report.text("EXECUÇÃO PARCIAL: há gravações omitidas. Confira cobertura antes de interpretar as médias.")
        report.table("Cobertura dos dados", coverage.rename(columns=dict(subject="Sujeito", partial="Parcial", available="Processados", omitted="Omitidos", missing="Ausentes")))
        if len(accuracy):
            fig, (ax, trend_ax) = plt.subplots(1, 2, figsize=(12, 4))
            for i, protocol in enumerate(PROTOCOLS):
                values = accuracy.loc[accuracy.protocol == protocol, "accuracy"].to_numpy()*100
                if len(values):
                    ax.scatter(np.full(len(values), i), values, s=25, alpha=.6)
                    ax.text(i, 101, f"N={len(values)}", ha="center", fontsize=8)
                    low, high = _bootstrap(values, cfg["classification"]["seed"])
                    ax.plot([i-.15, i+.15], [values.mean()]*2, color="k", lw=2)
                    if np.isfinite(low):
                        ax.plot([i, i], [low, high], color="k", lw=2)
            ax.set_xticks(range(4), PROTOCOLS.values())
            ax.set(ylim=(0, 103), ylabel="Acurácia média por sujeito (%)")
            ax.tick_params(axis="x", labelrotation=20)
            common = matched_online(blocks)
            common["position"] = (common.session-1)*4+common.block.str[-1].map(dict(zip("abcd", range(1, 5))))
            for protocol, color in (("frozen", "#777777"), ("adaptive", "#0072B2")):
                means, lower, upper = np.full(24, np.nan), np.full(24, np.nan), np.full(24, np.nan)
                for position, part in common.groupby("position"):
                    values = part["accuracy_"+protocol].to_numpy()*100
                    means[int(position)-1] = values.mean()
                    lower[int(position)-1], upper[int(position)-1] = _bootstrap(values, cfg["classification"]["seed"])
                trend_ax.plot(range(1, 25), means, "o-", ms=3, label=PROTOCOLS[protocol], color=color)
                trend_ax.fill_between(range(1, 25), lower, upper, color=color, alpha=.15)
            trend_ax.set(xlim=(.5, 24.5), ylim=(0, 103), xlabel="Bloco online (1–24)", ylabel="Acurácia média entre sujeitos (%)")
            trend_ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
            if len(common):
                counts = common.groupby("position").subject.nunique()
                trend_ax.text(.02, .02, f"N por bloco: {counts.min()}–{counts.max()}", transform=trend_ax.transAxes, fontsize=8)
            trend_ax.legend(fontsize=7)
            report.figure(fig, "Acurácia por protocolo · TGSP+LR", "Cada ponto é um sujeito; traço preto: média e IC bootstrap de 95% entre sujeitos (mínimo 3). Coberturas podem diferir entre protocolos; os testes abaixo usam apenas sessões/blocos pareados.")
            for metric, title in (("accuracy", "Acurácia por sujeito (%)"), ("balanced_accuracy", "Acurácia balanceada por sujeito (%)")):
                display = accuracy.pivot(index="subject", columns="protocol", values=metric).reindex(columns=PROTOCOLS)*100
                display = display.rename(columns=PROTOCOLS).reset_index().rename(columns={"subject": "Sujeito"})
                if metric == "balanced_accuracy":
                    report.parts.append("<details><summary>Acurácia balanceada e cobertura por protocolo</summary>")
                report.table(title, display)
            counts = accuracy.pivot(index="subject", columns="protocol", values="n_blocks").reindex(columns=PROTOCOLS)
            report.table("Número de blocos por sujeito", counts.rename(columns=PROTOCOLS).reset_index())
            report.parts.append("</details>")
        else:
            report.text("Sem acurácias comparáveis. Em --eda-only não há treino nem avaliação de classificadores.")
        report.text("Estatística exploratória: dois contrastes de acurácia, duas tendências de acurácia online e seis de tendência ERD, "
                    "com correção de Holm na família planejada de 10 testes. Unidade independente: sujeito. "
                    "Troca de sinais bilateral (exata até 15 sujeitos; 10.000 permutações acima disso), "
                    "sob hipótese de simetria/permutabilidade entre sujeitos. Menos de 3 sujeitos: não testado. "
                    "ICs bootstrap reamostram sujeitos, nunca folds/canais/tentativas como se fossem sujeitos.")
        report.text("Acurácia: MI−ME usa somente as mesmas sessões; adaptativo−congelado usa somente os mesmos "
                    "blocos online de cada sujeito. Não inclui folds parciais de CV nem sequências adaptativas "
                    "explicitamente autorizadas com lacunas. Uma diferença de tarefa não é efeito causal de adaptação.")
        report.text("Tendência de acurácia: inclinação linear por sujeito sobre os mesmos blocos disponíveis nos dois "
                    "protocolos (mínimo 4), em pontos percentuais ao longo do intervalo observado. "
                    "Os testes usam as inclinações individuais; as curvas de grupo podem mudar de composição onde há dados faltantes.")
        names = {"mi_cv minus me_cv": "CV MI − CV ME", "adaptive minus frozen": "Adaptativo − congelado",
                 "Accuracy slope frozen": "Tendência acc: congelado", "Accuracy slope adaptive": "Tendência acc: adaptativo"}
        names.update({f"ERD slope {kind}/{band}": f"Tendência {KINDS[kind]} · {BANDS[band]}" for kind in KINDS for band in BANDS})
        display = tests[["test", "n_subjects", "effect_mean", "ci_low", "ci_high", "p_raw", "p_holm", "status"]].copy()
        display["test"] = display.test.map(names)
        display["status"] = display.status.map({"tested": "Testado", "insufficient_subjects": "Não testado (N<3)"})
        report.table("Testes estatísticos", display.rename(columns=dict(test="Contraste", n_subjects="N sujeitos", effect_mean="Efeito (p.p.)", ci_low="IC 95% inferior", ci_high="IC 95% superior", p_raw="p bruto", p_holm="p Holm", status="Resultado")))
        report.text("Eletrofisiologia: média dos ERD/ERS dos canais disponíveis da ROI "
                    + ", ".join(cfg["spectral"]["channels_of_interest"]) + ", em cada tentativa. "
                    "Alfa/mu=8–13 Hz e beta=13–30 Hz. Negativo=ERD; positivo=ERS. "
                    "ME, MI e online são analisados separadamente. Inclinação linear ajustada para a classe "
                    "da cue, em pontos percentuais ao longo da sequência observada (0–1); mínimo 8 tentativas "
                    "e 3 por classe. Não é tendência por tempo físico nem prova de aprendizagem/causalidade. "
                    "Inclinação ERD positiva significa ERS maior ou ERD menos negativo, não necessariamente melhora. "
                    "Só entram tentativas EDA aceitas com ERD finito; quantidades/canais constam das tabelas. "
                    "Ausências e rejeições podem causar seleção. Ganho, montagem e protocolo precisam ser comparáveis.")
        if len(slopes):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, band in zip(axes, BANDS):
                for i, kind in enumerate(("me", "im", "online")):
                    values = slopes.loc[(slopes.band == band) & (slopes.kind == kind), "slope_pp"].dropna()
                    ax.scatter(np.full(len(values), i), values, alpha=.6)
                    ax.text(i, .98, f"N={len(values)}", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
                    if len(values):
                        ax.plot([i-.15, i+.15], [values.mean()]*2, color="k", lw=2)
                ax.axhline(0, color="gray", lw=.8)
                ax.set_xticks(range(3), ["ME", "MI", "Online"])
                ax.set(title=BANDS[band], ylabel="Inclinação ERD/ERS (p.p./sequência)")
            report.figure(fig, "Tendências eletrofisiológicas por sujeito", "Cada ponto é uma inclinação estimada de um sujeito, ajustada para a classe da cue. A linha preta marca a média. Não se agregam tentativas de pessoas diferentes em um único teste.")
        report.parts.append("<details><summary>Estimativas por sujeito e comparações pareadas</summary>")
        report.table("Acurácia: comparações pareadas (p.p.)", paired)
        report.table("Acurácia: inclinações por sujeito (p.p./intervalo observado)", acc_slopes)
        report.table("ERD/ERS: inclinações por sujeito (p.p./sequência)", slopes)
        report.parts.append("</details>")
        for subject in subjects:
            report.parts.append(f"<details><summary>Verificação sucinta: {escape(subject)}</summary>")
            report.parts.append(f'<p><a href="{quote(subject)}/report.html">Relatório individual detalhado</a></p>')
            report.figure(_subject_figure(subject, blocks, physiology), f"{subject} · acurácia e alfa/beta", "Pontos: tentativas individuais; curvas ERD: média móvel descritiva das últimas 10 tentativas de cada classe, sem alterar a inferência. Online: os espaços indicam blocos sem avaliação. Nomes anatômicos permanecem provisórios quando a montagem não está confirmada.")
            report.parts.append("</details>")
        tables = dict(accuracy_by_block=blocks, accuracy_by_subject=accuracy, matched_accuracy=paired, accuracy_slopes=acc_slopes,
                      erd_trials=physiology, erd_slopes=slopes, statistics=tests, coverage=coverage)
        with ZipFile(work / "summary-tables.zip", "w", ZIP_DEFLATED) as archive:
            for name, table in tables.items():
                archive.writestr(name+".csv", table.to_csv(index=False))
        (work / "summary-info.json").write_text(json.dumps(dict(subjects=list(subjects), pipeline="TGSP+LR",
            synthetic=bool(synthetic and synthetic[0]), partial=bool(coverage.partial.any()),
            sources=sources, roi=cfg["spectral"]["channels_of_interest"],
            seed=cfg["classification"]["seed"], planned_tests=10), indent=2), encoding="utf-8")
        report.close()
        for path in work.iterdir():
            os.replace(path, output / path.name)
        print(f"Resumo de todos os sujeitos: {output / 'report.html'}", flush=True)
    finally:
        report.pdf.close()
        shutil.rmtree(work)
