"""Ponto de entrada: inventário, EDA/classificação e demonstração sintética."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import yaml

from eeg_io import BLOCKS, discover, inspect_folder, load_config, read_recording
from eeg_processing import event_table, prepare_block, spectral_summary, validate_config
from eeg_classification import classify_subject


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concat(frames):
    nonempty = [f for f in frames if not f.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return frames[0].iloc[:0] if frames else pd.DataFrame()


def _draw_block(report, rec, raw, filtered, block, trials, events, summary, powers, cfg):
    from eeg_figures import (raw_plot, timing_plot, spectral_plot, tfr_plot,
                             erd_curve_plot, topomap_plot)
    title = f"Sessão {block.meta['session']} · {block.meta['block']}"
    report.text(f"{title}: {len(block.X)}/{len(trials)} tentativas aceitas. "
                f"Exclusões: {trials.loc[trials.status == 'excluded', 'reason'].value_counts().to_dict()}")
    report.figure(raw_plot(rec, raw, filtered, block, trials, cfg), title+" · EEG bruto e pré-processado",
                  "Mesma janela, canais e escala. Mediana removida por canal somente para exibição. "
                  "Filtro IIR causal contínuo e referência configurada; não há ICA nem correção de EOG. "
                  "Zero no início da execução/imagética da tentativa de referência. Sem marcador motor, "
                  "o eixo mostra o tempo absoluto no arquivo. Linhas: atenção em cinza, cue na cor da classe "
                  "e início motor em verde. O filtro causal introduz atraso de fase.")
    report.figure(timing_plot(trials, events), title+" · Organização temporal das tentativas",
                  "Uma janela de referência, com fundos coloridos e linhas verticais nos tempos dos eventos. "
                  "Zero no início da execução/imagética; atenção e cue anteriores têm tempos negativos. "
                  "Tempos exatos na tabela, relativos ao início motor e ao arquivo. A identidade esquerda/direita "
                  "vem exclusivamente da CUE, nunca do início motor. Sem marcador motor, a referência não é desenhada. "
                  "Todas as tentativas e marcações permanecem nas tabelas do ZIP.")
    if summary is None:
        report.text("Nenhuma tentativa aceita: espectros e ERD/ERS não calculados para este bloco.")
        return
    report.figure(spectral_plot(summary, block, cfg), title+" · Espectro por classe",
                  "Welch por tentativa; média em potência linear antes de converter para dB. "
                  "Faixas: bootstrap de 95% entre tentativas, descritivo dentro deste sujeito.")
    report.figure(tfr_plot(summary, block, cfg), title+" · Espectrograma por classe",
                  "PSD tempo–frequência; média de dB entre tentativas. Eixo temporal relativo à cue. "
                  "Cores compartilhadas entre classes/canais; bordas sem dados ficam vazias.")
    report.figure(tfr_plot(summary, block, cfg, relative=True), title+" · ERD/ERS tempo–frequência",
                  "100 × [P(t,f)/P_baseline(f) − 1], calculado em cada tentativa ANTES da média. "
                  "A baseline vem da atenção dessa tentativa. Negativo = ERD; positivo = ERS. "
                  "Saturação visual em ±100%; os valores das tabelas não são truncados.")
    report.figure(erd_curve_plot(summary, block, cfg), title+" · ERD/ERS nas bandas motoras",
                  "Potência integrada na banda antes da razão por baseline. Área cinza: intervalo mediano "
                  "da baseline; linha vertical: cue. Faixas: bootstrap de 95% entre tentativas.")
    for relative in (False, True):
        fig = topomap_plot(powers, block, cfg, relative)
        if fig is None:
            report.text("Mapa omitido: canais sem coordenadas na montagem configurada.")
            break
        report.figure(fig, title+(" · Mapa de ERD/ERS" if relative else " · Mapa de potência espectral"),
                      "Topografia do ESCALPO, não localização de fontes corticais. Escalas compartilhadas "
                      "entre classes dentro de cada banda. Potência: média linear → dB. ERD: média das "
                      "razões por tentativa. Posições artificiais não permitem interpretação anatômica.")


def analyze_subject(table, cfg, overwrite=False):
    from eeg_figures import Report, classification_figures
    subject = str(table.subject.iloc[0])
    root = Path(cfg["paths"]["output"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / subject
    if target.exists() and not overwrite:
        raise ValueError(f"Saída {target} já existe. Use --overwrite para substituir os arquivos gerados.")
    work = Path(tempfile.mkdtemp(prefix=f".{subject}-", dir=root))
    report = Report(work, subject, cfg)
    blocks, all_trials, all_events, all_qc, all_power, inventory, selected = [], [], [], [], [], [], set()
    try:
        report.text("As tabelas completas estão em tables.zip. Há seis sessões esperadas, cada uma com "
                    "me, im e online a–d. Ausências são registradas, sem inventar gravações. "
                    "Seleção de figuras: "+cfg["figures"]["selection"]+"; métricas calculadas em todos os arquivos disponíveis.")
        report.text("Baseline: atenção + início configurado até cue − margem configurada. "
                    "Tentativas com baseline curta, ordem inválida, janela invadindo pausa ou artefatos "
                    "são excluídas com motivo registrado. Classificação: uma janela de 2 s por tentativa. "
                    "ERD é mudança relativa de potência; não é medida direta de ativação cortical.")
        for row in table.itertuples():
            meta = dict(subject=subject, group=row.group, session=int(row.session), block=row.block)
            print(f"{subject} · sessão {row.session} · {row.block}: lendo {Path(row.path).name}", flush=True)
            rec = read_recording(row.path, cfg)
            block, trials, raw, filtered, qc = prepare_block(rec, cfg, meta)
            kind = "online" if row.block.startswith("online") else row.block
            draw = cfg["figures"]["selection"] == "all" or kind not in selected
            selected.add(kind)
            summary, powers = spectral_summary(rec, filtered, block, cfg, with_tfr=draw)
            events = event_table(rec, cfg, meta, trials)
            qc["position_confirmed"] = cfg["figures"]["montage_confirmed"]
            original = cfg["input"].get("channels") or rec.channels
            mapping = dict(zip(rec.channels, original))
            qc["source_channel"] = qc.channel.map(mapping)
            all_qc.append(qc.assign(**meta))
            all_trials.append(trials.assign(**meta))
            all_events.append(events)
            all_power.append(powers.assign(**meta, position_confirmed=cfg["figures"]["montage_confirmed"]))
            inventory.append(dict(**meta, path=row.path, sha256=_sha256(row.path), sfreq=rec.sfreq,
                                  n_channels=len(block.channels), n_trials=len(trials), n_accepted=len(block.X),
                                  duration_s=len(rec.times)/rec.sfreq, status="available"))
            if draw:
                _draw_block(report, rec, raw, filtered, block, trials, events, summary, powers, cfg)
            blocks.append(block)
            print(f"  {len(block.X)}/{len(trials)} tentativas aceitas", flush=True)
        present = {(r["session"], r["block"]) for r in inventory}
        missing = [dict(subject=subject, group=table.group.iloc[0], session=s, block=b, status="missing")
                   for s in range(1, 7) for b in BLOCKS if (s, b) not in present]
        report.table("Inventário de arquivos disponíveis", pd.DataFrame(inventory).drop(columns=["path", "sha256"]))
        report.text("Blocos ausentes: "+", ".join(f"s{r['session']}/{r['block']}" for r in missing))
        print(f"{subject}: comparando classificadores…", flush=True)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=cfg["classification"].get("blas_threads", 1)):
            scores, predictions = classify_subject(blocks, cfg)
        report.text("CV offline: "+cfg["classification"]["cv_scheme"]+". TGSP = covariância Ledoit–Wolf → "
                    "espaço tangente riemanniano; CSP usa as mesmas covariâncias regularizadas. "
                    "Ambos usam escala aprendida no treino e LDA com shrinkage ou SVM de parâmetros fixos. "
                    "CV estratificada mede generalização dentro do bloco e pode ser otimista sob deriva temporal; "
                    "use cv_scheme=forward como sensibilidade. Não há seleção de hiperparâmetros no teste.")
        report.text("session_im_frozen: IM da mesma sessão → todos os online dessa sessão, sem atualizar. "
                    "Este é um comparador diagnóstico que pode usar IM2–IM6. adaptive_protocol: somente IM1 "
                    "inicializa; prever online 1a → incorporar 1a → prever 1b → … → prever 2a após 1d. "
                    "IM2–IM6 NUNCA entram nesse treino. Atualização é supervisionada, com rótulos conhecidos, "
                    "e refaz todo o pipeline cumulativamente. Não reproduz o feedback histórico do participante.")
        if cfg["classification"]["allow_protocol_gaps"]:
            report.text("ATENÇÃO: allow_protocol_gaps=true. A sequência é parcial e usa apenas os blocos "
                        "observados; não representa a reprodução integral do protocolo.")
        classification_figures(scores, predictions, report)
        if not scores.empty:
            columns = ["session", "block", "experiment", "pipeline", "status", "reason", "n_test",
                       "accuracy", "balanced_accuracy", "n_train", "train_blocks"]
            report.table("Resultados por bloco (fold=all); skips não são acertos nem falhas",
                         scores.loc[scores.fold == "all", [c for c in columns if c in scores]])
        tables = dict(inventory=pd.DataFrame(inventory+missing), trials=_concat(all_trials),
                      events=_concat(all_events), channel_qc=_concat(all_qc),
                      bandpower=_concat(all_power), scores=scores, predictions=predictions)
        # Mesmo com classificação desligada, manter CSVs legíveis com cabeçalho.
        if tables["scores"].empty:
            tables["scores"] = pd.DataFrame(columns=["subject", "group", "session", "block", "experiment",
                                          "pipeline", "fold", "status", "reason", "accuracy", "balanced_accuracy"])
        if tables["predictions"].empty:
            tables["predictions"] = pd.DataFrame(columns=["subject", "group", "session", "block", "experiment",
                                               "pipeline", "fold", "trial_id", "label", "predicted", "correct"])
        with ZipFile(work / "tables.zip", "w", compression=ZIP_DEFLATED) as archive:
            for name, frame in tables.items():
                archive.writestr(name+".csv", frame.to_csv(index=False))
        public_cfg = {k: v for k, v in cfg.items() if k != "_config_dir"}
        (work / "config-used.yaml").write_text(yaml.safe_dump(public_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        provenance = dict(python=sys.version, command=sys.argv, synthetic=bool(cfg.get("_synthetic")),
                          versions={name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib",
                                     "scikit-learn", "mne", "pyriemann", "PyYAML")},
                          input_kind="synthetic" if cfg.get("_synthetic") else "experimental",
                          note="Hashes no inventário identificam os dados processados; montagem/unidades no config.")
        # hashes do inventário identificam os arquivos efetivamente processados, inclusive em execuções locais.
        (work / "run-info.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        report.close()
        target.mkdir(exist_ok=True)
        for path in work.iterdir():
            os.replace(path, target / path.name)
        print(f"Concluído: {target / 'report.html'}", flush=True)
    finally:
        report.pdf.close()
        shutil.rmtree(work)


def make_demo(output, base_cfg):
    """Gerar exemplo SINTÉTICO separado, sem resultados atribuídos a sujeitos reais."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"{output} já existe; escolha outro destino para a demonstração.")
    data = output / "data" / "SYNTHETIC"
    data.mkdir(parents=True)
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    cfg.pop("_config_dir", None)
    cfg["paths"] = dict(data="data", output="results", manifest=None)
    cfg["input"].update(eeg_unit="uV", sfreq=128, filename_regex=None,
                        channels=[f"Channel {i}" for i in range(1, 17)])
    cfg["events"].update(attention=[33024], task_start=[33027])
    cfg["_synthetic"] = True
    cfg["figures"]["montage_confirmed"] = False
    rng = np.random.default_rng(42)
    fs, n_trials = 128, 8
    for session in (1, 2):
        for block in BLOCKS:
            t = np.arange(int((3+n_trials*9)*fs))/fs
            n = len(t)
            x = rng.normal(0, 1.8, (n, 16))
            for ch in range(16):
                x[:, ch] += (600*ch + 25*np.sin(2*np.pi*.2*t+ch) +
                              6*np.sin(2*np.pi*10*t+rng.uniform(0, 6)) +
                              3*np.sin(2*np.pi*21*t+rng.uniform(0, 6)) +
                              2*np.sin(2*np.pi*50*t))
            ids, dates, durations = [""]*n, [""]*n, [""]*n
            def mark(when, event):
                idx = round(when*fs)
                ids[idx], dates[idx], durations[idx] = str(event), str(when), "0"
            mark(.25, 32769)
            for i in range(n_trials):
                att = 3+9*i
                cue, task, rest = att+2, att+2.5, att+7
                label = 769 if i % 2 == 0 else 770
                for when, event in ((att, 33024), (cue, label), (task, 33027), (rest, 32777)):
                    mark(when, event)
                active = (t >= task) & (t < rest)
                channel = 10 if label == 769 else 6
                x[active, channel] += (10+session)*np.sin(2*np.pi*(12 if label == 769 else 18)*t[active])
            frame = pd.DataFrame(x, columns=cfg["input"]["channels"])
            frame.insert(0, "Time:128Hz", t)
            frame.insert(1, "Epoch", np.arange(n)//4)
            frame["Aux 1"], frame["Aux 2"] = 50000., -50000.
            frame["Event Id"], frame["Event Date"], frame["Event Duration"] = ids, dates, durations
            name = (f"motor_mi_SYNTHETIC_online_{session}{block[-1]}.csv" if block.startswith("online") else
                    f"motor_{'mi' if block == 'im' else 'me'}_SYNTHETIC_{session}.csv")
            frame.to_csv(data / name, index=False, float_format="%.8f")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Dados SINTÉTICOS gerados. Execute: python run_analysis.py run --config {output / 'config.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "run", "demo"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--data", help="Sobrescreve paths.data; aceita árvore com pastas de sujeitos.")
    parser.add_argument("--output", help="Sobrescreve paths.output; para demo é a pasta da demonstração.")
    parser.add_argument("--manifest", help="CSV explícito, opcional quando os nomes seguem a convenção.")
    parser.add_argument("--subject", help="Processar somente este sujeito.")
    parser.add_argument("--eda-only", action="store_true", help="Pular classificação.")
    parser.add_argument("--overwrite", action="store_true", help="Substituir somente os arquivos de saída gerados.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in ("data", "output", "manifest"):
        if getattr(args, name):
            cfg["paths"][name] = str(Path(getattr(args, name)).resolve())
    if args.command == "demo":
        make_demo(args.output or "demo", cfg)
    elif args.command == "inspect":
        target = Path(cfg["paths"]["output"])
        target.mkdir(parents=True, exist_ok=True)
        destination = target / "inventory.csv"
        if destination.exists() and not args.overwrite:
            raise ValueError(f"{destination} já existe; use outro --output ou --overwrite.")
        inventory, markers = inspect_folder(cfg)
        inventory.to_csv(destination, index=False)
        markers.to_csv(target / "markers.csv", index=False)
        print(inventory.to_string(index=False))
        print(f"Inventário salvo: {destination}")
        if (inventory.error != "").any():
            raise ValueError("Há arquivos com erro no inventário; confira separator/colunas/timestamps.")
    else:
        cfg["classification"]["enabled"] &= not args.eda_only
        validate_config(cfg)
        table = discover(cfg)
        if args.subject:
            table = table.loc[table.subject == args.subject]
            if table.empty:
                raise ValueError(f"Sujeito não encontrado: {args.subject}")
        for _, subject_table in table.groupby("subject", sort=True):
            analyze_subject(subject_table, cfg, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)
