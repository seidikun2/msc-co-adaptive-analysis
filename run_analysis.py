"""Ponto de entrada: inventário, EDA/classificação e demonstração sintética."""
from __future__ import annotations

import argparse
from dataclasses import replace
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

from eeg_io import BLOCKS, EmptyRecordingError, InvalidEEGError, discover, inspect_folder, load_config, read_recording
from eeg_processing import MissingAttentionError, event_table, prepare_block, spectral_summary, validate_config
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


def diagnose(table, cfg, destination, overwrite=False):
    """QC por arquivo, sem relatórios pesados; erros não impedem examinar os seguintes."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination} já existe; use outro --output ou --overwrite.")
    results = []
    for row in table.itertuples():
        item = dict(path=row.path, subject=row.subject, session=int(row.session), block=row.block)
        try:
            rec = read_recording(row.path, cfg)
            item["event_counts"] = {str(int(k)): int(v) for k, v in rec.events.code.value_counts().items()}
            block, trials, *_ = prepare_block(rec, cfg, dict(
                subject=row.subject, group=row.group, session=int(row.session), block=row.block))
            reasons = trials.loc[trials.status == "excluded", "reason"].value_counts().to_dict()
            eda_reasons = trials.loc[trials.eda_status == "excluded", "eda_reason"].value_counts().to_dict()
            item.update(status="ok", n_trials=len(trials), n_accepted=len(block.X),
                        rejection_reasons=reasons, n_eda_accepted=len(block.eda_trials),
                        eda_rejection_reasons=eda_reasons,
                        trials=json.loads(trials.to_json(orient="records", double_precision=10)))
            print(f"{row.subject} · s{row.session}/{row.block}: "
                  f"classificação {len(block.X)}/{len(trials)}; exclusões: {reasons}; "
                  f"EDA {len(block.eda_trials)}/{len(trials)}; exclusões EDA: {eda_reasons}", flush=True)
        except (ValueError, OSError, KeyError) as exc:
            item.update(status="error", error=str(exc))
            print(f"{row.path}: ERRO: {exc}", flush=True)
        results.append(item)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Diagnóstico salvo: {destination}", flush=True)
    if any(r["status"] == "error" for r in results):
        raise ValueError(f"Há arquivos com erro. Diagnóstico completo preservado em {destination}.")


def _draw_block(report, rec, raw, filtered, block, trials, events, summary, powers, cfg):
    from eeg_figures import raw_plot, timing_plot, spectral_plot, tfr_plot, topomap_plot
    title = f"Sessão {block.meta['session']} · {block.meta['block']}"
    report.text(f"{title}: classificação {len(block.X)}/{len(trials)}; "
                f"EDA {len(block.trials)}/{len(trials)} tentativas aceitas. "
                f"Exclusões classificação: {trials.loc[trials.status == 'excluded', 'reason'].value_counts().to_dict()}. "
                f"Exclusões EDA: {trials.loc[trials.eda_status == 'excluded', 'eda_reason'].value_counts().to_dict()}")
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
        report.text("Nenhuma tentativa aceita para a janela EDA configurada: espectros e ERD/ERS "
                    "não calculados para este bloco. Tentativas válidas para classificação são preservadas.")
        return
    report.figure(spectral_plot(summary, block, cfg), title+" · Espectro por classe",
                  "Welch por tentativa; média em potência linear antes de converter para dB. "
                  "Faixas: bootstrap de 95% entre tentativas, descritivo dentro deste sujeito. "
                  "Escala vertical ajustada apenas às frequências visíveis; compartilhada entre classes por canal.")
    report.figure(tfr_plot(summary, block, cfg), title+" · Espectrograma por classe",
                  "PSD tempo–frequência; média de dB entre tentativas. Eixo temporal relativo à cue. "
                  "Cores compartilhadas entre classes de cada canal; bordas sem dados ficam vazias. "
                  "Logo abaixo de cada mapa: curvas mu/beta do mesmo canal/classe, com bootstrap descritivo de 95%.")
    report.figure(tfr_plot(summary, block, cfg, relative=True), title+" · ERD/ERS tempo–frequência",
                  "100 × [P(t,f)/P_baseline(f) − 1], calculado em cada tentativa ANTES da média. "
                  "A baseline vem da atenção dessa tentativa. Negativo = ERD; positivo = ERS. "
                  "Saturação visual em ±100%; os valores das tabelas não são truncados. "
                  "Cor forte: clusters bilaterais de log-potência/baseline pós-cue, com correção por "
                  "tempo/frequência e Bonferroni entre canais × classes deste bloco. "
                  "Teste exploratório entre tentativas, sujeito à hipótese de simetria/troca de sinais; "
                  "não é inferência de grupo nem localiza precisamente início/frequência do efeito. "
                  "Teste não aplicado é indicado, não interpretado como ausência de efeito. "
                  "Curvas mu/beta abaixo de cada mapa integram potência antes da razão; ICs são descritivos. "
                  "Método, contagens e p corrigidos ficam em run-info.json.")
    for relative in (False, True):
        fig = topomap_plot(powers, block, cfg, relative)
        if fig is None:
            report.text("Mapa omitido: canais sem coordenadas na montagem configurada.")
            break
        report.figure(fig, title+(" · Mapa de ERD/ERS" if relative else " · Mapa de potência espectral"),
                      "Topografia do ESCALPO, não localização de fontes corticais. Escalas compartilhadas "
                      "entre classes dentro de cada banda. Potência: média linear → dB. ERD: média das "
                      "razões por tentativa. Interpolação limitada ao casco convexo dos eletrodos, sem cores fora dele. "
                      "Posições artificiais não permitem interpretação anatômica.")


def analyze_subject(table, cfg, overwrite=False, skip_empty=False, skip_missing_attention=False, skip_invalid_eeg=False):
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
    tfr_tests = []
    try:
        report.text("As tabelas completas estão em tables.zip. Há seis sessões esperadas, cada uma com "
                    "me, im e online a–d. Ausências são registradas, sem inventar gravações. "
                    "Seleção de figuras: "+cfg["figures"]["selection"]+"; métricas calculadas em todos os arquivos disponíveis.")
        report.text("Baseline: atenção + início configurado até cue − margem configurada. "
                    "Tentativas com baseline curta, ordem inválida, janela invadindo pausa ou artefatos "
                    "são excluídas com motivo registrado para cada finalidade separadamente. "
                    "Classificação: uma janela de 2 s por tentativa. "
                    "ERD é mudança relativa de potência; não é medida direta de ativação cortical.")
        for row in table.itertuples():
            meta = dict(subject=subject, group=row.group, session=int(row.session), block=row.block)
            print(f"{subject} · sessão {row.session} · {row.block}: lendo {Path(row.path).name}", flush=True)
            try:
                rec = read_recording(row.path, cfg)
            except (EmptyRecordingError, InvalidEEGError) as exc:
                empty = isinstance(exc, EmptyRecordingError)
                if not (skip_empty if empty else skip_invalid_eeg):
                    raise
                status = "skipped_empty" if empty else "skipped_invalid_eeg"
                label = "arquivo vazio omitido" if empty else "EEG inválido omitido"
                message = f"EXECUÇÃO PARCIAL — {label}: {exc}"
                print(message, flush=True)
                report.text(message)
                inventory.append(dict(**meta, path=row.path, sha256=_sha256(row.path),
                                      status=status, error=str(exc), n_trials=0 if empty else np.nan,
                                      validation_details=json.dumps(getattr(exc, "details", {}), ensure_ascii=False),
                                      n_accepted=0, n_eda_accepted=0, duplicate_paths=row.duplicate_paths,
                                      duplicate_entries=row.duplicate_entries, duplicate_sha256=row.duplicate_sha256))
                continue
            try:
                block, trials, raw, filtered, qc = prepare_block(rec, cfg, meta)
            except MissingAttentionError as exc:
                if not skip_missing_attention:
                    raise
                message = f"EXECUÇÃO PARCIAL — gravação sem atenção omitida: {exc}"
                print(message, flush=True)
                report.text(message)
                inventory.append(dict(**meta, path=row.path, sha256=_sha256(row.path),
                                      status="skipped_missing_attention", error=str(exc), n_trials=np.nan,
                                      n_accepted=0, n_eda_accepted=0,
                                      sfreq=rec.sfreq, n_channels=len(rec.channels),
                                      duration_s=len(rec.times)/rec.sfreq,
                                      event_counts=json.dumps({str(int(k)): int(v) for k, v in rec.events.code.value_counts().items()}),
                                      duplicate_paths=row.duplicate_paths, duplicate_entries=row.duplicate_entries,
                                      duplicate_sha256=row.duplicate_sha256))
                all_events.append(event_table(rec, cfg, meta, pd.DataFrame()))
                continue
            print(f"  Classificação: {len(block.X)}/{len(trials)}; "
                  f"EDA: {len(block.eda_trials)}/{len(trials)} tentativas aceitas", flush=True)
            reasons = trials.loc[trials.status == "excluded", "reason"].value_counts().to_dict()
            if reasons:
                print(f"  Motivos de exclusão classificação: {reasons}", flush=True)
            eda_reasons = trials.loc[trials.eda_status == "excluded", "eda_reason"].value_counts().to_dict()
            if eda_reasons:
                print(f"  Motivos de exclusão EDA: {eda_reasons}", flush=True)
            kind = "online" if row.block.startswith("online") else row.block
            draw = cfg["figures"]["selection"] == "all" or kind not in selected
            selected.add(kind)
            eda_block = replace(block, trials=block.eda_trials)
            summary, powers = spectral_summary(rec, filtered, eda_block, cfg, with_tfr=draw)
            if summary is not None and draw:
                tfr_tests.extend(dict(**meta, **result) for result in summary.get("tfr_tests", []))
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
                                  duplicate_paths=row.duplicate_paths,
                                  duplicate_entries=row.duplicate_entries,
                                  duplicate_sha256=row.duplicate_sha256,
                                  n_channels=len(block.channels), n_trials=len(trials), n_accepted=len(block.X),
                                  n_eda_accepted=len(block.eda_trials),
                                  duration_s=len(rec.times)/rec.sfreq, status="available"))
            if draw:
                _draw_block(report, rec, raw, filtered, eda_block, trials, events, summary, powers, cfg)
            blocks.append(block)
        if any(r["status"].startswith("skipped_") for r in inventory):
            report.parts.insert(1, "<p><strong>EXECUÇÃO PARCIAL: há gravações omitidas. "
                                "Confira os caminhos e motivos no inventário.</strong></p>")
        present = {(r["session"], r["block"]) for r in inventory}
        missing = [dict(subject=subject, group=table.group.iloc[0], session=s, block=b, status="missing")
                   for s in range(1, 7) for b in BLOCKS if (s, b) not in present]
        report.table("Inventário de arquivos disponíveis", pd.DataFrame(inventory).drop(columns=["path", "sha256"]))
        report.text("Blocos ausentes: "+", ".join(f"s{r['session']}/{r['block']}" for r in missing))
        print(f"{subject}: comparando classificadores…", flush=True)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=cfg["classification"].get("blas_threads", 1)):
            scores, predictions = classify_subject(blocks, cfg) if blocks else (pd.DataFrame(), pd.DataFrame())
        if not blocks:
            report.text("Nenhuma gravação utilizável neste sujeito; todos os arquivos foram omitidos. "
                        "Não há medidas EDA nem resultados de classificação.")
        report.text("CV offline: "+cfg["classification"]["cv_scheme"]+". Modelo padrão: TGSP+LR, "
                    "covariância Ledoit–Wolf → espaço tangente → StandardScaler → regressão logística "
                    "balanceada (C=1). Representação e escala são aprendidas só no treino de cada fold. "
                    "CV estratificada mede generalização dentro do bloco e pode ser otimista sob deriva temporal; "
                    "use cv_scheme=forward como sensibilidade. Não há seleção de hiperparâmetros no teste.")
        report.text("session_im_frozen: IM da mesma sessão → todos os online dessa sessão, sem atualizar. "
                    "Este é um comparador diagnóstico que pode usar IM2–IM6. adaptive_protocol: somente IM1 "
                    "inicializa; mantém o mesmo modelo no lote online e só então incorpora seus rótulos. "
                    "IM2–IM6 NUNCA entram nesse treino. Atualização é supervisionada, com rótulos conhecidos, "
                    f"e refaz o pipeline cumulativamente a cada {cfg['classification'].get('adaptive_update_blocks', 2)} "
                    "blocos online. Não reproduz o feedback histórico do participante.")
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
        for name, columns in dict(
            trials=["trial_id", "status", "reason", "eda_status", "eda_reason"],
            events=["trial_id", "code", "time_s"], channel_qc=["channel", "source_channel"],
            bandpower=["trial_id", "channel", "band", "baseline_power", "activity_power", "erd_percent"]
        ).items():
            if len(tables[name].columns) == 0:
                tables[name] = pd.DataFrame(columns=["subject", "group", "session", "block", *columns])
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
                          partial=any(r["status"].startswith("skipped_") for r in inventory),
                          tfr_tests=tfr_tests,
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
    parser.add_argument("command", choices=["inspect", "diagnose", "run", "summarize", "demo"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--data", help="Sobrescreve paths.data; aceita árvore com pastas de sujeitos.")
    parser.add_argument("--output", help="Sobrescreve paths.output; para demo é a pasta da demonstração.")
    parser.add_argument("--manifest", help="CSV explícito, opcional quando os nomes seguem a convenção.")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                        help="Acrescentar exclusão à busca automática; pode repetir. Nenhum arquivo é apagado.")
    parser.add_argument("--subject", help="Processar somente este sujeito.")
    parser.add_argument("--eda-only", action="store_true", help="Pular classificação.")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Em run, omitir CSVs vazios com registro de execução parcial; outros erros continuam fatais.")
    parser.add_argument("--skip-missing-attention", action="store_true",
                        help="Em run, omitir gravações sem o código de atenção, preservando marcadores e registro de execução parcial.")
    parser.add_argument("--skip-invalid-eeg", action="store_true",
                        help="Em run, omitir EEG insuficiente, NaN/Inf ou timestamps não crescentes, com diagnóstico no inventário.")
    parser.add_argument("--activity-window", type=float, nargs=2, metavar=("START", "STOP"),
                        help="Janela EDA fixa em segundos após o anchor; não altera os 2 s da classificação.")
    parser.add_argument("--overwrite", action="store_true", help="Substituir somente os arquivos de saída gerados.")
    args = parser.parse_args()
    if (args.skip_empty or args.skip_missing_attention or args.skip_invalid_eeg) and args.command != "run":
        raise ValueError("Opções --skip-* são exclusivas de run; inspect/diagnose sempre registram os erros.")
    cfg = load_config(args.config)
    if args.activity_window is not None:
        cfg["epochs"]["activity_s"] = args.activity_window
    if args.exclude:
        patterns = cfg["input"].get("exclude_globs") or []
        if not isinstance(patterns, list):
            raise ValueError("input.exclude_globs deve ser uma lista de nomes/padrões.")
        cfg["input"]["exclude_globs"] = patterns + args.exclude
    for name in ("data", "output", "manifest"):
        if getattr(args, name):
            cfg["paths"][name] = str(Path(getattr(args, name)).resolve())
    if args.command == "demo":
        make_demo(args.output or "demo", cfg)
    elif args.command == "summarize":
        from eeg_summary import write_summary
        output = Path(cfg["paths"]["output"])
        subjects = sorted(p.name for p in output.iterdir() if p.is_dir() and (p / "tables.zip").is_file())
        if args.subject:
            subjects = [s for s in subjects if s == args.subject]
        if not subjects:
            raise ValueError("Nenhum relatório individual encontrado para resumir.")
        write_summary(output, subjects, cfg, args.overwrite)
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
        if args.command == "diagnose":
            diagnose(table, cfg, Path(cfg["paths"]["output"]) / "diagnostics.json", args.overwrite)
        else:
            output = Path(cfg["paths"]["output"])
            if cfg["classification"].get("pipelines", "tgsp_lr") == "all" and (output / "report.html").exists():
                raise ValueError("Modo all exige outra saída para preservar o resumo TGSP+LR existente.")
            if (output / "report.html").exists() and not args.overwrite:
                raise ValueError(f"Resumo {output / 'report.html'} já existe; use outra saída ou --overwrite.")
            subjects = []
            for _, subject_table in table.groupby("subject", sort=True):
                analyze_subject(subject_table, cfg, args.overwrite, args.skip_empty, args.skip_missing_attention,
                                args.skip_invalid_eeg)
                subjects.append(str(subject_table.subject.iloc[0]))
            if cfg["classification"].get("pipelines", "tgsp_lr") == "tgsp_lr":
                from eeg_summary import write_summary
                write_summary(output, subjects, cfg, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)
