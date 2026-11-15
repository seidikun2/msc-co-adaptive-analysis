"""Leitura OpenViBE, inventário e organização explícita de sessões/blocos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import yaml


@dataclass
class Recording:
    path: Path
    times: np.ndarray
    signal: np.ndarray          # canais × amostras, na unidade declarada pelo arquivo
    channels: list[str]
    sfreq: float
    events: pd.DataFrame       # code, time_s, duration_s, source_row


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_dir"] = str(path.parent)
    for key in ("data", "output", "manifest"):
        value = cfg["paths"].get(key)
        if value:
            cfg["paths"][key] = str((path.parent / value).resolve())
    return cfg


def code(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def _normal(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def read_frame(path, cfg):
    options = cfg["input"]
    sep = options["separator"]
    if sep == "auto":
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            sample = f.read(16384)
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            sep = ","
    # Strings nos eventos: não perder códigos compostos, como 769:32779.
    header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
    event_cols = [c for c in header if _normal(c).startswith("event")]
    event_cols += [options[k] for k in ("event_id_column", "event_date_column",
                                      "event_duration_column") if options.get(k)]
    frame = pd.read_csv(path, sep=sep, decimal=options["decimal"],
                        dtype={c: str for c in event_cols}, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _column(frame, configured, names, required=False):
    if configured:
        if configured not in frame:
            raise ValueError(f"Coluna configurada ausente: {configured}")
        return configured
    found = [c for c in frame if any(_normal(c).startswith(n) for n in names)]
    if len(found) > 1:
        raise ValueError(f"Colunas ambíguas: {found}. Configure o nome explicitamente.")
    if not found and required:
        raise ValueError(f"Coluna não encontrada: {names}")
    return found[0] if found else None


def _tokens(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [s.strip() for s in str(value).split(":") if s.strip()]


def read_recording(path, cfg):
    frame = read_frame(path, cfg)
    options = cfg["input"]
    time_col = _column(frame, options.get("time_column"), ["time"], required=True)
    id_col = _column(frame, options.get("event_id_column"), ["eventid"])
    date_col = _column(frame, options.get("event_date_column"), ["eventdate"])
    duration_col = _column(frame, options.get("event_duration_column"), ["eventduration"])
    meta = {time_col, id_col, date_col, duration_col}
    channels = options.get("channels") or [
        c for c in frame if c not in meta and _normal(c) not in {"epoch", "samplingrate"}
        and not str(c).startswith("Unnamed:")]
    if not channels or any(c not in frame for c in channels):
        raise ValueError("Seleção de canais vazia ou com colunas ausentes.")
    signals = frame[channels].apply(pd.to_numeric, errors="raise")
    times = pd.to_numeric(frame[time_col], errors="raise")
    event_rows = []
    if id_col:
        for row, val in frame[id_col].dropna().items():
            ids = _tokens(val)
            if not ids:
                continue
            dates = _tokens(frame.at[row, date_col]) if date_col else []
            durations = _tokens(frame.at[row, duration_col]) if duration_col else []
            mode = options["event_time"]
            if mode not in {"row", "absolute", "relative_to_row"}:
                raise ValueError("input.event_time inválido")
            if mode != "row" and len(dates) != len(ids):
                raise ValueError(f"Linha {row}: números de Event Id e Event Date diferem.")
            if durations and len(durations) != len(ids):
                raise ValueError(f"Linha {row}: números de Event Id e Event Duration diferem.")
            for j, item in enumerate(ids):
                event_time = float(times.iloc[row]) if mode == "row" else float(dates[j])
                if mode == "relative_to_row":
                    event_time += float(times.iloc[row])
                event_rows.append((code(item), event_time,
                                   float(durations[j]) if durations else 0., row))
    events = pd.DataFrame(event_rows, columns=["code", "time_s", "duration_s", "source_row"])
    # OpenViBE pode repetir um evento em linhas de buffers. Remover só duplicatas exatas.
    events = events.drop_duplicates(["code", "time_s", "duration_s"]).sort_values("time_s")
    if len(events) and (not np.isfinite(events[["time_s", "duration_s"]]).all().all()
                        or (events.duration_s < 0).any()):
        raise ValueError("Eventos com tempos não finitos ou durações negativas.")
    # Linhas contendo só eventos são permitidas; amostras parcialmente faltantes não.
    valid = ~signals.isna().all(axis=1)
    x = signals.loc[valid].to_numpy(dtype=float).T
    t = times.loc[valid].to_numpy(dtype=float)
    if len(t) < 3 or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("EEG insuficiente, NaN/Inf ou amostras parcialmente ausentes.")
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("Timestamps do EEG não são estritamente crescentes.")
    inferred = 1. / np.median(dt)
    pin = re.search(r"([\d.]+)\s*Hz", time_col, flags=re.I)
    fs = float(options.get("sfreq") or (pin.group(1) if pin else inferred))
    if abs(inferred / fs - 1.) > .01 or np.max(np.abs(dt * fs - 1.)) > .05:
        raise ValueError(f"Timestamps descontínuos ou taxa incompatível: cabeçalho={fs:g}, "
                         f"timestamps={inferred:g} Hz. Não interpolar silenciosamente.")
    if len(events) and ((events.time_s < t[0] - 1/fs).any()
                       or (events.time_s > t[-1] + 1/fs).any()):
        raise ValueError("Evento fora da gravação. Confira input.event_time e Event Date.")
    names = options.get("channel_names")
    if names:
        if len(names) != len(channels):
            raise ValueError("channel_names deve ter o mesmo tamanho de channels.")
        channels = list(names)
    if len(set(channels)) != len(channels):
        raise ValueError("Nomes de canais duplicados.")
    return Recording(Path(path), t, x, list(channels), fs, events.reset_index(drop=True))


BLOCKS = ["me", "im", "online_a", "online_b", "online_c", "online_d"]

OPENVIBE_NAMES = {768: "GDF_Start_Of_Trial", 769: "GDF_Left", 770: "GDF_Right",
                  800: "GDF_End_Of_Trial", 32769: "ExperimentStart", 32770: "ExperimentStop",
                  32773: "TrialStart", 32774: "TrialStop", 32775: "BaselineStart",
                  32776: "BaselineStop", 32777: "RestStart", 32778: "RestStop"}


def identify(path, cfg):
    custom = cfg["input"].get("filename_regex")
    if custom:
        match = re.search(custom, Path(path).name, re.I)
        return match.groupdict() if match else None
    offline = re.fullmatch(r"motor_(me|mi)_([A-Za-z0-9_-]+)_([1-6])\.csv", Path(path).name, re.I)
    online = re.fullmatch(r"motor_mi_([A-Za-z0-9_-]+)_online_([1-6])([abcd])\.csv", Path(path).name, re.I)
    if offline:
        return dict(subject=offline[2], session=offline[3], block="me" if offline[1].lower() == "me" else "im")
    if online:
        return dict(subject=online[1], session=online[2], block="online_"+online[3].lower())
    return None


def discover(cfg):
    root = Path(cfg["paths"]["data"]).resolve()
    manifest = cfg["paths"].get("manifest")
    if manifest:
        table = pd.read_csv(manifest, dtype={"subject": str, "group": str})
        required = {"path", "subject", "session", "block"}
        if not required <= set(table):
            raise ValueError(f"Manifest precisa de colunas {sorted(required)}")
        table = table.dropna(subset=["path"]).copy()
        table["path"] = table.path.map(lambda p: str((root / p).resolve()))
    else:
        rows, unknown = [], []
        for path in sorted(root.rglob("*.csv")):
            fields = identify(path, cfg)
            if fields:
                rows.append(dict(path=str(path.resolve()), **fields))
            else:
                unknown.append(str(path.relative_to(root)))
        if unknown:
            raise ValueError("Arquivos sem sessão/bloco inequívocos. Use `inspect` e um "
                             f"manifest, sem adivinhar a numeração: {unknown[:8]}")
        table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"Nenhuma gravação encontrada em {root}")
    if table[["subject", "session", "block"]].isna().any().any():
        raise ValueError("Preencha subject, session e block para TODAS as linhas do manifest.")
    sessions = pd.to_numeric(table.session, errors="raise")
    if not sessions.isin(range(1, 7)).all():
        raise ValueError("session deve ser um inteiro de 1 a 6.")
    table["session"] = sessions.astype(int)
    table["block"] = table.block.str.lower()
    if not table.block.isin(BLOCKS).all():
        raise ValueError(f"block deve ser um de {BLOCKS}; use im para imagética.")
    if table.duplicated(["subject", "session", "block"]).any() or table.path.duplicated().any():
        raise ValueError("Manifest com bloco ou arquivo duplicado.")
    if not table.path.map(lambda p: Path(p).is_file()).all():
        raise ValueError("Arquivo do manifest não encontrado.")
    if not table.subject.str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("subject só pode conter letras, números, '_' e '-'.")
    if "group" not in table:
        table["group"] = "unspecified"
    table["group"] = table.group.fillna("unspecified")
    if (table.groupby("subject").group.nunique() > 1).any():
        raise ValueError("Um sujeito está atribuído a mais de um grupo.")
    table["_order"] = table.block.map(dict(zip(BLOCKS, range(len(BLOCKS)))))
    return table.sort_values(["subject", "session", "_order"]).drop(columns="_order")


def inspect_folder(cfg):
    """Lê um arquivo de cada vez; não exige unidades, códigos de atenção ou manifest."""
    root = Path(cfg["paths"]["data"]).resolve()
    rows, markers = [], []
    for path in sorted(root.rglob("*.csv")):
        row = dict(path=str(path.relative_to(root)), subject="", session="", block="", group="")
        row.update(identify(path, cfg) or {})
        try:
            rec = read_recording(path, cfg)
            markers.append(rec.events.assign(path=row["path"],
                           standard_name=rec.events.code.map(OPENVIBE_NAMES).fillna("não mapeado")))
            row.update(sfreq=rec.sfreq, duration_s=len(rec.times)/rec.sfreq,
                       n_channels=len(rec.channels), channels=";".join(rec.channels),
                       event_counts=";".join(f"{int(k)} (0x{int(k):04X}):{v}"
                                             for k, v in rec.events.code.value_counts().items()),
                       error="")
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            row["error"] = str(exc)
        rows.append(row)
        print(f"Inspecionado: {path.name}", flush=True)
    if not rows:
        raise ValueError(f"Nenhum CSV em {root}")
    return pd.DataFrame(rows), pd.concat(markers, ignore_index=True) if markers else pd.DataFrame()
