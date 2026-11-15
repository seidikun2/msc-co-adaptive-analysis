"""Testes científicos/temporais; todos os sinais aqui são SINTÉTICOS."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eeg_classification as classification
from eeg_io import Recording, discover, load_config, read_recording
from eeg_processing import EpochBlock, prepare_block, preprocess, spectral_summary, trial_table


@pytest.fixture
def cfg():
    cfg = load_config(Path(__file__).parents[1] / "config.yaml")
    cfg["input"].update(eeg_unit="uV", sfreq=128)
    cfg["events"].update(attention=[33024], task_start=[33027])
    return cfg


def recording(cfg, n_trials=6):
    fs = 128
    t = np.arange((3+9*n_trials)*fs)/fs
    rng = np.random.default_rng(12)
    x = rng.normal(0, 3, (16, len(t)))
    events = []
    for i in range(n_trials):
        a = 3+i*9
        for when, marker in ((a, 33024), (a+2, 769+i % 2), (a+2.5, 33027), (a+7, 32777)):
            events.append((marker, when, 0., 0))
    return Recording(Path("synthetic.csv"), t, x, list(cfg["input"]["channel_names"]), fs,
                     pd.DataFrame(events, columns=["code", "time_s", "duration_s", "source_row"]))


def test_openvibe_events_preserve_actual_dates_and_exclude_aux(tmp_path, cfg):
    t = np.arange(128)/128
    data = pd.DataFrame({"Time:128Hz": t, "Epoch": np.arange(128)//4,
                         **{f"Channel {i}": np.sin(t*i) for i in range(1, 17)},
                         "Aux 1": 999999, "Aux 2": -999999,
                         "Event Id": "", "Event Date": "", "Event Duration": ""})
    data.loc[2, ["Event Id", "Event Date", "Event Duration"]] = ["0x0301:770", ".0234241611:.5", "0:0"]
    path = tmp_path / "file.csv"
    data.to_csv(path, index=False)
    rec = read_recording(path, cfg)
    assert len(rec.channels) == 16
    assert rec.sfreq == 128
    assert rec.events.code.tolist() == [769, 770]
    np.testing.assert_allclose(rec.events.time_s, [.0234241611, .5])
    assert rec.signal.max() < 2


def test_discontinuous_signal_is_not_silently_interpolated(tmp_path, cfg):
    t = np.delete(np.arange(128)/128, 10)
    frame = pd.DataFrame({"Time:128Hz": t, **{f"Channel {i}": t for i in range(1, 17)}})
    path = tmp_path / "gap.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="descontínuos"):
        read_recording(path, cfg)


def test_confirmed_filename_convention_and_order(tmp_path, cfg):
    sub = tmp_path / "cohort" / "EG102"
    sub.mkdir(parents=True)
    for name in ("motor_mi_EG102_online_1b.csv", "motor_mi_EG102_2.csv",
                 "motor_me_EG102_1.csv", "motor_mi_EG102_online_1a.csv"):
        (sub / name).touch()
    cfg["paths"].update(data=str(tmp_path), manifest=None)
    result = discover(cfg)
    assert list(zip(result.session, result.block)) == [(1, "me"), (1, "online_a"), (1, "online_b"), (2, "im")]
    assert set(result.subject) == {"EG102"}


def test_baseline_is_attention_until_cue_and_window_is_two_seconds(cfg):
    rec = recording(cfg)
    block, trials, *_ = prepare_block(rec, cfg, dict(subject="SYNTHETIC", group="demo", session=1, block="im"))
    assert len(block.X) == 6
    assert block.X.shape == (6, 16, 256)
    np.testing.assert_allclose(trials.baseline_start_s, trials.attention_s+.2)
    np.testing.assert_allclose(trials.baseline_end_s, trials.cue_s-.2)
    assert (trials.baseline_end_s < trials.cue_s).all()


def test_short_baselines_and_rest_overlap_are_excluded(cfg):
    rec = recording(cfg)
    rec.events.loc[rec.events.code == 769, "time_s"] -= 1.2
    table = trial_table(rec, cfg)
    assert (table.loc[table.label == "left", "reason"] == "short_attention_baseline").all()
    rec = recording(cfg)
    rec.events.loc[rec.events.code == 32777, "time_s"] -= 3
    assert (trial_table(rec, cfg).reason == "activity_overlaps_rest_or_next_trial").all()


def test_filters_do_not_use_future_samples(cfg):
    rec = recording(cfg)
    original = preprocess(rec, cfg)
    rec.signal[:, rec.times >= 20] += 1000
    changed = preprocess(rec, cfg)
    for i in (1, 2):
        np.testing.assert_array_equal(original[i][:, rec.times < 20], changed[i][:, rec.times < 20])


def test_native_units_do_not_apply_physical_thresholds(cfg):
    rec = recording(cfg)
    rec.signal *= 1000
    meta = dict(subject="SYNTHETIC", group="demo", session=1, block="im")
    physical, *_ = prepare_block(rec, cfg, meta)
    assert len(physical.X) == 0
    cfg["input"]["eeg_unit"] = "native"
    native, trials, _, _, qc = prepare_block(rec, cfg, meta)
    assert len(native.X) == 6
    assert not trials.physical_amplitude_qc_applied.any()
    assert set(qc.amplitude_unit) == {"u.a."}


def test_class_identity_comes_only_from_cue(cfg):
    rec = recording(cfg)
    table = trial_table(rec, cfg)
    assert table.label.tolist() == ["left", "right"]*3
    assert table.cue_code.tolist() == [769, 770]*3
    assert set(table.class_source) == {"cue"}
    # O mesmo código motor é usado nas duas classes e não pode definir a identidade.
    assert set(rec.events.loc[rec.events.code == 33027, "time_s"]) == set(table.task_start_s)


def test_reference_timeline_uses_motor_onset_zero_background_and_lines(cfg):
    from eeg_figures import timing_plot
    from eeg_processing import event_table
    import matplotlib.pyplot as plt
    rec = recording(cfg)
    meta = dict(subject="SYNTHETIC", group="demo", session=1, block="im")
    _, trials, *_ = prepare_block(rec, cfg, meta)
    events = event_table(rec, cfg, meta, trials)
    fig = timing_plot(trials, events)
    assert len(fig.axes[0].patches) >= 4
    assert len(fig.axes[0].lines) >= 4
    assert not fig.axes[0].collections # sem scatter
    cells = fig.axes[1].tables[0].get_celld()
    # Atenção em 3 s; cue em 5 s; início motor em 5,5 s no arquivo.
    assert cells[(1, 2)].get_text().get_text() == "-2.5000000000"
    assert cells[(2, 2)].get_text().get_text() == "-0.5000000000"
    assert cells[(3, 2)].get_text().get_text() == "0.0000000000"
    assert cells[(3, 3)].get_text().get_text() == "5.5000000000"
    assert "[-2.3000000000, -0.7000000000]" in fig.texts[0].get_text()
    text = " ".join(cell.get_text().get_text() for cell in cells.values())
    assert "Cue — mão esquerda" in text
    assert fig.axes[0].get_xlim()[0] < 0
    np.testing.assert_allclose(events.loc[events.event_name == "attention", "from_task_start_s"], -2.5)
    np.testing.assert_allclose(events.loc[events.event_name == "left", "from_task_start_s"], -.5)
    np.testing.assert_allclose(events.loc[events.event_name == "task_start", "from_task_start_s"], 0)
    plt.close(fig)


def test_reference_does_not_substitute_cue_when_motor_onset_is_missing(cfg):
    from eeg_figures import timing_plot
    from eeg_processing import event_table
    import matplotlib.pyplot as plt
    rec = recording(cfg)
    cfg["events"]["task_start"] = []
    meta = dict(subject="SYNTHETIC", group="demo", session=1, block="im")
    _, trials, *_ = prepare_block(rec, cfg, meta)
    fig = timing_plot(trials, event_table(rec, cfg, meta, trials))
    assert "Configure events.task_start" in fig.axes[0].texts[0].get_text()
    assert not fig.axes[0].lines
    plt.close(fig)


def test_erd_is_minus_75_percent_when_amplitude_halves(cfg):
    fs = 128
    times = np.arange(4*fs)/fs
    x = np.tile(10e-6*np.sin(2*np.pi*10*times), (16, 1))
    x[:, times >= 2] *= .5
    rec = Recording(Path("synthetic"), times, x, cfg["input"]["channel_names"], fs, pd.DataFrame())
    trials = pd.DataFrame([dict(trial_id="synthetic/1", trial=1, label="left",
                                baseline_start_s=0., baseline_end_s=2., anchor_s=2.)])
    block = EpochBlock({}, rec.channels, fs, np.empty((1, 16, 256)), trials)
    cfg["epochs"]["activity_s"] = [0, 2]
    _, powers = spectral_summary(rec, x, block, cfg)
    np.testing.assert_allclose(powers.loc[powers.band == "mu", "erd_percent"], -75, atol=1e-10)


@pytest.mark.parametrize("feature", ["TGSP", "CSP"])
@pytest.mark.parametrize("classifier", ["LDA", "SVM-linear", "SVM-rbf"])
def test_real_pipelines_decode_independent_synthetic_covariances(cfg, feature, classifier):
    rng = np.random.default_rng(27)
    X = rng.normal(size=(60, 4, 256))*1e-6
    y = np.array(["left", "right"]*30)
    X[y == "left", 0] *= 3
    X[y == "right", 1] *= 3
    model = classification.make_model(feature, classifier, 4, cfg)
    model.fit(X[:40], y[:40])
    assert (model.predict(X[40:]) == y[40:]).mean() >= .9
    if feature == "TGSP":
        ref = model.named_steps["tangentspace"].reference_.copy()
        model.predict(X[40:]*10)
        np.testing.assert_array_equal(ref, model.named_steps["tangentspace"].reference_)


def toy_block(session, block, identifier):
    trials = pd.DataFrame(dict(trial_id=[f"s{session}/{block}/{i}" for i in range(4)],
                               trial=range(1, 5), label=["left", "right"]*2))
    # Cada sinal carrega uma identidade para auditar exatamente os conjuntos do .fit.
    X = np.stack([np.full((4, 256), identifier+i) for i in range(4)])
    return EpochBlock(dict(subject="SYNTHETIC", group="demo", session=session, block=block),
                      ["C3", "C4", "Cz", "Fz"], 128, X, trials)


def spy_factory(log):
    class Spy:
        def fit(self, X, y):
            self.ids = set(X[:, 0, 0])
            log.append(self.ids)
            return self

        def predict(self, X):
            assert not self.ids & set(X[:, 0, 0]), "Treino e teste compartilham tentativas"
            return np.array(["left"]*len(X))
    return lambda *args: Spy()


def test_adaptive_protocol_excludes_im2_and_current_online_test(cfg, monkeypatch):
    fits = []
    monkeypatch.setattr(classification, "make_model", spy_factory(fits))
    blocks = [toy_block(1, "im", 100), toy_block(2, "im", 200)]
    blocks += [toy_block(1, "online_"+letter, 110+10*i) for i, letter in enumerate("abcd")]
    blocks += [toy_block(2, "online_a", 210), toy_block(2, "online_b", 220)]
    scores, predictions = classification.classify_subject(blocks, cfg)
    adaptive = scores.loc[(scores.experiment == "adaptive_protocol") & (scores.session == 2)]
    assert adaptive.iloc[0].train_blocks == "s01/im;s01/online_a;s01/online_b;s01/online_c;s01/online_d"
    assert adaptive.iloc[0].n_train == 20
    assert "s02/online_a" in adaptive.iloc[1].train_blocks
    assert not adaptive.train_blocks.str.contains("s02/im").any()
    frozen = scores.loc[(scores.experiment == "session_im_frozen") & (scores.session == 2)]
    assert set(frozen.train_blocks) == {"s02/im"}
    assert len(predictions.loc[predictions.experiment == "adaptive_protocol"]) == 24


def test_protocol_does_not_jump_missing_blocks(cfg, monkeypatch):
    monkeypatch.setattr(classification, "make_model", spy_factory([]))
    blocks = [toy_block(1, "im", 100), toy_block(1, "online_a", 110), toy_block(1, "online_c", 130)]
    scores, _ = classification.classify_subject(blocks, cfg)
    row = scores.loc[(scores.experiment == "adaptive_protocol") & (scores.block == "online_c")].iloc[0]
    assert row.status == "skipped"
    assert "online_b" in row.reason


def test_protocol_never_initializes_from_im2(cfg, monkeypatch):
    monkeypatch.setattr(classification, "make_model", spy_factory([]))
    scores, _ = classification.classify_subject([toy_block(2, "im", 200), toy_block(2, "online_a", 210)], cfg)
    assert scores.loc[scores.experiment == "adaptive_protocol", "reason"].tolist() == ["missing_session_1_im"]
