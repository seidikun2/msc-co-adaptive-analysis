"""Agrupamento e inferência sem pseudorreplicação; somente dados sintéticos."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

from eeg_io import load_config
from eeg_summary import (accuracy_tables, physiology_tables, subject_test,
                         correct_tests, summary_statistics, write_summary)


def scores_for(subject):
    rows = []
    for experiment, block, value in [('offline_cv', 'me', .6), ('offline_cv', 'im', .7),
                                      ('session_im_frozen', 'online_a', .5),
                                      ('session_im_frozen', 'online_b', .6),
                                      ('adaptive_protocol', 'online_b', .9),
                                      ('adaptive_protocol', 'online_c', .1)]:
        rows.append(dict(subject=subject, session=1, block=block, experiment=experiment,
                         pipeline='TGSP+LR', status='ok', fold='all', accuracy=value,
                         balanced_accuracy=value, coverage=1 if experiment == 'offline_cv' else np.nan))
    return pd.DataFrame(rows)


def physiology_for(subject):
    trials, powers = [], []
    for trial in range(1, 13):
        label = 'right' if trial % 2 == 0 else 'left'
        trial_id = f'{subject}/s01/im/{trial}'
        trials.append(dict(subject=subject, session=1, block='im', trial_id=trial_id,
                           trial=trial, label=label, eda_status='accepted'))
        for band in ('mu', 'beta'):
            for channel in ('C3', 'Cz', 'C4', 'Fz'):
                value = 5+10*(trial-1)/11+100*(label == 'right') if channel != 'Fz' else 1e9
                powers.append(dict(subject=subject, trial_id=trial_id, band=band, channel=channel, erd_percent=value))
    return pd.DataFrame(powers), pd.DataFrame(trials)


def test_accuracy_uses_oof_once_and_matches_identical_online_blocks():
    scores = pd.concat([scores_for(s) for s in ('A', 'B', 'C')], ignore_index=True)
    duplicate_fold = scores.iloc[[0]].assign(fold='1', accuracy=0.)
    blocks, by_subject, matched = accuracy_tables(pd.concat([scores, duplicate_fold]))
    assert len(blocks) == 18 and len(by_subject) == 12
    assert by_subject.loc[by_subject.protocol == 'me_cv', 'accuracy'].eq(.6).all()
    online = matched.loc[matched.comparison == 'adaptive minus frozen']
    assert online.n_matched.eq(1).all()
    np.testing.assert_allclose(online.difference_pp, 30.)
    with pytest.raises(ValueError, match='duplicados'):
        accuracy_tables(pd.concat([scores, scores.iloc[[0]]]))


def test_partial_cv_and_authorized_gaps_are_not_primary_group_evidence():
    scores = scores_for('A')
    scores.loc[scores.block == 'im', 'coverage'] = .5
    scores['partial_sequence'] = False
    scores.loc[scores.experiment == 'adaptive_protocol', 'partial_sequence'] = True
    blocks, _, pairs = accuracy_tables(scores)
    assert not blocks.protocol.isin(['mi_cv', 'adaptive']).any()
    assert pairs.empty


def test_erd_averages_channels_once_orders_trials_and_adjusts_cue_class():
    power, trials = physiology_for('A')
    values, slopes = physiology_tables(power.sample(frac=1, random_state=1),
                                      trials.sample(frac=1, random_state=2), ['C3', 'Cz', 'C4'])
    assert len(values) == 24 and values.n_channels.eq(3).all()
    np.testing.assert_allclose(slopes.slope_pp, 10., atol=1e-10)
    assert slopes.n_trials.eq(12).all()  # 12 trials, not 36 channel observations
    assert values.loc[values.trial == 1, 'progress'].eq(0).all()
    with pytest.raises(ValueError, match='duplicadas'):
        physiology_tables(pd.concat([power, power.iloc[[0]]]), trials, ['C3', 'Cz', 'C4'])


def test_sign_flip_is_exact_and_holm_covers_planned_tests():
    result = subject_test([10.]*10, 'paired')
    assert result['p_raw'] == 2/1024 and result['effect_mean'] == 10
    assert result['ci_low'] == result['ci_high'] == 10
    assert subject_test([1.], 'one')['status'] == 'insufficient_subjects'
    rows = [dict(p_raw=.01), dict(p_raw=.03), dict(p_raw=np.nan)]
    corrected = correct_tests(rows)
    np.testing.assert_allclose(corrected.p_holm[:2], [.03, .06])
    assert subject_test([0.]*4, 'zero')['p_raw'] == 1


def write_subject(output, subject, cfg):
    folder = output / subject
    folder.mkdir(parents=True)
    power, trials = physiology_for(subject)
    inventory = pd.DataFrame([dict(status='available')]*3+[dict(status='missing')]*33)
    with ZipFile(folder / 'tables.zip', 'w') as z:
        for name, table in dict(scores=scores_for(subject), trials=trials, bandpower=power, inventory=inventory).items():
            z.writestr(name+'.csv', table.to_csv(index=False))
    (folder / 'run-info.json').write_text(json.dumps(dict(synthetic=True, partial=False)))
    (folder / 'config-used.yaml').write_text(yaml.safe_dump(cfg))


def test_multisubject_report_is_self_contained_and_ignores_stale_folders(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
    for subject in ('A', 'B', 'C', 'STALE'):
        write_subject(tmp_path, subject, cfg)
    write_summary(tmp_path, ['A', 'B', 'C'], cfg)
    info = json.loads((tmp_path / 'summary-info.json').read_text())
    assert info['subjects'] == ['A', 'B', 'C'] and info['planned_tests'] == 10
    html = (tmp_path / 'report.html').read_text()
    assert 'STALE' not in html and 'data:image/png;base64,' in html
    assert '<details>' in html and 'A/report.html' in html
    with ZipFile(tmp_path / 'summary-tables.zip') as z:
        tests = pd.read_csv(z.open('statistics.csv'))
        assert len(tests) == 10
        paired = tests.loc[tests.test == 'adaptive minus frozen'].iloc[0]
        assert paired.n_subjects == 3 and paired.effect_mean == pytest.approx(30.)
    assert (tmp_path / 'figures.pdf').read_bytes().startswith(b'%PDF')
    with pytest.raises(ValueError, match='já existe'):
        write_summary(tmp_path, ['A'], cfg)
    assert (tmp_path / 'report.html').read_text() == html


def test_mixed_scientific_configurations_cannot_generate_a_summary(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
    write_subject(tmp_path, 'A', cfg)
    cfg['epochs']['activity_s'] = [.5, 3.5]
    write_subject(tmp_path, 'B', cfg)
    with pytest.raises(ValueError, match='configurações comparáveis'):
        write_summary(tmp_path, ['A', 'B'], cfg)
    assert not (tmp_path / 'report.html').exists()


def test_logistic_regression_is_the_actual_default_head():
    from eeg_classification import make_model, pipeline_specs
    from sklearn.linear_model import LogisticRegression
    cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
    assert pipeline_specs(cfg) == [('TGSP', 'LR')]
    model = make_model('TGSP', 'LR', 4, cfg)
    assert isinstance(model.steps[-1][1], LogisticRegression)
    assert model.steps[-1][1].class_weight == 'balanced'


def test_online_accuracy_trend_uses_same_blocks_for_both_protocols():
    from eeg_summary import accuracy_trends
    rows = []
    for protocol in ('frozen', 'adaptive'):
        for i, block in enumerate(('online_a', 'online_b', 'online_c', 'online_d')):
            rows.append(dict(subject='A', session=1, block=block, protocol=protocol,
                             accuracy=.4+i*.1 if protocol == 'adaptive' else .6))
    rows.append(dict(subject='A', session=2, block='online_a', protocol='adaptive', accuracy=0.))
    trends = accuracy_trends(pd.DataFrame(rows))
    assert trends.n_blocks.eq(4).all()
    assert trends.loc[trends.protocol == 'adaptive', 'slope_pp'].iloc[0] == pytest.approx(30.)
    assert trends.loc[trends.protocol == 'frozen', 'slope_pp'].iloc[0] == pytest.approx(0., abs=1e-10)


def test_orphan_power_is_not_silently_dropped():
    power, trials = physiology_for('A')
    with pytest.raises(ValueError, match='sem tentativa'):
        physiology_tables(power, trials.iloc[1:], ['C3', 'Cz', 'C4'])
