"""Version-pinned public OOF stacking and fixed-grid blends. Never submits."""
from pathlib import Path
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

H = Path(__file__).resolve().parent
R = H.parents[1]
P = H / 'public'
ID, TARGET = 'id', 'Will_Buy_EV'
CORE = {
    'naji_blend': ('01_blend_oof.csv', '01_submission.csv'),
    'naji_lgb1': ('Pure LGBM_V1_oof.csv', 'Pure LGBM_V1_test.csv'),
    'naji_lgb3': ('Pure LGBM_V3_oof.csv', 'Pure LGBM_V3_test.csv'),
    'sergey_lgb': ('Sergey_LGBM_oof.csv', 'Sergey_LGBM_submission.csv'),
    'naji_xgb10': ('XGBoost_Triple_TE_10folds_oof.csv', 'XGBoost_Triple_TE_10folds_test.csv'),
    'naji_xgb5': ('XGBoost_Triple_TE_5folds_oof.csv', 'XGBoost_Triple_TE_5folds_test.csv'),
}
SIX = ['A_lgbm_triple_te_digits_3seed', 'B_xgb_on_A_features',
       'C_no_digits_windows_lift_sm2_30_300', 'D_no_exact_key_ladder_windows',
       'E_ladder25_250_2500_lift_sm5_50_500', 'F_exact_rate_as_init_score', 'ensemble']
REAL = ['G_realmlp_10fold', 'G_realmlp_3seed']
WEIGHTS = [.05, .10, .25, .50, .75, 1.0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def rank(values):
    return rankdata(np.asarray(values), method='average') / len(values)


def read_aligned(path, ids):
    df = pd.read_csv(path, float_precision='round_trip')
    assert df[ID].is_unique and len(df) == len(ids), path
    assert set(df[ID]) == set(ids), path
    return df.set_index(ID).loc[ids].reset_index()


def export(filename, ids, pred):
    destination = R / filename
    assert len(pred) == len(ids) and np.isfinite(pred).all()
    assert np.min(pred) >= 0 and np.max(pred) <= 1
    frame = pd.DataFrame({ID: ids, TARGET: pred})
    encoded = frame.to_csv(index=False, lineterminator='\n').encode()
    if destination.exists():
        assert destination.read_bytes() == encoded, 'Refuse changed overwrite'
    else:
        destination.write_bytes(encoded)
    df = pd.read_csv(destination, float_precision='round_trip')
    assert df.columns.tolist() == [ID, TARGET]
    assert np.array_equal(df[ID], ids) and df[ID].is_unique
    assert np.array_equal(df[TARGET].to_numpy(), pred)
    assert np.array_equal(rank(df[TARGET]), rank(pred))
    return {'path': str(destination), 'sha256': sha(destination), 'rows': len(df),
            'public_score': None, 'range': [float(np.min(pred)), float(np.max(pred))]}


def load_inputs():
    train = pd.read_csv(R / 'train.csv', usecols=[ID, TARGET])
    test = pd.read_csv(R / 'test.csv', usecols=[ID])
    sample = pd.read_csv(R / 'sample_submission.csv', usecols=[ID])
    assert np.array_equal(test[ID], sample[ID])
    ids, tids = train[ID].to_numpy(), test[ID].to_numpy()
    assert set(train[TARGET]) == {'Yes', 'No'}
    y = train[TARGET].eq('Yes').to_numpy().astype(np.int8)
    fold10 = np.empty(len(y), dtype=np.int8)
    for f, (_, ix) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(y, y)):
        fold10[ix] = f
    used = [R / 'train.csv', R / 'test.csv', H / 'run.py', H / 'PLAN.md', H / 'EXPERIMENT.md']
    oo, tt = {}, {}
    for directory in P.iterdir():
        if not directory.is_dir():
            continue
        download_manifest = directory / '_downloads.json'
        if download_manifest.exists():
            manifest = json.loads(download_manifest.read_text(encoding='utf-8'))
            for name, record in manifest.items():
                path = directory / name
                assert sha(path) == record['sha256'], path
            used.append(download_manifest)
    folder = P / 'najiama__s6e9-oof'
    for name, (of, tf) in CORE.items():
        for filename, row_ids, dest in [(of, ids, oo), (tf, tids, tt)]:
            path = folder / filename
            df = read_aligned(path, row_ids)
            cols = [c for c in df if c != ID]
            assert len(cols) == 1, path
            dest[name] = df[cols[0]].to_numpy(float)
            used.append(path)
    folder = P / 'megayak__s6e9-six-feature-views-oof-library'
    for label, columns in [('six_views', SIX), ('realmlp_g', REAL)]:
        of, tf = folder / f'oof_{label}.csv', folder / f'test_{label}.csv'
        a, b = read_aligned(of, ids), read_aligned(tf, tids)
        assert np.array_equal(a[TARGET].to_numpy(), y)
        assert np.array_equal(a['fold'].to_numpy(), fold10), 'Public fold mismatch'
        assert set(b.columns) == {ID, *columns}
        for c in columns:
            oo[c], tt[c] = a[c].to_numpy(float), b[c].to_numpy(float)
        used.extend([of, tf])

    def member(folder, model):
        out = np.full(len(y), np.nan)
        tests = []
        for f in range(10):
            path = R / 'work' / folder / 'runs' / model / f'fold{f}.npz'
            with np.load(path) as z:
                ix = np.where(fold10 == f)[0]
                assert np.array_equal(z['indices'], ix)
                assert np.array_equal(z['valid_ids'], ids[ix])
                assert np.array_equal(z['test_ids'], tids)
                out[ix] = z['valid']
                tests.append(z['test'])
            used.append(path)
        tst = np.mean(tests, axis=0)
        # Saved arrays are independently reconstructed from ID-bearing caches.
        for name, value in [('oof_model.npy', out), ('test_model.npy', tst)]:
            path = R / 'work' / folder / name
            assert np.array_equal(np.load(path), value), path
            used.append(path)
        return out, tst

    actual_path = R / 'submission_round25.csv'
    baseline = np.load(R / 'work/round25/oof_proxy.npy')
    actual = read_aligned(actual_path, tids)[TARGET].to_numpy()
    used.extend([actual_path, R / 'work/round25/oof_proxy.npy'])
    assert sha(actual_path) == 'bab79dd454074f214aabad50e6e75a9db48c606794bd984446ef627ece5c6b81'
    o22, t22 = member('round22', 'tabm_ple')
    o25, t25 = member('round25', 'recipe_margin')
    o19_path, t19_path = R / 'work/round19/oof_candidate.npy', R / 'work/round19/test_candidate.npy'
    o19, t19 = np.load(o19_path), np.load(t19_path)
    used.extend([o19_path, t19_path])
    o22mix, t22mix = .9*rank(o19)+.1*rank(o22), .9*rank(t19)+.1*rank(t22)
    assert np.array_equal(.75*rank(o22mix)+.25*rank(o25), baseline)
    aligned = rank(.75*rank(t22mix)+.25*rank(t25))
    public_names = list(oo)
    oo['own_round25'], tt['own_round25'] = baseline, actual
    for name, folder, model in [('own_reference', 'round26_reference', 'exact_te_margin'),
                                ('own_residual', 'round27', 'residual_keys'),
                                ('own_periodic', 'round23', 'tabm_periodic')]:
        oo[name], tt[name] = member(folder, model)
    assert len(public_names) == 15 and len(oo) == 19
    for name in oo:
        assert len(oo[name]) == len(y) and len(tt[name]) == len(tids)
        for v in [oo[name], tt[name]]:
            assert np.isfinite(v).all() and len(np.unique(v)) > 100
        oo[name], tt[name] = rank(oo[name]), rank(tt[name])
    ta = dict(tt)
    ta['own_round25'] = rank(aligned)
    hashes = {str(p.relative_to(R)): sha(p) for p in used}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {'fingerprint': fingerprint, 'sha256': hashes, 'public_columns': public_names,
                'local_columns': [x for x in oo if x not in public_names],
                'rows': len(ids), 'test_rows': len(tids), 'public_fold10_matches': True,
                'najiama_fold_labels_available': False}
    saved = H / 'input_manifest.json'
    if saved.exists():
        assert json.loads(saved.read_text(encoding='utf-8')) == manifest, 'Frozen inputs changed'
    else:
        dump(saved, manifest)
    np.savez(H / 'inputs.npz', ids=ids, test_ids=tids, y=y, fold=fold10,
             columns=np.array(list(oo)), X=np.column_stack(list(oo.values())),
             T=np.column_stack(list(tt.values())), T_aligned=np.column_stack(list(ta.values())))
    return y, ids, tids, fold10, oo, tt, ta, public_names, fingerprint


def fit_meta(name, columns, y, oo, tt, ta, fingerprint):
    folder = H / name
    folder.mkdir(exist_ok=True)
    X = np.column_stack([oo[c] for c in columns])
    T = np.column_stack([tt[c] for c in columns])
    A = np.column_stack([ta[c] for c in columns])
    out = np.empty(len(y))
    tests, aligned_tests, log = [], [], []
    for f, (fit, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        path = folder / f'fold{f}.npz'
        info = folder / f'fold{f}.json'
        if path.exists() and info.exists():
            rec = json.loads(info.read_text())
            assert rec['fingerprint'] == fingerprint and rec['columns'] == columns
            with np.load(path) as z:
                assert np.array_equal(z['indices'], val)
                out[val] = z['valid']
                tests.append(z['test']); aligned_tests.append(z['test_aligned'])
            log.append(rec)
            print(name, 'cached fold', f, flush=True)
            continue
        start = time.perf_counter()
        scaler = StandardScaler().fit(X[fit])
        model = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1200,
                                   tol=1e-5, random_state=42+f, n_jobs=1)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            model.fit(scaler.transform(X[fit]), y[fit])
        assert not any(issubclass(w.category, ConvergenceWarning) for w in notices), 'Meta fit did not converge'
        out[val] = model.predict_proba(scaler.transform(X[val]))[:, 1]
        tests.append(model.predict_proba(scaler.transform(T))[:, 1])
        aligned_tests.append(model.predict_proba(scaler.transform(A))[:, 1])
        rec = {'fingerprint': fingerprint, 'columns': columns, 'fold': f,
               'seconds': time.perf_counter()-start, 'iterations': int(model.n_iter_[0]),
               'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
               'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0])}
        np.savez(path, indices=val, valid=out[val], test=tests[-1], test_aligned=aligned_tests[-1])
        dump(info, rec); log.append(rec)
        print(name, 'fold', f, 'iterations', rec['iterations'], 'seconds', round(rec['seconds'], 1), flush=True)
    assert np.isfinite(out).all()
    test_pred, aligned_pred = np.mean(tests, axis=0), np.mean(aligned_tests, axis=0)
    np.save(folder / 'oof.npy', out)
    np.save(folder / 'test.npy', test_pred)
    np.save(folder / 'test_aligned.npy', aligned_pred)
    dump(folder / 'fit.json', log)
    return rank(out), rank(test_pred), rank(aligned_pred)


def main():
    threadpool_limits(limits=6)
    y, ids, tids, folds, oo, tt, ta, public_names, fingerprint = load_inputs()
    print('Input contracts and exact public fold assignments verified.', flush=True)
    candidates = {}
    for name, names in [('public15', public_names), ('public15_local4', list(oo))]:
        candidates[name] = fit_meta(name, names, y, oo, tt, ta, fingerprint)
    make_public = lambda d: rank(.57*d['ensemble']+.29*d['G_realmlp_3seed']+.14*d['naji_xgb10'])
    make_local = lambda d: rank((d['own_reference']+d['own_residual']+d['own_periodic'])/3)
    candidates['public_bundle'] = tuple(make_public(d) for d in [oo, tt, ta])
    candidates['local_bundle'] = tuple(make_local(d) for d in [oo, tt, ta])
    base, btest, baligned = oo['own_round25'], tt['own_round25'], ta['own_round25']
    screen, confirm = folds < 3, folds >= 3
    auc = lambda v, mask: float(roc_auc_score(y[mask], v[mask]))
    rows = []
    for name, (o, t, a) in candidates.items():
        for w in WEIGHTS:
            pred = (1-w)*base+w*o
            rows.append({'name': name, 'weight': w, 'screen_auc': auc(pred, screen),
                         'screen_delta': auc(pred, screen)-auc(base, screen)})
    # Freeze the screen choice before computing any confirmation AUC.
    best = {'name': 'baseline', 'weight': 0.0, 'screen_auc': auc(base, screen), 'screen_delta': 0.0}
    for row in rows:
        if row['screen_auc'] > best['screen_auc']+1e-7 or (
            abs(row['screen_auc']-best['screen_auc']) <= 1e-7 and row['weight'] < best['weight']):
            best = dict(row)
    choice = {'fingerprint': fingerprint, 'selected': best, 'screen': rows}
    path = H / 'choice.json'
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        dump(path, choice)
    print('FROZEN CHOICE', best, flush=True)
    allrows = np.ones(len(y), dtype=bool)
    for row in rows:
        o = candidates[row['name']][0]
        pred = (1-row['weight'])*base+row['weight']*o
        row.update(auc=auc(pred, allrows), delta=auc(pred, allrows)-auc(base, allrows),
                   confirmation_delta=auc(pred, confirm)-auc(base, confirm),
                   fold_deltas=[auc(pred, folds == f)-auc(base, folds == f) for f in range(10)])
        row['fold_wins'] = int(sum(v > 0 for v in row['fold_deltas']))
    pd.DataFrame(rows).to_csv(H / 'weight_curve.csv', index=False)
    source_metrics = {name: {'oof_auc': auc(values, allrows),
                            'rank_correlation_with_round25': float(np.corrcoef(base, values)[0, 1])}
                      for name, values in oo.items()}
    dump(H / 'source_metrics.json', source_metrics)
    if best['name'] == 'baseline':
        new = base, btest, baligned
    else:
        new = candidates[best['name']]
    w = best['weight']
    out, test_out, aligned_out = tuple((1-w)*b+w*n for b, n in zip([base,btest,baligned], new))
    np.save(H / 'oof_proxy.npy', out)
    np.save(H / 'test_main.npy', rank(test_out))
    np.save(H / 'test_aligned.npy', rank(aligned_out))
    main_export = export('submission_round28.csv', tids, rank(test_out))
    aligned_export = export('submission_round28_aligned.csv', tids, rank(aligned_out))
    public_export = export('submission_round28_public_stack.csv', tids, candidates['public15'][1])
    fold_delta = [auc(out, folds == f)-auc(base, folds == f) for f in range(10)]
    delta, cdelta = auc(out, allrows)-auc(base, allrows), auc(out, confirm)-auc(base, confirm)
    report = {'selected': best, 'fingerprint': fingerprint,
              'baseline_proxy_auc': auc(base, allrows), 'proxy_auc': auc(out, allrows),
              'delta': delta, 'confirmation_delta': cdelta, 'fold_deltas': fold_delta,
              'passed_local_gate': bool(delta > 1e-5 and cdelta > 0 and sum(d>0 for d in fold_delta)>=7),
              'candidate_oof_auc': {k: auc(v[0], allrows) for k, v in candidates.items()},
              'exports': [main_export, aligned_export, public_export],
              'main_submission_joint_oof_auc': None,
              'aligned_composition_proxy_auc': auc(out, allrows),
              'caveats': ['Round25 test/proxy historical lineage differs; aligned export repairs the known Round19/20 substitution only.',
                          'Historical model selection and cross-row label dependencies remain; meta OOF is not fully nested base-model retraining.',
                          'No new Kaggle score has been measured.']}
    dump(H / 'final_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
