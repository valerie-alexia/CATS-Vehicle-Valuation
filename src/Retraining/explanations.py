"""Native exact TreeSHAP; sum one-hot contributions back to input fields."""
import numpy as np
from src.Training.pipeline import CATEGORICAL, NUMERIC, FEATURES

LABELS = {'Merk': 'Merek', 'Model': 'Model', 'Varian': 'Varian',
          'JenisMesin': 'Bahan bakar', 'JenisTransmisi': 'Transmisi',
          'TahunUnit': 'Tahun kendaraan', 'KapasitasMesin': 'Kapasitas mesin (cc)',
          'Kilometer': 'Kilometer (km)'}


def explain_branch(branch, row):
    prep, model = branch['preprocessor'], branch['model']
    X = prep.transform(row)
    if branch['algorithm'] == 'xgboost':
        from xgboost import DMatrix
        contributions = model.get_booster().predict(DMatrix(X), pred_contribs=True)[0]
    elif branch['algorithm'] == 'lightgbm':
        contributions = np.asarray(model.booster_.predict(X, pred_contrib=True))[0]
    else:
        raise ValueError('TreeSHAP hanya mendukung XGBoost dan LightGBM')
    contributions = np.asarray(contributions, dtype=float)
    features = branch.get('features', FEATURES)
    categorical = branch.get('categorical', CATEGORICAL)
    totals = {field: 0. for field in features}
    names = prep.get_feature_names_out()
    for name, value in zip(names, contributions[:-1]):
        raw_name = name.split('__', 1)[1]
        if name.startswith('category__'):
            field = next((c for c in sorted(categorical, key=len, reverse=True) if raw_name.startswith(c + '_')), None)
        else:
            field = raw_name.removeprefix('missingindicator_')
        if field not in totals:
            raise ValueError(f'Fitur SHAP tidak terpetakan: {name}')
        totals[field] += float(value)
    baseline = float(contributions[-1])
    raw_prediction = float(model.predict(X)[0])
    reconstructed = baseline + sum(totals.values())
    if not np.isclose(raw_prediction, reconstructed, atol=2e-4, rtol=1e-5):
        raise ValueError('Rekonstruksi TreeSHAP tidak cocok dengan prediksi model')
    items = []
    for field, contribution in sorted(totals.items(), key=lambda pair: -abs(pair[1])):
        value = row[field].iloc[0]
        value = None if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) else value
        if isinstance(value, np.generic):
            value = value.item()
        items.append({'fitur': field, 'label': LABELS.get(field, field), 'nilai_input': value,
                      'kontribusi_log_harga': contribution,
                      'arah': 'menaikkan' if contribution > 0 else 'menurunkan' if contribution < 0 else 'netral'})
    return {'metode': 'TreeSHAP native (agregasi per fitur input)', 'skala': 'log1p(harga)',
            'baseline_log_harga': baseline, 'baseline_harga': max(0., float(np.expm1(baseline))),
            'prediksi_log_harga': raw_prediction,
            'galat_rekonstruksi_log': abs(raw_prediction-reconstructed),
            'kontribusi_fitur': items,
            'catatan': 'Kontribusi menjelaskan perilaku model relatif terhadap nilai dasarnya, bukan hubungan sebab-akibat. '
                       'Kontribusi dijumlahkan pada skala log1p(harga), bukan dalam rupiah.'}
