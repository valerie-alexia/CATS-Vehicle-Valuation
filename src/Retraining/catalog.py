"""Closed, cascading demo options from the fitted bundle's CSV-derived catalog."""
from src.Retraining.input_values import OTHER, is_other

INPUT_ORDER = ['Merk', 'Model', 'Varian', 'TahunUnit', 'JenisMesin',
               'JenisTransmisi', 'KapasitasMesin', 'Kilometer']


def options_for(catalog, selected, field):
    candidates = catalog
    for previous, value in selected.items():
        if not is_other(value):
            candidates = candidates[candidates[previous].eq(value)]
    values = candidates[field].dropna().unique().tolist()
    values = [v for v in values if str(v).strip().upper() not in {'UNKNOWN', 'LAINNYA', ''}]
    values = sorted(values, reverse=field == 'TahunUnit')
    return values + [OTHER]


def display_option(value, field):
    if is_other(value):
        return OTHER
    if field in {'TahunUnit', 'KapasitasMesin', 'Kilometer'}:
        number = float(value)
        formatted = f'{number:,.0f}' if number.is_integer() else f'{number:,.1f}'
        if field == 'TahunUnit':
            return str(int(number))
        return formatted.replace(',', '.') + (' km' if field == 'Kilometer' else ' cc')
    return str(value)
