"""Presentation-only categorical options; never rewrite model categories.

The original catalog and API helper remain unchanged. Numeric values are not
dropdown dependencies: real years, kilometres and cc need not occur in fitting.
"""
from src.Retraining.catalog import options_for
from src.Retraining.input_values import is_other


HIDDEN_VARIANTS = frozenset({
    ('HONDA', 'HR-V', 'SE CUMA 290CASH'),
    ('HONDA', 'HR-V', 'SE MERIAH'),
    ('HONDA', 'HR-V', 'ATAU 3 BARIS'),
    ('HONDA', 'BRIO', 'BEBAS BANJIR MESIN HALUS'),
})

# Only upstream categories constrain subsequent widgets. Ignore stale downstream
# choices after changing brand/model, and never constrain categories by numbers.
DEPENDENCIES = {
    'Merk': (),
    'Model': ('Merk',),
    'Varian': ('Merk', 'Model'),
    'JenisMesin': ('Merk', 'Model', 'Varian'),
    'JenisTransmisi': ('Merk', 'Model', 'Varian', 'JenisMesin'),
}


def _key(value):
    return '' if value is None else str(value).strip().casefold()


_HIDDEN_KEYS = frozenset(tuple(_key(value) for value in row) for row in HIDDEN_VARIANTS)


def options_for_display(catalog, selected, field):
    """Return original category values, hiding only four reviewed variant tuples.

    Matching is case-insensitive, but returned values retain their exact spelling.
    Supports categorical fields only; numeric fields belong in number inputs.
    ``Lainnya`` is preserved even when no visible variant remains. No catalog,
    payload, model, or official API options are mutated by this helper.
    """
    if field not in DEPENDENCIES:
        raise ValueError('Dropdown tampilan hanya untuk fitur kategorikal; gunakan number input untuk numerik')
    candidates = catalog
    for previous in DEPENDENCIES[field]:
        value = selected.get(previous)
        if value is None or is_other(value):
            continue
        candidates = candidates.loc[candidates[previous].map(_key).eq(_key(value))]
    if field == 'Varian':
        visible = [tuple(_key(value) for value in row) not in _HIDDEN_KEYS
                   for row in candidates[['Merk', 'Model', 'Varian']].itertuples(index=False, name=None)]
        candidates = candidates.loc[visible]
    # Reuse official ordering, missing-category exclusion and Lainnya sentinel.
    return options_for(candidates, {}, field)
