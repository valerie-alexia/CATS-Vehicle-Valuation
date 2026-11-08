"""Immutable April aggregates, owned by one cached bundle/reference identity.

Same normalization, groupby(dropna=True), median, sorted-first mode and per-field
fallback as experiment_granular.enrich. No vehicles, predictions or request data
are cached. New DataFrames are created for each enrichment call.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from src.Training import experiment_granular as exp


@dataclass(frozen=True)
class PreparedReference:
    identity: tuple
    levels: tuple[Mapping, ...]


def prepare_reference(reference, identity):
    ref = exp.normalize_keys(reference[exp.LEVELS[0] + exp.SPECS])
    for c in exp.EXTRA_NUM:
        ref[c] = pd.to_numeric(ref[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        ref[c] = ref[c].where(ref[c] > 0)
    for c in exp.EXTRA_CAT:
        ref[c] = exp.clean_category(ref[c])
    lookups = []
    for keys in exp.LEVELS:
        table = ref.groupby(keys, dropna=True)[exp.EXTRA_NUM].median()
        for c in exp.EXTRA_CAT:
            # Equivalent to dropna().mode().iloc[0] for normalized strings:
            # largest frequency, lexicographically smallest value on a tie.
            # Vectorized counts avoid a Python Series.mode call for every group.
            counts = ref.groupby(keys + [c], dropna=True).size().rename('_count').reset_index()
            winners = counts.sort_values(['_count', c], ascending=[False, True], kind='stable')
            winners = winners.drop_duplicates(keys).set_index(keys)[c]
            table[c] = winners.reindex(table.index)
        # Only immutable scalar tuples are retained, never a shared DataFrame.
        lookup = {tuple(key): tuple(values) for key, values in
                  zip(table.index.tolist(), table[exp.SPECS].itertuples(index=False, name=None))}
        lookups.append(MappingProxyType(lookup))
    return PreparedReference(tuple(identity), tuple(lookups))


def enrich_prepared(target, prepared):
    result = exp.normalize_keys(target[exp.LEVELS[0]]).reset_index(drop=True)
    values = pd.DataFrame({c: pd.Series(np.nan, index=result.index, dtype=object) for c in exp.SPECS})
    levels = pd.DataFrame(0, index=result.index, columns=exp.SPECS)
    for number, (keys, lookup) in enumerate(zip(exp.LEVELS, prepared.levels), 1):
        for row_number, key in enumerate(result[keys].itertuples(index=False, name=None)):
            # Missing group keys were excluded by pandas groupby in the reference.
            if any(pd.isna(v) for v in key):
                continue
            found = lookup.get(tuple(key))
            if found is None:
                continue
            for c, value in zip(exp.SPECS, found):
                if pd.isna(values.at[row_number, c]) and pd.notna(value):
                    values.at[row_number, c] = value
                    levels.at[row_number, c] = number
    coverage = []
    for c in exp.SPECS:
        coverage.append({'feature': c, 'rows': len(result), 'missing': int(values[c].isna().sum()),
                         'coverage': float(values[c].notna().mean()), **{
                             f'level_{i}_rows': int(levels[c].eq(i).sum()) for i in range(1, 4)}, **{
                             f'level_{i}_proportion': float(levels[c].eq(i).mean()) for i in range(1, 4)}})
    audit = {'rows_without_any_reference': int(values.isna().all(axis=1).sum()),
             'rows_with_all_specs': int(values.notna().all(axis=1).sum()),
             'rows_receiving_any_attribute_by_level_nonexclusive': {
                 str(i): int(levels.eq(i).any(axis=1).sum()) for i in range(1, 4)}}
    return values, pd.DataFrame(coverage), audit
