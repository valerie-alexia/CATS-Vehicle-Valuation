"""Shared input sentinels; independent of model loading and serving."""
OTHER = 'Lainnya'


def is_other(value):
    return isinstance(value, str) and value.strip().casefold() == OTHER.casefold()
