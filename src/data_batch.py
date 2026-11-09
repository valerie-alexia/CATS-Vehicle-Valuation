"""Monthly batch naming and crash-safe CSV persistence; no browser or data I/O on import."""
from datetime import datetime
import os
from pathlib import Path
import re
import tempfile


def default_batch():
    return os.environ.get('CATS_BATCH', '').strip() or datetime.now().strftime('%Y-%m')


def validate_batch(batch):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', batch):
        raise ValueError('Batch harus berupa suffix nama file tanpa separator direktori')
    return batch


def latest_raw_batch(directory, prefixes):
    """Select actual monthly RAW inputs; never mix vehicle batches or infer a legacy year."""
    latest = []
    for prefix in prefixes:
        batches = [p.name[len(prefix + '_raw_dataset_'):-4]
                   for p in Path(directory).glob(f'{prefix}_raw_dataset_*.csv')
                   if re.fullmatch(prefix + r'_raw_dataset_\d{4}-(0[1-9]|1[0-2])\.csv', p.name)
                   and p.stat().st_size > 0]
        if not batches:
            raise ValueError(f'File RAW bulanan tidak ditemukan untuk {prefix}; gunakan CATS_BATCH untuk batch historis.')
        latest.append(max(batches))
    if len(set(latest)) != 1:
        raise ValueError('Batch RAW terbaru berbeda antar kendaraan; pilih --vehicle atau CATS_BATCH secara eksplisit.')
    return latest[0]


def atomic_csvs(outputs):
    """Stage/read back all CSVs, then replace; roll back completed replacements on error.

    Atomic per file. Multi-file rollback handles Python exceptions, not power loss.
    Only one writer per batch is supported.
    """
    import pandas as pd
    staged, previous, replaced = {}, {}, []
    try:
        for target, frame in outputs:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
            os.close(fd)
            staged[target] = Path(name)
            frame.to_csv(name, index=False)
            with open(name, 'r+b') as stream:
                os.fsync(stream.fileno())
            check = pd.read_csv(name, low_memory=False)
            if list(check.columns) != list(frame.columns) or len(check) != len(frame):
                raise ValueError('Validasi schema/jumlah baris CSV gagal; output sebelumnya dipertahankan.')
            previous[target] = target.read_bytes() if target.exists() else None
        for target, stage in staged.items():
            os.replace(stage, target)
            replaced.append(target)
    except BaseException:
        for target in reversed(replaced):
            old = previous[target]
            if old is None:
                target.unlink()
            else:
                # Restore through the staged sibling, without truncating the destination.
                staged[target].write_bytes(old)
                os.replace(staged[target], target)
        raise
    finally:
        for stage in staged.values():
            stage.unlink(missing_ok=True)


def save_scrape(frame, path, append=False):
    import pandas as pd
    path = Path(path)
    if append and path.exists():
        frame = pd.concat([pd.read_csv(path, low_memory=False), frame], ignore_index=True)
    if 'link' not in frame or frame['link'].isna().any() or frame['link'].astype(str).str.strip().eq('').any():
        raise ValueError('Output scraper harus memiliki URL link yang tidak kosong.')
    frame = frame.drop_duplicates('link', keep='first').reset_index(drop=True)
    atomic_csvs([(path, frame)])
