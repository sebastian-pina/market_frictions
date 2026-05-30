"""
Download Oct 2024 AAPL/NVDA/SPY MBP-10 CSV.zst from FTP batch,
filter by symbol, and save as parquet into data/raw/<SYM>/mbp-10/.
Streams each file to disk first, then reads in 200k-row chunks
to avoid loading the full ~4 GB uncompressed CSV into memory.
"""
import ftplib, io, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

HOST    = 'ftp.databento.com'
USER    = 'srcontreras@g.ucla.edu'
PASS    = 'vY3aD!rqymsyS@w'
FTPPATH = '/KUAN8WAS/XNAS-20260529-WEVMVNN8X9'

SYMBOLS  = ['AAPL', 'NVDA', 'SPY']
DATA_DIR = Path('data/raw')
CHUNK    = 200_000   # rows per chunk

for sym in SYMBOLS:
    (DATA_DIR / sym / 'mbp-10').mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    'xnas-itch-20241001.mbp-10.csv.zst',
    'xnas-itch-20241002.mbp-10.csv.zst',
    'xnas-itch-20241003.mbp-10.csv.zst',
    'xnas-itch-20241004.mbp-10.csv.zst',
    'xnas-itch-20241007.mbp-10.csv.zst',
    'xnas-itch-20241008.mbp-10.csv.zst',
    'xnas-itch-20241009.mbp-10.csv.zst',
    'xnas-itch-20241010.mbp-10.csv.zst',
    'xnas-itch-20241011.mbp-10.csv.zst',
    'xnas-itch-20241014.mbp-10.csv.zst',
    'xnas-itch-20241015.mbp-10.csv.zst',
    'xnas-itch-20241016.mbp-10.csv.zst',
    'xnas-itch-20241017.mbp-10.csv.zst',
    'xnas-itch-20241018.mbp-10.csv.zst',
    'xnas-itch-20241021.mbp-10.csv.zst',
    'xnas-itch-20241022.mbp-10.csv.zst',
    'xnas-itch-20241023.mbp-10.csv.zst',
]

KEEP_COLS = (
    [f'bid_px_{i:02d}' for i in range(10)] +
    [f'ask_px_{i:02d}' for i in range(10)] +
    [f'bid_sz_{i:02d}' for i in range(10)] +
    [f'ask_sz_{i:02d}' for i in range(10)]
)


def already_done(date_str):
    return all(
        (DATA_DIR / sym / 'mbp-10' / f'{date_str}.parquet').exists()
        for sym in SYMBOLS
    )


READ_COLS = ['ts_recv', 'symbol'] + KEEP_COLS
DTYPE_MAP = {c: 'float32' for c in KEEP_COLS if '_px_' in c}
DTYPE_MAP.update({c: 'Int32' for c in KEEP_COLS if '_sz_' in c})
DTYPE_MAP['symbol'] = 'category'


def process_zst_file(tmp_path, date_str):
    """Chunked read (usecols + dtype) -> filter by symbol -> write parquet."""
    sym_chunks = {s: [] for s in SYMBOLS}
    done = {s: (DATA_DIR / s / 'mbp-10' / f'{date_str}.parquet').exists() for s in SYMBOLS}

    if all(done.values()):
        return

    reader = pd.read_csv(
        tmp_path,
        compression='zstd',
        usecols=READ_COLS,
        dtype=DTYPE_MAP,
        parse_dates=['ts_recv'],
        index_col='ts_recv',
        chunksize=CHUNK,
    )
    for chunk in reader:
        for sym in SYMBOLS:
            if done[sym]:
                continue
            sub = chunk[chunk['symbol'] == sym]
            if sub.empty:
                continue
            cols_ok = [c for c in KEEP_COLS if c in sub.columns]
            sub = sub[cols_ok].copy()
            # cast sizes to int64
            for c in [x for x in sub.columns if '_sz_' in x]:
                sub[c] = sub[c].fillna(0).astype(np.int64)
            sym_chunks[sym].append(sub)

    for sym in SYMBOLS:
        out_path = DATA_DIR / sym / 'mbp-10' / f'{date_str}.parquet'
        if done[sym]:
            print(f'    {sym} {date_str}: already exists, skip')
            continue
        if not sym_chunks[sym]:
            print(f'    {sym} {date_str}: no rows found')
            continue
        df_sym = pd.concat(sym_chunks[sym])
        df_sym.to_parquet(out_path, engine='fastparquet', index=True)
        print(f'    {sym} {date_str}: {len(df_sym):,} rows saved')
        del df_sym
    del sym_chunks


def ftp_download(fname, tmp_path):
    """Open a fresh FTP connection, download one file, close connection."""
    ftp = ftplib.FTP(HOST, timeout=300)
    ftp.login(USER, PASS)
    ftp.cwd(FTPPATH)
    with open(tmp_path, 'wb') as f:
        ftp.retrbinary(f'RETR {fname}', f.write)
    try:
        ftp.quit()
    except Exception:
        pass


def main():
    for i, fname in enumerate(DATA_FILES, 1):
        raw_date = fname.split('-')[2].split('.')[0]
        date_str = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}'

        if already_done(date_str):
            print(f'[{i:02d}/{len(DATA_FILES)}] {date_str}: done, skip')
            continue

        tmp_path = Path(f'_tmp_{date_str}.csv.zst')
        print(f'[{i:02d}/{len(DATA_FILES)}] Downloading {fname}...', end=' ', flush=True)
        ftp_download(fname, tmp_path)
        mb = tmp_path.stat().st_size / 1e6
        print(f'{mb:.0f} MB -> processing...', flush=True)

        try:
            process_zst_file(tmp_path, date_str)
        except Exception as e:
            print(f'  ERROR: {e}')
        finally:
            tmp_path.unlink(missing_ok=True)

    print('\nDone.')
    for sym in SYMBOLS:
        files = sorted((DATA_DIR / sym / 'mbp-10').glob('2024-10-*.parquet'))
        print(f'  {sym}: {len(files)} Oct-2024 files')


if __name__ == '__main__':
    main()
