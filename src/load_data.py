"""
Part 1: Download MIT-BIH Arrhythmia Database records.

Run once:
    python load_data.py            # downloads any of the 48 records not already present
    python load_data.py --test     # only downloads 3 records, for a quick test
"""

import os
import sys
import wfdb

from config import DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

# Full list of standard MIT-BIH record names (one per patient recording).
ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Small subset for a quick smoke test.
TEST_RECORDS = ['100', '101', '103']


def available_records(data_dir=DATA_DIR):
    """Return records that have all three required files (.hea, .dat, .atr), in numeric order."""
    record_names = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.hea'):
            continue
        record_name = filename[:-4]
        required = (f'{record_name}.hea', f'{record_name}.dat', f'{record_name}.atr')
        if all(os.path.isfile(os.path.join(data_dir, r)) for r in required):
            record_names.append(record_name)
    return sorted(record_names, key=int)


def download_records(record_list, dl_dir=DATA_DIR):
    """Download records from PhysioNet, skipping any that are already complete locally."""
    already = set(available_records(dl_dir))
    missing = [r for r in record_list if r not in already]
    if not missing:
        print(f"All {len(record_list)} requested record(s) are already in {dl_dir}.")
        return
    print(f"Downloading {len(missing)} record(s) to {dl_dir} ...")
    wfdb.dl_database('mitdb', dl_dir=dl_dir, records=missing)
    print("Download complete.")


if __name__ == "__main__":
    download_records(TEST_RECORDS if "--test" in sys.argv else ALL_RECORDS)
