"""
update_drought.py
-----------------
Run this every Thursday after 8:30 AM EST to update drought data.
It downloads the latest USDA Drought Monitor data and updates the dashboard.

HOW TO RUN:
  python update_drought.py

SCHEDULE AUTOMATICALLY (Windows Task Scheduler):
  - Run every Thursday at 9:00 AM EST
"""

import urllib.request
import shutil
import os
import json
from datetime import datetime, timedelta
import pathlib

HERE = pathlib.Path(__file__).parent

def get_latest_drought_date():
    """Drought Monitor updates every Tuesday for the previous week, published Thursday."""
    today = datetime.today()
    # Find the most recent Tuesday
    days_since_tuesday = (today.weekday() - 1) % 7
    last_tuesday = today - timedelta(days=days_since_tuesday)
    return last_tuesday.strftime('%Y%m%d')

def download_drought_data():
    print("=" * 50)
    print("DROUGHT MONITOR UPDATE")
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Build download URL
    end_date   = get_latest_drought_date()
    start_date = '20060320'  # keep full history
    url = (f"https://droughtmonitor.unl.edu/Data/GISData.aspx"
           f"?mode=table&aoi=state&date={end_date}&startdate={start_date}&enddate={end_date}")

    out_file = HERE / 'dm_export_latest.csv'
    print(f"\nDownloading drought data through {end_date}...")
    print(f"URL: {url}")

    try:
        urllib.request.urlretrieve(url, out_file)
        size = os.path.getsize(out_file)
        print(f"✓ Downloaded {size/1024:.0f} KB → {out_file.name}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual download steps:")
        print("1. Go to: https://droughtmonitor.unl.edu/DmData/DataTables.aspx")
        print("2. Select: State → All States → Current week")
        print("3. Download CSV → save as dm_export_latest.csv in this folder")
        return False

    return True

def update_current_drought_values():
    """
    After downloading, update the current D2+/D3+ values in app.py.
    This reads the new CSV and patches the CUR_D2 / CUR_D3 arrays.
    """
    import csv

    csv_file = HERE / 'dm_export_latest.csv'
    if not csv_file.exists():
        print("✗ dm_export_latest.csv not found — run download first")
        return

    # Cotton belt states in order matching app.py
    STATES = ['TX','GA','AR','MS','NC','AL','SC','OK','LA','TN','MO','KS','AZ','VA','FL','NM','CA']

    state_d2 = {s: 0.0 for s in STATES}
    state_d3 = {s: 0.0 for s in STATES}

    latest_date = ''

    with open(csv_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Get the latest date in file
    date_col = 'MapDate' if 'MapDate' in rows[0] else list(rows[0].keys())[0]
    dates = sorted(set(r[date_col] for r in rows), reverse=True)
    latest_date = dates[0] if dates else 'unknown'
    latest_rows = [r for r in rows if r[date_col] == latest_date]

    for row in latest_rows:
        st = row.get('StateAbbreviation', row.get('State', ''))
        if st in STATES:
            # D2+ = D2 + D3 + D4
            try:
                d2 = float(row.get('D2', 0) or 0)
                d3 = float(row.get('D3', 0) or 0)
                d4 = float(row.get('D4', 0) or 0)
                state_d2[st] = round(d2 + d3 + d4, 1)
                state_d3[st] = round(d3 + d4, 1)
            except (ValueError, TypeError):
                pass

    print(f"\n✓ Latest drought date: {latest_date}")
    print(f"  Cotton belt D2+ values:")
    for s in STATES:
        print(f"    {s}: D2+={state_d2[s]}%  D3+={state_d3[s]}%")

    # Patch app.py
    app_file = HERE / 'app.py'
    with open(app_file) as f:
        src = f.read()

    # Replace CUR_D2 and CUR_D3 arrays
    import re
    new_d2 = 'CUR_D2   = [' + ','.join(str(state_d2[s]) for s in STATES) + ']'
    new_d3 = 'CUR_D3   = [' + ','.join(str(state_d3[s]) for s in STATES) + ']'

    src = re.sub(r'CUR_D2\s*=\s*\[[^\]]+\]', new_d2, src)
    src = re.sub(r'CUR_D3\s*=\s*\[[^\]]+\]', new_d3, src)

    with open(app_file, 'w') as f:
        f.write(src)

    print(f"\n✓ app.py updated with latest drought data ({latest_date})")
    print(f"  Restart the dashboard to see changes.")

if __name__ == '__main__':
    success = download_drought_data()
    if success:
        update_current_drought_values()
    print("\nDone. Restart app.py to see updated data.")
