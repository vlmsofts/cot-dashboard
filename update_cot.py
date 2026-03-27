"""
update_cot.py
-------------
Run every Friday after 3:30 PM EST to update COT positioning data.
Uses the CFTC Public Reporting API (no login, no blocked downloads).

HOW TO RUN:
  python update_cot.py
"""

import urllib.request
import json
import re
import os
import pathlib
from datetime import datetime

HERE = pathlib.Path(__file__).parent

# Markets we track — exactly as they appear in the CFTC API
MARKET_NAMES = {
    'COTTON NO. 2 - ICE FUTURES U.S.':                          'Cotton',
    'CORN - CHICAGO BOARD OF TRADE':                            'Corn',
    'SOYBEANS - CHICAGO BOARD OF TRADE':                        'Soybeans',
    'WHEAT-SRW - CHICAGO BOARD OF TRADE':                       'SRW Wheat',
    'WHEAT-HRW - CHICAGO BOARD OF TRADE':                       'HRW Wheat',
    'SUGAR NO. 11 - ICE FUTURES U.S.':                          'Sugar',
    'CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE':    'WTI Crude',
    'GOLD - COMMODITY EXCHANGE INC.':                           'Gold',
    'SILVER - COMMODITY EXCHANGE INC.':                         'Silver',
    'LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE':                'Live Cattle',
    'LEAN HOGS - CHICAGO MERCANTILE EXCHANGE':                  'Lean Hogs',
    'SOYBEAN OIL - CHICAGO BOARD OF TRADE':                     'Soy Oil',
    'SOYBEAN MEAL - CHICAGO BOARD OF TRADE':                    'Soy Meal',
    'COCOA - ICE FUTURES U.S.':                                 'Cocoa',
    'COFFEE C - ICE FUTURES U.S.':                              'Coffee',
}

def fetch_latest_cot():
    """
    Pull latest week's COT data via CFTC Public Reporting API.
    No zip file, no blocked URLs — this API works reliably.
    """
    print("Fetching COT data from CFTC Public API...")

    # The CFTC Socrata API endpoint for Disaggregated Futures Only
    # $limit=500 gets all markets, $order sorts by date descending
    url = (
        "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
        "?$limit=500"
        "&$order=report_date_as_yyyy_mm_dd+DESC"
        "&$where=report_date_as_yyyy_mm_dd>='2026-01-01'"
    )

    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json',
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        print(f"  API returned {len(data)} records")
        return data
    except Exception as e:
        print(f"  ✗ API call failed: {e}")
        return None


def extract_positions(data):
    """Find the latest week's positions for our tracked commodities."""
    if not data:
        return {}

    results = {}
    latest_dates = {}

    for row in data:
        # Market name field
        mkt = row.get('market_and_exchange_names', '').strip().upper()
        comm = MARKET_NAMES.get(mkt)
        if not comm:
            continue

        date_str = row.get('report_date_as_yyyy_mm_dd', '')[:10]

        # Keep only the latest date per commodity
        if comm in latest_dates and date_str <= latest_dates[comm]:
            continue

        latest_dates[comm] = date_str

        try:
            oi        = int(float(row.get('open_interest_all', 0) or 0))
            mm_long   = int(float(row.get('m_money_positions_long_all',  0) or 0))
            mm_short  = int(float(row.get('m_money_positions_short_all', 0) or 0))
            pm_long   = int(float(row.get('prod_merc_positions_long_all',  0) or 0))
            pm_short  = int(float(row.get('prod_merc_positions_short_all', 0) or 0))
            sw_long   = int(float(row.get('swap_positions_long_all',   0) or 0))
            sw_short  = int(float(row.get('swap__positions_short_all', 0) or 0))
            or_long   = int(float(row.get('other_rept_positions_long_all',  0) or 0))
            or_short  = int(float(row.get('other_rept_positions_short_all', 0) or 0))
        except (ValueError, TypeError):
            continue

        mm_net = mm_long - mm_short
        results[comm] = {
            'date':         date_str,
            'oi':           oi,
            'mm_net':       mm_net,
            'mm_long':      mm_long,
            'mm_short':     mm_short,
            'mm_net_pct':   round(mm_net / oi * 100, 1) if oi else 0,
            'prod_net':     pm_long - pm_short,
            'swap_net':     sw_long - sw_short,
            'other_net':    or_long - or_short,
        }

    return results


def patch_app(positions):
    """Update the BCOM dict in app.py with new contract counts."""
    if not positions:
        print("No positions data to patch.")
        return

    app_file = HERE / 'app.py'
    if not app_file.exists():
        print(f"✗ app.py not found in {HERE}")
        return

    with open(app_file, encoding="utf-8") as f:
        src = f.read()

    updated = []
    skipped = []

    for comm, p in positions.items():
        pattern = re.compile(
            rf"('{re.escape(comm)}':\s*{{[^}}]*?'mm_net':\s*)-?\d+"
            rf"([^}}]*?'mm_long':\s*)\d+"
            rf"([^}}]*?'mm_short':\s*)\d+"
            rf"([^}}]*?'prod_net':\s*)-?\d+"
            rf"([^}}]*?'swap_net':\s*)-?\d+"
            rf"([^}}]*?'other_net':\s*)-?\d+"
        )
        replacement = (
            rf"\g<1>{p['mm_net']}\g<2>{p['mm_long']}\g<3>{p['mm_short']}"
            rf"\g<4>{p['prod_net']}\g<5>{p['swap_net']}\g<6>{p['other_net']}"
        )
        new_src, n = pattern.subn(replacement, src)
        if n:
            src = new_src
            updated.append(comm)
        else:
            skipped.append(comm)

    with open(app_file, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"\n  ✓ Updated in app.py: {', '.join(updated)}")
    if skipped:
        print(f"  ⚠ Not found (pattern mismatch): {', '.join(skipped)}")


def patch_heatmap(positions):
    """Also update _EMBEDDED_HEATMAP values in app.py with fresh contract counts."""
    import re
    app_file = HERE / 'app.py'
    if not app_file.exists():
        return

    with open(app_file, encoding='utf-8') as f:
        src = f.read()

    updated = []
    for comm, p in positions.items():
        if p['oi'] <= 0:
            continue

        mm_long_pct  = round(p['mm_long']  / p['oi'] * 100, 1)
        mm_short_pct = round(p['mm_short'] / p['oi'] * 100, 1)
        mm_net_pct   = round(p['mm_net']   / p['oi'] * 100, 1)

        # Update OI and date in heatmap entry
        oi_pat   = re.compile(rf"('{re.escape(comm)}':\s*{{)'OI':\s*\d+")
        src, n1  = oi_pat.subn(rf"\g<1>'OI': {p['oi']}", src)

        date_pat = re.compile(rf"('{re.escape(comm)}':[^{{]*{{[^}}]*)'date':\s*'[^']*'")
        src, n2  = date_pat.subn(rf"\g<1>'date': '{p['date']}'", src)

        # Update MM Net value in heatmap (keep existing percentile rank - Bloomberg updates those)
        mm_net_pat = re.compile(
            rf"('{re.escape(comm)}'.*?'MM'.*?'Net'.*?'v':\s*)-?[\d.]+",
            re.DOTALL
        )
        src, n3 = mm_net_pat.subn(rf"\g<1>{float(p['mm_net'])}", src, count=1)

        # Update MM Long value
        mm_long_pat = re.compile(
            rf"('{re.escape(comm)}'.*?'MM'.*?'Long'.*?'v':\s*)[\d.]+",
            re.DOTALL
        )
        src, n4 = mm_long_pat.subn(rf"\g<1>{float(p['mm_long'])}", src, count=1)

        # Update MM Short value
        mm_short_pat = re.compile(
            rf"('{re.escape(comm)}'.*?'MM'.*?'Short'.*?'v':\s*)[\d.]+",
            re.DOTALL
        )
        src, n5 = mm_short_pat.subn(rf"\g<1>{float(p['mm_short'])}", src, count=1)

        if any([n1, n2, n3]):
            updated.append(comm)

    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(src)

    if updated:
        print(f"  ✓ Heatmap updated for: {', '.join(updated)}")
    else:
        print("  ⚠ Heatmap patterns not matched — manual update may be needed")



if __name__ == '__main__':
    print("=" * 50)
    print("CFTC COT UPDATE")
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    data = fetch_latest_cot()

    if not data:
        print("\n✗ Could not retrieve data from CFTC API.")
        print("  Try again after 3:30 PM EST on Fridays.")
        print("  Or check your internet connection.")
    else:
        positions = extract_positions(data)

        print(f"\n✓ Latest COT positions ({len(positions)} markets found):")
        for comm, p in sorted(positions.items()):
            print(f"  {comm:12s}  {p['date']}  OI={p['oi']:,}  "
                  f"MM_net={p['mm_net']:+,}  ({p['mm_net_pct']:+.1f}% OI)")

        patch_app(positions)
        print(f"\n✓ Done. Restart app.py or push to GitHub to see updates.")
    patch_heatmap(positions)

