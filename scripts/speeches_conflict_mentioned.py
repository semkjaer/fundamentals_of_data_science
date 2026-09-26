import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import country_converter as coco

cc = coco.CountryConverter()

# ---------------------------------------------------------
# 1. Load UCDP/PRIO Conflict Data & Convert GW Codes to ISO3
# ---------------------------------------------------------
conflict = pd.read_excel("../data/2026_ucdp-prio-acd-261.xlsx")

# Initialize country_converter instance
cc = coco.CountryConverter()

# Explicit overrides for historical GW codes that country_converter cannot resolve
HISTORICAL_GW_TO_ISO3 = {
    316: "RUS",  # USSR / Russian SFSR
    345: "YUG",  # Yugoslavia (former)
    678: "YEM",  # Yemen Arab Republic (North Yemen)
    680: "YEM",  # People's Democratic Republic of Yemen (South Yemen)
    751: "IND",  # Hyderabad (annexed into India in 1948)
    816: "VNM",  # Democratic Republic of Vietnam (North Vietnam)
    817: "VNM",  # Republic of Vietnam (South Vietnam)
}

def convert_single_gw_code(code_val) -> str:
    """
    Converts a single GW numeric code (or string representation of one) into ISO3.
    Returns None if missing, invalid, or zero.
    """
    if pd.isna(code_val):
        return None
    
    # Clean string and convert to integer
    try:
        code_int = int(float(str(code_val).strip()))
    except (ValueError, TypeError):
        return None
        
    if code_int <= 0:
        return None
        
    # 1. Check manual historical dictionary
    if code_int in HISTORICAL_GW_TO_ISO3:
        return HISTORICAL_GW_TO_ISO3[code_int]
        
    # 2. Convert via country_converter
    res = cc.convert(names=code_int, src="GWcode", to="ISO3", not_found=None)
    return res if res and res != "not found" else None

def convert_gw_cell(val):
    """
    Handles single codes, floats, and comma-separated lists of codes.
    E.g.:
      145       -> "BOL"
      "28, 3"   -> "GBR, USA"
      np.nan    -> np.nan
    """
    if pd.isna(val):
        return np.nan
        
    val_str = str(val).strip()
    if not val_str or val_str in ['0', 'nan', 'None']:
        return np.nan
        
    # Case A: Comma-separated list of codes
    if "," in val_str:
        raw_codes = [c.strip() for c in val_str.split(",")]
        converted = [convert_single_gw_code(c) for c in raw_codes]
        valid_codes = [c for c in converted if c is not None]
        return ", ".join(valid_codes) if valid_codes else np.nan
        
    # Case B: Single code
    res = convert_single_gw_code(val_str)
    return res if res is not None else np.nan

gw_columns = [
    'gwno_a',      # Primary actor Side A (usually Government)
    'gwno_a_2nd',  # Supporting actors Side A
    'gwno_b',      # Primary actor Side B (State if interstate, NaN if rebel group)
    'gwno_b_2nd',  # Supporting actors Side B
    'gwno_loc'     # Conflict location
]

for col in gw_columns:
    if col in conflict.columns:
        target_col = col.replace('gwno_', 'iso3_')
        conflict[target_col] = conflict[col].apply(convert_gw_cell)
        print(f"Converted {col} -> {target_col} (Non-null: {conflict[target_col].notna().sum()} / {len(conflict)})")

# ---------------------------------------------------------
# 2. Build Regex Search Patterns per Conflict Location
# ---------------------------------------------------------
country_data = cc.data
country_names = country_data[['ISO3', 'IEA', 'name_official', 'name_short', 'regex']]

# Merge country names based on the conflict's location ISO3 code
conflict_patterns = conflict.merge(country_names, left_on='iso3_loc', right_on='ISO3', how='left')

def build_pattern(row):
    cols = ['IEA', 'name_official', 'name_short', 'regex']
    variants = [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip() not in ['', 'nan', '0']]
    if not variants:
        return None
    escaped = [re.escape(v) for v in variants if len(v) > 0]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

conflict_patterns['compiled_pattern'] = conflict_patterns.apply(build_pattern, axis=1)

# Keep relevant metadata from conflict file
conflict_meta_cols = [
    'conflict_id', 'year', 'side_a', 'side_a_2nd', 'side_b', 'side_b_2nd',
    'iso3_a', 'iso3_a_2nd', 'iso3_b', 'iso3_b_2nd', 'iso3_loc', 'intensity_level',
    'cumulative_intensity', 'type_of_conflict', 'compiled_pattern'
]
conflict_clean = conflict_patterns[conflict_meta_cols]

# ---------------------------------------------------------
# 3. Create Full Conflict-Speaker Cross-Join Grid per Year
# ---------------------------------------------------------
speeches = pd.read_parquet("../data/un_speeches.parquet")

# Extract unique speaker countries present in each year
speakers_per_year = speeches[['year', 'country_code']].drop_duplicates()

# Merge conflicts with all active speaker countries in that same year
conflict_speaker_df = conflict_clean.merge(speakers_per_year, on='year', how='inner')

# Combine speech text per speaker country in each year (concatenating if multiple)
speech_texts = speeches.groupby(['year', 'country_code'])['text'].apply(lambda x: " ".join(x.dropna())).reset_index()
conflict_speaker_df = conflict_speaker_df.merge(speech_texts, on=['year', 'country_code'], how='left')

# ---------------------------------------------------------
# 4. Check Speech Text for Conflict Mentions (Boolean Flag)
# ---------------------------------------------------------
tqdm.pandas(desc="Evaluating conflict mentions in speeches")

def check_mention(row):
    text = row['text']
    pattern = row['compiled_pattern']
    if pd.isna(text) or not pattern or not isinstance(text, str):
        return False
    return bool(pattern.search(text))

conflict_speaker_df['has_conflict_mention'] = conflict_speaker_df.progress_apply(check_mention, axis=1)

# Drop heavy regex objects and full text prior to merging features
conflict_speaker_df = conflict_speaker_df.drop(columns=['compiled_pattern', 'text'])

# ---------------------------------------------------------
# 5. Evaluate Actor & Host Classifications
# ---------------------------------------------------------
def is_main_actor(row):
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    for col in ['iso3_a', 'iso3_b']:
        val = row.get(col)
        if pd.isna(val):
            continue
        if isinstance(val, str):
            actors = [a.strip() for a in val.split(',')]
            if speaker in actors:
                return True
        elif isinstance(val, (list, tuple, set)):
            if speaker in [str(a).strip() for a in val]:
                return True
    return False

def is_secondary_actor(row):
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    for col in ['iso3_a_2nd', 'iso3_b_2nd']:
        val = row.get(col)
        if pd.isna(val):
            continue
        if isinstance(val, str):
            actors = [a.strip() for a in val.split(',')]
            if speaker in actors:
                return True
        elif isinstance(val, (list, tuple, set)):
            if speaker in [str(a).strip() for a in val]:
                return True
    return False

def is_conflict_host(row):
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    val = row.get('iso3_loc')
    if pd.isna(val):
        return False
    if isinstance(val, str):
        actors = [a.strip() for a in val.split(',')]
        if speaker in actors:
            return True
    elif isinstance(val, (list, tuple, set)):
        if speaker in [str(a).strip() for a in val]:
            return True
    return False

conflict_speaker_df['main_actor'] = conflict_speaker_df.apply(is_main_actor, axis=1)
conflict_speaker_df['secondary_actor'] = conflict_speaker_df.apply(is_secondary_actor, axis=1)
conflict_speaker_df['conflict_host'] = conflict_speaker_df.apply(is_conflict_host, axis=1)

# ---------------------------------------------------------
# 6. Merge Ideal Point Differences across All Pairs
# ---------------------------------------------------------
ideals = pd.read_csv('../data/IdealPointDyads1946-2025.csv')
ideals = ideals[['year', 'iso3c1', 'iso3c2', 'AbsIdealDiff']]

# Symmetrize dyadic ideal points
ideals_reversed = ideals.rename(columns={'iso3c1': 'iso3c2', 'iso3c2': 'iso3c1'})
ideals_symmetric = pd.concat([ideals, ideals_reversed], ignore_index=True).drop_duplicates(
    subset=['year', 'iso3c1', 'iso3c2']
)

# Merge difference against side_a
df = conflict_speaker_df.merge(
    ideals_symmetric,
    left_on=['year', 'country_code', 'iso3_a'],
    right_on=['year', 'iso3c1', 'iso3c2'],
    how='left'
).rename(columns={'AbsIdealDiff': 'side_a_ideals_diff'}).drop(columns=['iso3c1', 'iso3c2'])

# Merge difference against side_b
df = df.merge(
    ideals_symmetric,
    left_on=['year', 'country_code', 'iso3_b'],
    right_on=['year', 'iso3c1', 'iso3c2'],
    how='left'
).rename(columns={'AbsIdealDiff': 'side_b_ideals_diff'}).drop(columns=['iso3c1', 'iso3c2'])

# ---------------------------------------------------------
# 7. Merge CEPII Geographical Distances across All Pairs
# ---------------------------------------------------------
dist = pd.read_excel('../data/dist_cepii.xls')
dist_cols = ['iso_o', 'iso_d', 'distw']
dist_clean = dist[dist_cols]

# Symmetrize distance measure
dist_reversed = dist_clean.rename(columns={'iso_o': 'iso_d', 'iso_d': 'iso_o'})
dist_symmetric = pd.concat([dist_clean, dist_reversed], ignore_index=True).drop_duplicates(
    subset=['iso_o', 'iso_d']
)

# Merge weighted distance to side_a
df = df.merge(
    dist_symmetric,
    left_on=['country_code', 'iso3_a'],
    right_on=['iso_o', 'iso_d'],
    how='left'
).rename(columns={'distw': 'side_a_dist'}).drop(columns=['iso_o', 'iso_d'])

# Merge weighted distance to side_b
df = df.merge(
    dist_symmetric,
    left_on=['country_code', 'iso3_b'],
    right_on=['iso_o', 'iso_d'],
    how='left'
).rename(columns={'distw': 'side_b_dist'}).drop(columns=['iso_o', 'iso_d'])

# ---------------------------------------------------------
# 8. Join cleaned conflict data
# ---------------------------------------------------------
conflict_deaths = pd.read_csv("../conflict_deaths_1946_2025.csv")

# join only cols not already in main df
additive_conflict_cols = ['conflict_id', 'year'] + [col for col in conflict_deaths.columns if col not in df.columns and 'gwno' not in col and 'id' not in col]

conflict_deaths = conflict_deaths[additive_conflict_cols]
df = df.merge(conflict_deaths, on=['conflict_id', 'year'], how='left')

# ---------------------------------------------------------
# 9. save data
# ---------------------------------------------------------
df.to_parquet('../data/final_clean.csv')