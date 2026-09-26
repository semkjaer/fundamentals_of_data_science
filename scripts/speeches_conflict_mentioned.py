import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import country_converter as coco

cc = coco.CountryConverter()

conflict = pd.read_excel("../data/2026_ucdp-prio-acd-261.xlsx")

def convert_gwno_cell(val):
    """Converts a single GW code (int, float, or comma-separated string) to ISO3."""
    if pd.isna(val) or val == 0 or val == '0':
        return np.nan
    
    # Handle comma-separated lists of GW codes (e.g., "365, 370" -> ["365", "370"])
    if isinstance(val, str) and ',' in val:
        codes = [c.strip() for c in val.split(',') if c.strip() not in ['0', '']]
        converted = [cc.convert(names=c, src='GWcode', to='ISO3', not_found=None) for c in codes]
        # Drop Nones and join with comma (e.g., "RUS, BLR")
        valid = [c for c in converted if c is not None]
        return ', '.join(valid) if valid else np.nan

    # Handle single integer or string code
    try:
        clean_code = int(float(val))
        if clean_code == 0:
            return np.nan
        res = cc.convert(names=clean_code, src='GWcode', to='ISO3', not_found=None)
        return res if res != 'not found' else np.nan
    except (ValueError, TypeError):
        return np.nan

def convert_gwno(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            # Generate target column name (e.g., 'gwno_a' -> 'iso3_a')
            col_suffix = col.replace('gwno_', '')
            iso_col = f'iso3_{col_suffix}'
            
            # Apply cell-by-cell conversion
            df[iso_col] = df[col].apply(convert_gwno_cell)
            
    return df

to_convert = ['gwno_a', 'gwno_a_2nd', 'gwno_b', 'gwno_b_2nd', 'gwno_loc']
conflict = convert_gwno(conflict, to_convert)

country_data = cc.data

country_names = country_data[['ISO3', 'IEA', 'name_official', 'name_short', 'regex']]

df = conflict.merge(country_names, left_on='iso3_loc', right_on='ISO3', how='left')
df = df[['conflict_id', 'year', 'ISO3', 'IEA', 'name_official', 'name_short', 'regex']]

speeches = pd.read_parquet("../data/un_speeches.parquet")

# 1. Pre-build regex patterns once per conflict row
def build_pattern(row):
    cols = ['IEA', 'name_official', 'name_short', 'regex']
    variants = [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip() not in ['', 'nan', '0']]
    if not variants:
        return None
    escaped = [re.escape(v) for v in variants if len(v) > 0]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

df['compiled_pattern'] = df.apply(build_pattern, axis=1)

# 2. Pre-group conflicts by year into a dictionary
conflicts_by_year = {year: group for year, group in df.groupby('year')}

# 3. Fast lookup loop with progress bar
def match_speeches(speech_df):
    results = []
    # tqdm shows a progress bar so you can track exact completion speed
    for _, s_row in tqdm(speech_df.iterrows(), total=len(speech_df), desc="Processing Speeches"):
        s_year = s_row['year']
        s_text = s_row['text']
        
        if not isinstance(s_text, str) or s_year not in conflicts_by_year:
            results.append([])
            continue
            
        # Only evaluate conflicts active in that specific year
        year_conflicts = conflicts_by_year[s_year]
        matched = [
            int(c_row['conflict_id']) 
            for _, c_row in year_conflicts.iterrows() 
            if c_row['compiled_pattern'] and c_row['compiled_pattern'].search(s_text)
        ]
        results.append(matched)
        
    return results

# Run the fast matcher
speeches['discussed_conflict_ids'] = match_speeches(speeches)
speeches['num_conflicts_discussed'] = speeches['discussed_conflict_ids'].apply(len)
speeches['has_conflict_mention'] = speeches['num_conflicts_discussed'] > 0

speeches_exploded = speeches.explode('discussed_conflict_ids')
speeches_exploded = speeches_exploded.rename(columns={
    'discussed_conflict_ids': 'conflict_id',
})
# its now on the 'mention of a conflict in a speech' level so speaches can appear multiple times
speeches_exploded = speeches_exploded.merge(
    conflict[['conflict_id', 'year', 'iso3_a', 'iso3_a_2nd', 'iso3_b', 'iso3_b_2nd', 'iso3_loc',  'intensity_level', 'cumulative_intensity', 'type_of_conflict']],
    on=['conflict_id', 'year'],
    how='left'
)
speeches_exploded

def is_main_actor(row):
    """Whether a speech mentioning a conflict is delivered by a country which is a main actor in that conflict"""
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    for col in ['iso3_a', 'iso3_b']:
        val = row.get(col)
        if pd.isna(val):
            continue
            
        # Handle string containing comma-separated ISO3 codes
        if isinstance(val, str):
            actors = [a.strip() for a in val.split(',')]
            if speaker in actors:
                return True
                
        # Handle list or array objects
        elif isinstance(val, (list, tuple, set)):
            if speaker in [str(a).strip() for a in val]:
                return True
                
    return False

def is_secondary_actor(row):
    """Whether a speech mentioning a conflict is delivered by a country which is a secondary actor in that conflict"""
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    for col in ['iso3_a_2nd', 'iso3_b_2nd']:
        val = row.get(col)
        if pd.isna(val):
            continue
            
        # Handle string containing comma-separated ISO3 codes
        if isinstance(val, str):
            actors = [a.strip() for a in val.split(',')]
            if speaker in actors:
                return True
                
        # Handle list or array objects
        elif isinstance(val, (list, tuple, set)):
            if speaker in [str(a).strip() for a in val]:
                return True
                
    return False

def is_conflict_host(row):
    """Whether the speech mentioning a conflict is delivered by a country in which that conflict takes place"""
    speaker = str(row['country_code']).strip()
    if not speaker or pd.isna(row['country_code']):
        return False
        
    val = row.get('iso3_loc')
    if pd.isna(val):
        return False
        
    # Handle string containing comma-separated ISO3 codes
    if isinstance(val, str):
        actors = [a.strip() for a in val.split(',')]
        if speaker in actors:
            return True
            
    # Handle list or array objects
    elif isinstance(val, (list, tuple, set)):
        if speaker in [str(a).strip() for a in val]:
            return True
                
    return False

speeches_exploded['main_actor'] = speeches_exploded.apply(is_main_actor, axis=1)
speeches_exploded['secondary_actor'] = speeches_exploded.apply(is_secondary_actor, axis=1)
speeches_exploded['conflict_host'] = speeches_exploded.apply(is_conflict_host, axis=1)

ideals_added = speeches_exploded

# 1. Load and select columns
ideals = pd.read_csv('../data/IdealPointDyads1946-2025.csv')
ideals = ideals[['year', 'iso3c1', 'iso3c2', 'AbsIdealDiff']]

# 2. Make ideals undirected by duplicating rows with swapped country columns
ideals_reversed = ideals.rename(columns={'iso3c1': 'iso3c2', 'iso3c2': 'iso3c1'})
ideals_symmetric = pd.concat([ideals, ideals_reversed], ignore_index=True).drop_duplicates(
    subset=['year', 'iso3c1', 'iso3c2']
)

# 3. Merge for side_a
ideals_added = speeches_exploded.merge(
    ideals_symmetric,
    left_on=['year', 'country_code', 'iso3_a'],
    right_on=['year', 'iso3c1', 'iso3c2'],
    how='left'
).rename(columns={'AbsIdealDiff': 'side_a_ideals_diff'}).drop(columns=['iso3c1', 'iso3c2'])

# 4. Merge for side_b
ideals_added = ideals_added.merge(
    ideals_symmetric,
    left_on=['year', 'country_code', 'iso3_b'],
    right_on=['year', 'iso3c1', 'iso3c2'],
    how='left'
).rename(columns={'AbsIdealDiff': 'side_b_ideals_diff'}).drop(columns=['iso3c1', 'iso3c2'])

ideals_added.to_parquet("../data/conflict_mentions.parquet")