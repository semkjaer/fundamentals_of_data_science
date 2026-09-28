import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import country_converter as coco

# Initialize country_converter instance
cc = coco.CountryConverter()

# ---------------------------------------------------------
# 1. Load UCDP/PRIO Conflict Data & Convert GW Codes to ISO3
# ---------------------------------------------------------
conflict = pd.read_excel("data/2026_ucdp-prio-acd-261.xlsx")

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
    """Converts a single GW numeric code to ISO3."""
    if pd.isna(code_val):
        return None
    try:
        code_int = int(float(str(code_val).strip()))
    except (ValueError, TypeError):
        return None
        
    if code_int <= 0:
        return None
        
    if code_int in HISTORICAL_GW_TO_ISO3:
        return HISTORICAL_GW_TO_ISO3[code_int]
        
    res = cc.convert(names=code_int, src="GWcode", to="ISO3", not_found=None)
    return res if res and res != "not found" else None

def convert_gw_cell(val):
    """Handles single codes, floats, and delimited lists of GW codes (commas, slashes, spaces)."""
    if pd.isna(val):
        return np.nan
        
    val_str = str(val).strip()
    if not val_str or val_str in ['0', 'nan', 'None']:
        return np.nan
        
    # Split across commas, slashes, semicolons, or whitespace
    raw_codes = re.split(r'[,/;\s]+', val_str)
    raw_codes = [c.strip() for c in raw_codes if c.strip()]
    
    converted = [convert_single_gw_code(c) for c in raw_codes]
    valid_codes = [c for c in converted if c is not None]
    
    return ", ".join(list(dict.fromkeys(valid_codes))) if valid_codes else np.nan

gw_columns = [
    'gwno_a',      # Primary actor Side A
    'gwno_a_2nd',  # Supporting actors Side A
    'gwno_b',      # Primary actor Side B
    'gwno_b_2nd',  # Supporting actors Side B
    'gwno_loc'     # Conflict location
]

# PERFORM GW TO ISO3 CONVERSION FIRST BEFORE ANY PATTERN BUILDING
for col in gw_columns:
    if col in conflict.columns:
        target_col = col.replace('gwno_', 'iso3_')
        conflict[target_col] = conflict[col].apply(convert_gw_cell)
        print(f"Converted {col} -> {target_col} (Non-null: {conflict[target_col].notna().sum()} / {len(conflict)})")

# ---------------------------------------------------------
# 2. Build Target Entity Regex Patterns (Location + Side A/B)
# ---------------------------------------------------------
country_data = cc.data
country_names = country_data[['ISO3', 'IEA', 'name_official', 'name_short', 'regex']].dropna(subset=['ISO3'])

# Map ISO3 codes to all available name variants
iso3_to_variants = {}
for _, row in country_names.iterrows():
    iso3 = str(row['ISO3']).strip()
    cols = ['IEA', 'name_official', 'name_short', 'regex']
    variants = [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip() not in ['', 'nan', '0']]
    if variants:
        iso3_to_variants[iso3] = variants

def build_targeted_entity_pattern(row):
    """
    Extracts name variants for:
    1. Conflict Location (iso3_loc)
    2. Primary & Secondary Side A ISO3s + raw side_a text
    3. Primary & Secondary Side B ISO3s + raw side_b text (e.g. rebel/actor names)
    """
    iso_list = []
    
    # 1. Gather ISO3 codes across all actor and location fields
    for col in ['iso3_loc', 'iso3_a', 'iso3_b', 'iso3_a_2nd', 'iso3_b_2nd']:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
            codes = [code.strip() for code in str(val).split(',') if code.strip()]
            iso_list.extend(codes)

    all_variants = []
    for iso in set(iso_list):
        if iso in iso3_to_variants:
            all_variants.extend(iso3_to_variants[iso])
            
    # 2. Add fulltext names of side_a and side_b (e.g., specific non-state groups or state names)
    for actor_col in ['side_a', 'side_b', 'side_a_2nd', 'side_b_2nd']:
        text_val = row.get(actor_col)
        if pd.notna(text_val) and str(text_val).strip() not in ['', 'nan', 'None']:
            # Handle multiple actors delimited by commas or slashes
            actors = re.split(r'[,;/]+', str(text_val))
            for actor in actors:
                cleaned_actor = actor.strip()
                # Ignore generic placeholder text
                if len(cleaned_actor) > 2 and not cleaned_actor.lower().startswith('government of'):
                    all_variants.append(cleaned_actor)

    if not all_variants:
        return None
        
    unique_variants = list(set(all_variants))
    escaped = [re.escape(v) for v in unique_variants if len(v) > 0]
    escaped.sort(key=len, reverse=True)  # Match longest strings first
    
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

# Compile entity matching patterns
conflict['compiled_pattern'] = conflict.apply(build_targeted_entity_pattern, axis=1)

# Select metadata columns
conflict_meta_cols = [
    'conflict_id', 'year', 'side_a', 'side_a_2nd', 'side_b', 'side_b_2nd',
    'iso3_a', 'iso3_a_2nd', 'iso3_b', 'iso3_b_2nd', 'iso3_loc', 'intensity_level',
    'cumulative_intensity', 'type_of_conflict', 'compiled_pattern'
]
existing_cols = [col for col in conflict_meta_cols if col in conflict.columns]
conflict_clean = conflict[existing_cols]

# ---------------------------------------------------------
# 3. Create Full Conflict-Speaker Cross-Join Grid per Year
# ---------------------------------------------------------
speeches = pd.read_parquet("data/un_speeches.parquet")

# Extract unique speaker countries present in each year
speakers_per_year = speeches[['year', 'country_code']].drop_duplicates()

# Merge conflicts with all active speaker countries in that same year
conflict_speaker_df = conflict_clean.merge(speakers_per_year, on='year', how='inner')

# Combine speech text per speaker country in each year (concatenating if multiple)
speech_texts = speeches.groupby(['year', 'country_code'])['text'].apply(lambda x: " ".join(x.dropna())).reset_index()
conflict_speaker_df = conflict_speaker_df.merge(speech_texts, on=['year', 'country_code'], how='left')

# ---------------------------------------------------------
# 4. Check Speech Text for Conflict Mentions (Entity + War Term)
# ---------------------------------------------------------
tqdm.pandas(desc="Evaluating conflict mentions in speeches")

# Generic conflict/war anchor terms common in UN general debate speeches
WAR_ANCHOR_TERMS = [
    r'conflict', r'war', r'warfare', r'fighting', r'hostilities', r'ceasefire', 
    r'aggression', r'rebel', r'rebels', r'insurgency', r'insurgents', 
    r'armed groups?', r'violence', r'violent', r'crisis', r'clashes', 
    r'military operations?', r'peacekeeping', r'invasion', r'combatants?',
    r'terroris[mt]', r'attacks?', r'atrocities', r'casualties'
]

WAR_ANCHOR_PATTERN = re.compile(r'\b(' + '|'.join(WAR_ANCHOR_TERMS) + r')\b', flags=re.IGNORECASE)

def check_targeted_mention(row):
    text = row['text']
    pattern = row['compiled_pattern']
    
    if pd.isna(text) or not pattern or not isinstance(text, str):
        return False
        
    # Condition 1: Mentions conflict location, Side A, or Side B
    has_entity_mention = bool(pattern.search(text))
    if not has_entity_mention:
        return False
        
    # Condition 2: Speech ALSO mentions at least one war/conflict anchor term
    has_war_keyword = bool(WAR_ANCHOR_PATTERN.search(text))
    
    return has_entity_mention and has_war_keyword

conflict_speaker_df['has_conflict_mention'] = conflict_speaker_df.progress_apply(check_targeted_mention, axis=1)

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
ideals = pd.read_csv('data/IdealPointDyads1946-2025.csv')
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
dist = pd.read_excel('data/dist_cepii.xls')
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
conflict_deaths = pd.read_csv("data/conflict_deaths_1946_2025.csv")

# join only cols not already in main df
additive_conflict_cols = ['conflict_id', 'year'] + [col for col in conflict_deaths.columns if col not in df.columns and 'gwno' not in col and 'id' not in col]

conflict_deaths = conflict_deaths[additive_conflict_cols]
df = df.merge(conflict_deaths, on=['conflict_id', 'year'], how='left')

# ---------------------------------------------------------
# 9. Save Data
# ---------------------------------------------------------
df.to_csv('data/final_clean.csv', index=False)
df.to_csv('final_clean.csv', index=False)
df.to_parquet('data/final_clean.parquet', index=False)

print(f"Successfully saved final_clean dataset with {len(df)} rows!")
