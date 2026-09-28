import pandas as pd

# ---------------------------------------------------------
# 1. Load Conflict and Battle Deaths Datasets
# ---------------------------------------------------------
df_acd = pd.read_excel("data/2026_ucdp-prio-acd-261.xlsx")
df_bd26 = pd.read_csv("data/BattleDeaths_v26_1_conf.csv")
df_prio = pd.read_excel("data/PRIO Battle Deaths Dataset 31.xls")

df_acd['conflict_id'] = df_acd['conflict_id'].astype(int)
df_acd['year'] = df_acd['year'].astype(int)

df_bd26['conflict_id'] = df_bd26['conflict_id'].astype(int)
df_bd26['year'] = df_bd26['year'].astype(int)

df_prio['year'] = df_prio['year'].astype(int)
df_prio['new_conflict_id'] = pd.to_numeric(df_prio['id'], errors='coerce')

# ---------------------------------------------------------
# 2. Merge ACD and Historical PRIO v3.1 (1946-1988)
# ---------------------------------------------------------
merged_step1 = pd.merge(
    df_acd,
    df_prio,
    left_on=['conflict_id', 'year'],
    right_on=['new_conflict_id', 'year'],
    how='outer',
    suffixes=('', '_prio'),
    indicator='prio_merge'
)

# ---------------------------------------------------------
# 3. Merge Modern UCDP Battle Deaths v26.1 (1989-2025)
# ---------------------------------------------------------
merged_step2 = pd.merge(
    merged_step1,
    df_bd26[['conflict_id', 'year', 'bd_best', 'bd_low', 'bd_high']],
    on=['conflict_id', 'year'],
    how='left'
)

# ---------------------------------------------------------
# 4. Coalesce Metadata Columns
# ---------------------------------------------------------
merged_step2['conflict_id'] = merged_step2['conflict_id'].combine_first(merged_step2['new_conflict_id']).astype(int)
merged_step2['location'] = merged_step2['location'].combine_first(merged_step2['location_prio'])
merged_step2['side_a'] = merged_step2['side_a'].combine_first(merged_step2['sidea'])
merged_step2['side_b'] = merged_step2['side_b'].combine_first(merged_step2['sideb'])
merged_step2['type_of_conflict'] = merged_step2['type_of_conflict'].combine_first(merged_step2['type'])
merged_step2['incompatibility'] = merged_step2['incompatibility'].combine_first(merged_step2['incomp'])

# ---------------------------------------------------------
# 5. Coalesce Casualty Fields
# ---------------------------------------------------------
merged_step2['deaths_best'] = merged_step2['bd_best'].combine_first(merged_step2['bdeadbes'])
merged_step2['deaths_low'] = merged_step2['bd_low'].combine_first(merged_step2['bdeadlow'])
merged_step2['deaths_high'] = merged_step2['bd_high'].combine_first(merged_step2['bdeadhig'])

# Best continuous estimate: deaths_best if known, else midpoint of low and high
merged_step2['deaths_estimate'] = merged_step2['deaths_best'].copy()
range_mask = merged_step2['deaths_estimate'].isna() & (merged_step2['deaths_low'].notna() | merged_step2['deaths_high'].notna())
merged_step2.loc[range_mask, 'deaths_estimate'] = (
    merged_step2.loc[range_mask, 'deaths_low'].fillna(0) + merged_step2.loc[range_mask, 'deaths_high'].fillna(0)
) / 2

def determine_source(row):
    if pd.notna(row['bd_best']) or pd.notna(row['bd_low']) or pd.notna(row['bd_high']):
        return 'UCDP_v26.1'
    elif pd.notna(row['bdeadbes']):
        return 'PRIO_v3.1 (point)'
    elif pd.notna(row['bdeadlow']) or pd.notna(row['bdeadhig']):
        return 'PRIO_v3.1 (range)'
    return 'No Casualty Data'

merged_step2['deaths_source'] = merged_step2.apply(determine_source, axis=1)

df_combined = merged_step2.sort_values(by=['year', 'conflict_id']).reset_index(drop=True)

# ---------------------------------------------------------
# 6. Save Data
# ---------------------------------------------------------
df_combined.to_csv("data/conflict_deaths_1946_2025.csv", index=False)
df_combined.to_csv("conflict_deaths_1946_2025.csv", index=False)

print(f"Successfully exported unified conflict deaths dataset ({len(df_combined)} rows)!")
