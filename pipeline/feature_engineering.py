"""
Feature Engineering for Materials Discovery
Converts raw materials data into ML-ready features using domain knowledge.
"""

import pandas as pd
import numpy as np
from mendeleev import element
import re

def parse_formula(formula):
    """Extract element symbols from a chemical formula like 'Li2FeO4' -> ['Li', 'Fe', 'O']"""
    elements = re.findall(r'[A-Z][a-z]?', formula)
    return elements

def get_element_properties(symbol):
    """Fetch atomic properties for an element."""
    try:
        el = element(symbol)
        return {
            'atomic_number': el.atomic_number,
            'electronegativity': el.electronegativity() or 0,
            'atomic_radius': el.atomic_radius or 0,
            'atomic_mass': el.atomic_weight or 0,
        }
    except Exception:
        return {'atomic_number': 0, 'electronegativity': 0, 'atomic_radius': 0, 'atomic_mass': 0}

def compute_material_features(formula):
    """Compute aggregated atomic features for a full material formula."""
    elements = parse_formula(formula)
    if not elements:
        return {'avg_atomic_number': 0, 'avg_electronegativity': 0, 
                'avg_atomic_radius': 0, 'avg_atomic_mass': 0, 'num_elements': 0}
    
    props = [get_element_properties(e) for e in elements]
    
    return {
        'avg_atomic_number': np.mean([p['atomic_number'] for p in props]),
        'avg_electronegativity': np.mean([p['electronegativity'] for p in props]),
        'avg_atomic_radius': np.mean([p['atomic_radius'] for p in props]),
        'avg_atomic_mass': np.mean([p['atomic_mass'] for p in props]),
        'num_elements': len(elements),
    }

def engineer_features(input_csv='data/materials_data.csv', output_csv='data/materials_features.csv'):
    print("Loading raw materials data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials.")
    
    print("Engineering atomic features... (this may take a minute)")
    feature_records = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Processing {i}/{len(df)}...")
        features = compute_material_features(row['formula'])
        feature_records.append(features)
    
    features_df = pd.DataFrame(feature_records)
    combined = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # Fill any NaN
    combined = combined.fillna(0)
    
    combined.to_csv(output_csv, index=False)
    print(f"Saved enriched features to {output_csv} ✅")
    print(f"Shape: {combined.shape}")
    print(combined.head())
    
    return combined

if __name__ == "__main__":
    engineer_features()