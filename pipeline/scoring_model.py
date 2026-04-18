"""
Materials Scoring Model
Multi-objective weighted scoring for material recommendations.
Scores materials based on stability, electronic properties, and practical usability.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalise(df, columns):
    """Normalise selected columns to 0-1 range."""
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[columns] = scaler.fit_transform(df[columns].fillna(0))
    return df_norm

def compute_scores(df, weights=None):
    """
    Compute multi-objective score for each material.
    
    Score = w1 * stability_score + w2 * band_gap_score + w3 * density_score
    
    - stability_score: inverse of energy_above_hull (lower = more stable = better)
    - band_gap_score: higher band gap = better electronic property
    - density_score: balanced (not too heavy, not too light — target ~5 g/cm3)
    """
    if weights is None:
        weights = {'stability': 0.5, 'band_gap': 0.3, 'density': 0.2}
    
    df = df.copy()
    
    # 1. Stability: invert energy_above_hull (lower is better)
    df['stability_raw'] = 1 / (1 + df['energy_above_hull'].fillna(0))
    
    # 2. Band gap: higher is better for semiconductors (cap at 5 eV)
    df['band_gap_raw'] = df['band_gap'].fillna(0).clip(0, 5)
    
    # 3. Density: target 5 g/cm3 — penalise extremes
    df['density_raw'] = 1 - abs(df['density'].fillna(5) - 5) / 10
    df['density_raw'] = df['density_raw'].clip(0, 1)
    
    # Normalise all three to 0-1
    df_norm = normalise(df, ['stability_raw', 'band_gap_raw', 'density_raw'])
    
    # Weighted final score
    df['final_score'] = (
        weights['stability'] * df_norm['stability_raw'] +
        weights['band_gap'] * df_norm['band_gap_raw'] +
        weights['density'] * df_norm['density_raw']
    )
    
    # Individual component scores (for explainability later)
    df['stability_score'] = df_norm['stability_raw']
    df['band_gap_score'] = df_norm['band_gap_raw']
    df['density_score'] = df_norm['density_raw']
    
    # RAG classification
    df['rag_status'] = pd.cut(
        df['final_score'],
        bins=[-0.01, 0.4, 0.7, 1.01],
        labels=['Red', 'Amber', 'Green']
    )
    
    return df

def get_top_materials(df, n=10, element_filter=None):
    """Return top N materials, optionally filtered by element presence."""
    result = df.copy()
    
    if element_filter:
        result = result[result['elements'].str.contains(element_filter, na=False)]
    
    result = result.sort_values('final_score', ascending=False).head(n)
    return result[['material_id', 'formula', 'final_score', 'stability_score', 
                   'band_gap_score', 'density_score', 'rag_status', 'elements']]

def rank_materials(input_csv='data/materials_features.csv', 
                   output_csv='data/materials_scored.csv'):
    print("Loading engineered features...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials.")
    
    print("Computing multi-objective scores...")
    df_scored = compute_scores(df)
    
    # Save full scored dataset
    df_scored.to_csv(output_csv, index=False)
    print(f"Saved scored dataset to {output_csv} ✅")
    
    # Show top 10 overall
    print("\n🏆 TOP 10 MATERIALS OVERALL:")
    print(get_top_materials(df_scored, n=10).to_string(index=False))
    
    # Show top 5 containing lithium (for the PNNL story)
    print("\n⚡ TOP 5 LITHIUM-CONTAINING MATERIALS:")
    top_li = get_top_materials(df_scored, n=5, element_filter='Li')
    if len(top_li) > 0:
        print(top_li.to_string(index=False))
    else:
        print("No lithium materials found in current dataset.")
    
    # RAG distribution
    print("\n📊 RAG DISTRIBUTION:")
    print(df_scored['rag_status'].value_counts())
    
    return df_scored

if __name__ == "__main__":
    rank_materials()