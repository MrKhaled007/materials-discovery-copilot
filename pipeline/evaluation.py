"""
Evaluation Metrics for Materials Scoring Model
- Precision@K: Quality of top-K recommendations
- Spearman Correlation: Ranking agreement with domain heuristic
- Coverage: Proportion of materials classified as usable (Green/Amber)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

def domain_heuristic_ranking(df):
    """
    Domain-expert heuristic ranking (ground truth proxy).
    A 'good' material is one that is stable AND has non-zero band gap AND reasonable density.
    """
    df = df.copy()
    df['heuristic_score'] = (
        (df['energy_above_hull'] < 0.05).astype(int) * 2 +   # very stable
        (df['band_gap'].between(0.5, 4.0)).astype(int) * 2 + # useful semiconductor range
        (df['density'].between(2, 8)).astype(int) * 1        # reasonable density
    )
    return df

def precision_at_k(df, k=3):
    """
    Precision@K: Of the top K materials our model recommends,
    how many are actually 'relevant' (heuristic score >= 3)?
    """
    df = df.copy()
    df_sorted = df.sort_values('final_score', ascending=False).head(k)
    relevant = (df_sorted['heuristic_score'] >= 3).sum()
    return relevant / k

def spearman_correlation(df):
    """
    Spearman rank correlation between model score and domain heuristic.
    Measures ranking agreement between our model and domain logic.
    Range: -1 (opposite) to 1 (perfect agreement).
    """
    corr, p_value = spearmanr(df['final_score'], df['heuristic_score'])
    return corr, p_value

def usability_coverage(df):
    """Proportion of materials classified as usable (Green or Amber)."""
    usable = df['rag_status'].isin(['Green', 'Amber']).sum()
    return usable / len(df)

def evaluate_for_element(df, element_symbol, k=3):
    """Evaluate model performance for a specific element (e.g. 'Li')."""
    subset = df[df['elements'].str.contains(element_symbol, na=False)]
    if len(subset) < k:
        return None
    
    p_at_k = precision_at_k(subset, k=k)
    corr, _ = spearman_correlation(subset)
    return {
        'element': element_symbol,
        'material_count': len(subset),
        f'precision_at_{k}': p_at_k,
        'spearman_correlation': corr
    }

def run_evaluation(input_csv='data/materials_scored.csv'):
    print("Loading scored materials...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials.\n")
    
    # Compute domain heuristic ranking
    df = domain_heuristic_ranking(df)
    
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # 1. Precision@K at different K values
    print("\n📊 PRECISION@K (Top-K recommendation quality):")
    for k in [3, 5, 10]:
        p = precision_at_k(df, k=k)
        print(f"  Precision@{k}:  {p:.2%}")
    
    # 2. Spearman correlation
    print("\n🔗 SPEARMAN RANK CORRELATION:")
    corr, p_value = spearman_correlation(df)
    print(f"  Correlation:   {corr:.4f}")
    print(f"  P-value:       {p_value:.4e}")
    interpretation = (
        "Strong agreement ✅" if corr > 0.6 else
        "Moderate agreement ⚠️" if corr > 0.3 else
        "Weak agreement ❌"
    )
    print(f"  Interpretation: {interpretation}")
    
    # 3. Coverage
    print("\n🟢 USABILITY COVERAGE:")
    coverage = usability_coverage(df)
    print(f"  Green/Amber materials: {coverage:.2%}")
    
    # 4. Per-element evaluation
    print("\n🧪 PER-ELEMENT PERFORMANCE:")
    for el in ['Li', 'Fe', 'O', 'Na', 'Si']:
        result = evaluate_for_element(df, el, k=3)
        if result:
            print(f"  {result['element']:<3} ({result['material_count']:>3} materials) | "
                  f"P@3: {result['precision_at_3']:.2%} | "
                  f"Spearman: {result['spearman_correlation']:.3f}")
    
    # Save evaluation report
    report = {
        'total_materials': len(df),
        'precision_at_3': precision_at_k(df, 3),
        'precision_at_5': precision_at_k(df, 5),
        'precision_at_10': precision_at_k(df, 10),
        'spearman_correlation': corr,
        'spearman_p_value': p_value,
        'usability_coverage': coverage,
    }
    report_df = pd.DataFrame([report])
    report_df.to_csv('data/evaluation_report.csv', index=False)
    print("\n💾 Saved evaluation report to data/evaluation_report.csv ✅")
    
    return report

if __name__ == "__main__":
    run_evaluation()