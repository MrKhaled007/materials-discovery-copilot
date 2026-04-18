"""
SHAP Explainability for Materials Scoring
Explains why a specific material received its score.
Generates both numerical attributions and plain English explanations.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Features used for the model
FEATURE_COLS = [
    'energy_above_hull',
    'band_gap',
    'density',
    'volume',
    'avg_atomic_number',
    'avg_electronegativity',
    'avg_atomic_radius',
    'avg_atomic_mass',
    'num_elements'
]

def train_surrogate_model(df):
    """
    Train a Gradient Boosting model to learn the weighted scoring function.
    This surrogate model allows us to apply SHAP to our rule-based scorer.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df['final_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  Surrogate model R² (train): {train_score:.4f}")
    print(f"  Surrogate model R² (test):  {test_score:.4f}")
    
    return model, X

def explain_material(model, X, df, material_id, save_plot=True):
    """Generate SHAP explanation for a specific material."""
    explainer = shap.TreeExplainer(model)
    
    # Get the index of this material
    idx = df.index[df['material_id'] == material_id]
    if len(idx) == 0:
        return None
    idx = idx[0]
    
    shap_values = explainer.shap_values(X.iloc[[idx]])
    # Ensure base_value is a scalar float (not array)
    base_value = explainer.expected_value
    if hasattr(base_value, '__len__'):
        base_value = float(base_value[0])
    else:
        base_value = float(base_value)
    
    material = df.loc[idx]
    contributions = dict(zip(FEATURE_COLS, shap_values[0]))
    
    # Save waterfall plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=np.array(shap_values[0]),
                base_values=base_value,
                data=X.iloc[idx].values,
                feature_names=FEATURE_COLS
            ),
            show=False
        )
        plt.title(f"Why was {material['formula']} recommended?", fontsize=12)
        plt.tight_layout()
        plot_path = f"data/shap_{material_id}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved SHAP plot: {plot_path}")
    
    return {
        'material_id': material_id,
        'formula': material['formula'],
        'final_score': material['final_score'],
        'base_value': base_value,
        'contributions': contributions,
        'top_positive': sorted(contributions.items(), key=lambda x: -x[1])[:3],
        'top_negative': sorted(contributions.items(), key=lambda x: x[1])[:3],
    }

def generate_english_explanation(explanation):
    """Convert SHAP values into plain English a Copilot agent can repeat."""
    formula = explanation['formula']
    score = explanation['final_score']
    
    # Friendly feature name mapping
    feature_names = {
        'energy_above_hull': 'thermodynamic stability',
        'band_gap': 'band gap (electronic property)',
        'density': 'material density',
        'volume': 'unit cell volume',
        'avg_atomic_number': 'average atomic number',
        'avg_electronegativity': 'electronegativity',
        'avg_atomic_radius': 'atomic radius',
        'avg_atomic_mass': 'average atomic mass',
        'num_elements': 'compositional complexity'
    }
    
    positive_reasons = []
    for feat, val in explanation['top_positive']:
        if val > 0.001:
            positive_reasons.append(feature_names.get(feat, feat))
    
    negative_reasons = []
    for feat, val in explanation['top_negative']:
        if val < -0.001:
            negative_reasons.append(feature_names.get(feat, feat))
    
    # Build explanation
    text = f"**{formula}** received a score of {score:.3f}.\n\n"
    
    if positive_reasons:
        text += f"✅ **Strengths:** This material scored well primarily due to its "
        text += ", ".join(positive_reasons[:2])
        text += ".\n\n"
    
    if negative_reasons:
        text += f"⚠️ **Considerations:** The score was reduced by "
        text += ", ".join(negative_reasons[:2])
        text += ".\n\n"
    
    # Recommendation
    if score > 0.7:
        text += "🟢 **Recommendation:** Strong candidate — suitable for further investigation."
    elif score > 0.4:
        text += "🟡 **Recommendation:** Moderate candidate — review trade-offs before use."
    else:
        text += "🔴 **Recommendation:** Weak candidate — consider alternatives."
    
    return text

def explain_top_materials(input_csv='data/materials_scored.csv', n=5):
    print("Loading scored materials...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials.\n")
    
    print("Training SHAP surrogate model...")
    model, X = train_surrogate_model(df)
    
    # Get top N materials
    top_materials = df.sort_values('final_score', ascending=False).head(n)
    
    print(f"\nGenerating SHAP explanations for top {n} materials...\n")
    print("=" * 70)
    
    explanations = []
    for _, row in top_materials.iterrows():
        explanation = explain_material(model, X, df, row['material_id'], save_plot=True)
        if explanation:
            english = generate_english_explanation(explanation)
            print(f"\n{english}")
            print("-" * 70)
            
            # Save for agent consumption
            explanations.append({
                'material_id': explanation['material_id'],
                'formula': explanation['formula'],
                'final_score': explanation['final_score'],
                'explanation_text': english,
                **{f'shap_{k}': v for k, v in explanation['contributions'].items()}
            })
    
    # Save all explanations
    exp_df = pd.DataFrame(explanations)
    exp_df.to_csv('data/material_explanations.csv', index=False)
    print(f"\n💾 Saved {len(explanations)} explanations to data/material_explanations.csv ✅")
    
    return exp_df

if __name__ == "__main__":
    explain_top_materials(n=5)