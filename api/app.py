"""
Materials Discovery API
Flask REST API serving the ML scoring and SHAP explanation backend.
Endpoints:
  GET  /health                     - health check
  POST /recommend                  - recommend materials by criteria
  POST /explain                    - explain why a material was recommended
  POST /compare                    - compare two materials
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys

# Add parent directory to path so we can import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Copilot Studio

# Load pre-computed data at startup
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
df_scored = pd.read_csv(os.path.join(DATA_PATH, 'materials_scored.csv'))
df_explanations = pd.read_csv(os.path.join(DATA_PATH, 'material_explanations.csv'))

print(f"✅ Loaded {len(df_scored)} materials")
print(f"✅ Loaded {len(df_explanations)} pre-computed explanations")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'materials_loaded': len(df_scored),
        'api_version': '1.0'
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommend top materials based on criteria.
    
    Body:
    {
        "element": "Li",      (optional, filter by element)
        "top_n": 3,           (default 3)
        "min_score": 0.5      (optional minimum score)
    }
    """
    data = request.get_json() or {}
    element = data.get('element')
    top_n = int(data.get('top_n', 3))
    min_score = float(data.get('min_score', 0))
    
    result = df_scored.copy()
    
    if element:
        result = result[result['elements'].str.contains(element, na=False)]
    
    if min_score > 0:
        result = result[result['final_score'] >= min_score]
    
    result = result.sort_values('final_score', ascending=False).head(top_n)
    
    materials = []
    for _, row in result.iterrows():
        materials.append({
            'material_id': row['material_id'],
            'formula': row['formula'],
            'final_score': round(float(row['final_score']), 4),
            'stability_score': round(float(row['stability_score']), 4),
            'band_gap_score': round(float(row['band_gap_score']), 4),
            'density_score': round(float(row['density_score']), 4),
            'rag_status': str(row['rag_status']),
            'band_gap': round(float(row['band_gap']), 3),
            'density': round(float(row['density']), 3),
            'energy_above_hull': round(float(row['energy_above_hull']), 4),
            'elements': str(row['elements'])
        })
    
    return jsonify({
        'count': len(materials),
        'filter_element': element,
        'materials': materials
    })


@app.route('/explain', methods=['POST'])
def explain():
    """
    Get natural language explanation for a material.
    
    Body:
    {
        "material_id": "mp-1234"
    }
    """
    data = request.get_json() or {}
    material_id = data.get('material_id')
    
    if not material_id:
        return jsonify({'error': 'material_id is required'}), 400
    
    exp_row = df_explanations[df_explanations['material_id'] == material_id]
    
    if len(exp_row) == 0:
        return jsonify({
            'error': f'No pre-computed explanation for {material_id}. Only top materials are explained.'
        }), 404
    
    exp_row = exp_row.iloc[0]
    return jsonify({
        'material_id': material_id,
        'formula': exp_row['formula'],
        'final_score': round(float(exp_row['final_score']), 4),
        'explanation': exp_row['explanation_text']
    })


@app.route('/compare', methods=['POST'])
def compare():
    """
    Compare two materials side by side.
    
    Body:
    {
        "material_a": "mp-1234",
        "material_b": "mp-5678"
    }
    """
    data = request.get_json() or {}
    mat_a = data.get('material_a')
    mat_b = data.get('material_b')
    
    if not mat_a or not mat_b:
        return jsonify({'error': 'material_a and material_b required'}), 400
    
    row_a = df_scored[df_scored['material_id'] == mat_a]
    row_b = df_scored[df_scored['material_id'] == mat_b]
    
    if len(row_a) == 0 or len(row_b) == 0:
        return jsonify({'error': 'One or both materials not found'}), 404
    
    a, b = row_a.iloc[0], row_b.iloc[0]
    
    # Build comparison
    winner = a['formula'] if a['final_score'] > b['final_score'] else b['formula']
    diff = abs(a['final_score'] - b['final_score'])
    
    return jsonify({
        'material_a': {
            'formula': a['formula'],
            'final_score': round(float(a['final_score']), 4),
            'rag_status': str(a['rag_status']),
            'band_gap': round(float(a['band_gap']), 3),
            'density': round(float(a['density']), 3),
        },
        'material_b': {
            'formula': b['formula'],
            'final_score': round(float(b['final_score']), 4),
            'rag_status': str(b['rag_status']),
            'band_gap': round(float(b['band_gap']), 3),
            'density': round(float(b['density']), 3),
        },
        'summary': f"{winner} scores higher by {diff:.4f} points."
    })


if __name__ == '__main__':
    print("\n🚀 Materials Discovery API starting...")
    print("   Endpoints: /health, /recommend, /explain, /compare")
    print("   Running on http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)