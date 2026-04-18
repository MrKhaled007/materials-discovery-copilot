import os
import pandas as pd
import duckdb
from mp_api.client import MPRester
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")

def fetch_materials():
    print("Connecting to Materials Project API...")
    
    with MPRester(API_KEY) as mpr:
        # Fetch materials with key properties
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.1),  # Stable materials only
            fields=[
                "material_id",
                "formula_pretty",
                "energy_above_hull",
                "band_gap",
                "density",
                "volume",
                "symmetry",
                "elements",
            ],
            num_chunks=1,
            chunk_size=500  # Fetch 500 materials
        )
    
    print(f"Fetched {len(docs)} materials.")
    
    # Convert to DataFrame
    records = []
    for doc in docs:
        records.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "energy_above_hull": doc.energy_above_hull,
            "band_gap": doc.band_gap,
            "density": doc.density,
            "volume": doc.volume,
            "crystal_system": doc.symmetry.crystal_system if doc.symmetry else None,
            "elements": ", ".join([str(e) for e in doc.elements])
        })
    
    df = pd.DataFrame(records)
    print(df.head())
    
    # Save to CSV
    df.to_csv("data/materials_data.csv", index=False)
    print("Saved to data/materials_data.csv ✅")
    
    # Save to DuckDB
    con = duckdb.connect("data/materials.db")
    con.execute("DROP TABLE IF EXISTS materials")
    con.execute("CREATE TABLE materials AS SELECT * FROM df")
    con.close()
    print("Saved to data/materials.db ✅")
    
    return df

if __name__ == "__main__":
    fetch_materials()