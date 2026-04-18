"""
Fetch diverse materials from the Materials Project API.
Targets battery-relevant elements (Li, Na, Mg, etc.) to align with the 
Microsoft x PNNL lithium-reduction story.
"""

import os
import pandas as pd
import duckdb
from mp_api.client import MPRester

API_KEY = os.getenv("MP_API_KEY")

# Target elements: battery-relevant + common semiconductors + common metals
TARGET_ELEMENTS = [
    "Li", "Na", "K", "Mg", "Ca",           # Alkali & alkaline earth (battery relevant)
    "Fe", "Co", "Ni", "Mn", "Cu", "Zn",    # Transition metals (cathodes)
    "Al", "Si", "P", "S",                  # Common in batteries/semiconductors
    "O", "F", "Cl"                         # Common anions
]

def fetch_materials():
    print("Connecting to Materials Project API...")
    all_docs = []

    with MPRester(API_KEY) as mpr:
        for element in TARGET_ELEMENTS:
            print(f"  Fetching materials containing {element}...")
            try:
                docs = mpr.materials.summary.search(
                    elements=[element],
                    energy_above_hull=(0, 0.1),
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
                    chunk_size=40  # ~40 per element = ~720 total before dedup
                )
                all_docs.extend(docs)
            except Exception as e:
                print(f"    ⚠️ Failed to fetch {element}: {e}")

    print(f"\nFetched {len(all_docs)} materials (with duplicates).")
    
    # Deduplicate by material_id
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.material_id not in seen:
            seen.add(doc.material_id)
            unique_docs.append(doc)
    
    print(f"After deduplication: {len(unique_docs)} unique materials.")

    # Convert to DataFrame
    records = []
    for doc in unique_docs:
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
    print(f"\nSample of formulas: {df['formula'].head(10).tolist()}")

    # Save to CSV
    df.to_csv("data/materials_data.csv", index=False)
    print("Saved to data/materials_data.csv ✅")

    # Save to DuckDB
    con = duckdb.connect("data/materials.db")
    con.execute("DROP TABLE IF EXISTS materials")
    con.execute("CREATE TABLE materials AS SELECT * FROM df")
    con.close()
    print("Saved to data/materials.db ✅")

    # Quick element check
    print("\n🔍 Element coverage:")
    for el in ["Li", "Na", "Mg", "Fe", "O", "Si"]:
        count = df['elements'].str.contains(el, na=False).sum()
        print(f"  {el}: {count} materials")

    return df


if __name__ == "__main__":
    fetch_materials()