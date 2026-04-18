"""Quick inspection of what elements are in our dataset."""
import pandas as pd

df = pd.read_csv('data/materials_scored.csv')

# Collect all unique elements
all_elements = set()
for e in df['elements']:
    all_elements.update(str(e).split(', '))

print(f"Total unique elements: {len(all_elements)}")
print(f"\nAll elements present:")
print(sorted(all_elements))

print(f"\n🔍 Element checks:")
for el in ['Li', 'Fe', 'O', 'Na', 'Si', 'Mg', 'Ca', 'Al']:
    count = df['elements'].str.contains(el, na=False).sum()
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {el}: {count} materials")

print(f"\n📊 Sample of formulas:")
print(df['formula'].head(20).tolist())