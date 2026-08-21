import json
from rdkit import Chem
from rdkit.Chem import Descriptors


def compute_sascore(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        from sascorer import calculateScore
        return calculateScore(mol)
    except Exception:
        return 10.0

with open('/home/zyluo/code/constrained-dllm/evaluation/results/cdd_eval_results_20260531_153535.json', 'r') as f:
    data = json.load(f)

cdd_samples = data['cdd_samples']
print(f"Total CDD samples: {len(cdd_samples)}")
print("\nCDD Samples SA Scores:")
print("-" * 80)

all_below_4 = True
for i, smi in enumerate(cdd_samples):
    smi_clean = smi.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()
    sa = compute_sascore(smi_clean)
    status = "OK" if sa < 4.0 else "ABOVE 4.0"
    if sa >= 4.0:
        all_below_4 = False
    print(f"[{i+1:2d}] SA: {sa:.4f} | {status} | SMILES: {smi_clean}")

print("-" * 80)
print(f"\nAll SA scores < 4.0: {'YES' if all_below_4 else 'NO'}")