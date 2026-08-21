#!/usr/bin/env python
"""CDD Evaluation for UDLM-QM9."""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem as rdChem
from rdkit import rdBase
from transformers import AutoModelForMaskedLM, AutoTokenizer
import datasets

rdBase.DisableLog('rdApp.error')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'cdd'))

from cdd.alm_projection import project_to_constraint
from cdd.delta_g import make_delta_g_fn


def compute_sascore(smiles):
    try:
        mol = rdChem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        from sascorer import calculateScore
        return calculateScore(mol)
    except Exception:
        return 10.0


def compute_qed(smiles):
    try:
        mol = rdChem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        from rdkit.Chem import QED
        return QED.qed(mol)
    except Exception:
        return 0.0


def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _compute_posterior(x_theta, xt, alpha_s, alpha_t, vocab_size):
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, vocab_size)
    uniform_dist = 1.0 / vocab_size
    return (
        (alpha_t * vocab_size * x_theta * xt_one_hot +
         (alpha_ts - alpha_t) * xt_one_hot +
         d_alpha * x_theta +
         (1 - alpha_ts) * (1 - alpha_s) * uniform_dist)
        /
        (alpha_t * vocab_size * torch.gather(x_theta, -1, xt[..., None]) +
         (1 - alpha_t))
    )


def _loglinear_sigma(t, eps=1e-3):
    return -torch.log1p(-(1 - eps) * t)


def sample_udlm_qm9(model, tokenizer, device, delta_g_fn=None, num_steps=128, batch_size=16):
    """Sample from UDLM-QM9 using discrete diffusion with optional CDD."""
    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    vocab_size = tokenizer.vocab_size

    xt = torch.randint(0, vocab_size, (batch_size, 32), device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = (1 - eps) / num_steps
        t_tensor = t * torch.ones(batch_size, device=device)

        with torch.no_grad():
            sigma_t = _loglinear_sigma(t_tensor)
            logits = model(xt, sigma_t, None)
            x_theta = logits.log_softmax(dim=-1).exp()

            sigma_s = _loglinear_sigma((t - dt) * torch.ones(batch_size, device=device))
            move_chance_t = 1 - torch.exp(-sigma_t)
            move_chance_s = 1 - torch.exp(-sigma_s)

            alpha_s = 1 - move_chance_s
            alpha_t = 1 - move_chance_t

            q_xs = _compute_posterior(
                x_theta=x_theta,
                xt=xt,
                alpha_s=alpha_s[:, None, None],
                alpha_t=alpha_t[:, None, None],
                vocab_size=vocab_size
            )

            if delta_g_fn is not None:
                with torch.enable_grad():
                    q_xs = project_to_constraint(
                        q_xs,
                        delta_g_fn,
                        λ_init=CDD_LAMBDA_INIT,
                        μ_init=CDD_MU_INIT,
                        μ_max=CDD_MU_MAX,
                        inner_iter_max=CDD_INNER_ITER,
                        outer_iter_max=CDD_OUTER_ITER,
                        η=CDD_LR,
                        eps=CDD_EPS,
                        gumbel_temperature=CDD_GUMBEL_TEMP,
                        verbose=False
                    )

            xt = _sample_categorical(q_xs)

    return xt


def evaluate_samples(samples, dataset_smiles, tau=3.5):
    """Evaluate generated samples."""
    valids, invalids, sascores, qeds, violations = [], [], [], [], []

    for smi in samples:
        smi_clean = smi.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()
        try:
            mol = rdChem.MolFromSmiles(smi_clean)
            if mol is not None and len(smi_clean) > 0:
                valids.append(smi_clean)
                sa = compute_sascore(smi_clean)
                sascores.append(sa)
                qeds.append(compute_qed(smi_clean))
                violations.append(1.0 if sa > tau else 0.0)
            else:
                invalids.append(smi_clean)
        except Exception:
            invalids.append(smi_clean)

    valid_count = len(valids)
    unique = len(set(valids))
    novel = len(set(valids) - set(dataset_smiles))

    return {
        'valid': valids,
        'invalids': invalids,
        'valid_pct': valid_count / len(samples) if len(samples) > 0 else 0,
        'unique_pct': unique / valid_count if valid_count > 0 else 0,
        'novel_pct': novel / valid_count if valid_count > 0 else 0,
        'sascores': sascores,
        'qeds': qeds,
        'unique': unique,
        'novel': novel,
        'violations': violations,
        'violation_pct': sum(violations) / len(violations) if violations else 0.0,
    }

# ===== CDD hyperparameters =====
CDD_GUMBEL_TEMP = float(os.environ.get('CDD_GUMBEL_TEMP', '0.5'))
CDD_LAMBDA_INIT = 0.0
CDD_MU_INIT = 1.0
CDD_MU_MAX = 1000.0
CDD_RHO = 2.0
CDD_INNER_ITER = 100
CDD_OUTER_ITER = 30
CDD_LR = 1.0
CDD_EPS = 0.001
TAU = float(os.environ.get('TAU', '3.5'))


def main():
    print("=" * 60)
    print("CDD Evaluation - UDLM-QM9 with Real SASCORE Constraint")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('yairschiff/qm9-tokenizer', trust_remote_code=True)

    print("\n2. Loading UDLM-QM9 model...")
    model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/udlm-qm9', trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print("\n3. Preparing CDD config...")
    sascore_ckpt = str(Path(__file__).parent.parent / 'cdd' / 'sascore' / 'checkpoints' / 'run_20260504_052230' / 'best_model.pt')
    print(f"   SA surrogate ckpt:    {sascore_ckpt}")
    print(f"   Tau (SA threshold):   {TAU}")

    print("\n4. Loading QM9 dataset...")
    qm9_dataset = datasets.load_dataset('yairschiff/qm9', split='train')
    dataset_smiles = set(qm9_dataset['canonical_smiles'])
    print(f"   Loaded {len(dataset_smiles)} canonical SMILES")

    BATCH_SIZE = 1
    NUM_BATCHES = int(os.environ.get('NUM_BATCHES', '64'))
    SAMPLING_STEPS = int(os.environ.get('SAMPLING_STEPS', '128'))

    print("\n" + "=" * 60)
    print("Hyperparameters")
    print("=" * 60)
    print(f"  Base model:           kuleshov-group/udlm-qm9")
    print(f"  SA surrogate ckpt:    {sascore_ckpt}")
    print(f"  Tau (SA threshold):  {TAU}")
    print(f"  Sampling steps:      {SAMPLING_STEPS}")
    print(f"  Num batches:         {NUM_BATCHES}")
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Total samples:       {BATCH_SIZE * NUM_BATCHES}")
    print(f"")
    print(f"  CDD gumbel temperature:     {CDD_GUMBEL_TEMP}")
    print(f"  CDD lambda_init:     {CDD_LAMBDA_INIT}")
    print(f"  CDD mu_init:         {CDD_MU_INIT}")
    print(f"  CDD mu_max:          {CDD_MU_MAX}")
    print(f"  CDD rho:             {CDD_RHO}")
    print(f"  CDD inner_iter_max:  {CDD_INNER_ITER}")
    print(f"  CDD outer_iter_max:  {CDD_OUTER_ITER}")
    print(f"  CDD lr:              {CDD_LR}")
    print(f"  CDD eps:             {CDD_EPS}")
    print(f"")
    print(f"  _sample_categorical: Gumbel-max")
    print(f"  Noise schedule:      LogLinear (eps=1e-3)")

    print("\n" + "=" * 60)
    print("5. Running sampling without CDD (baseline)...")
    print("=" * 60)

    samples_baseline = []
    start_time = time.time()
    for batch_idx in range(NUM_BATCHES):
        xt = sample_udlm_qm9(model, tokenizer, device, delta_g_fn=None, num_steps=SAMPLING_STEPS, batch_size=BATCH_SIZE)
        smi = tokenizer.decode(xt[0].cpu().tolist())
        samples_baseline.append(smi)
        _validate = False
        try:
            mol = rdChem.MolFromSmiles(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
            if mol is not None:
                _validate = True
                _sa = compute_sascore(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
                _qed = compute_qed(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
                _viol = "Y" if _sa > TAU else "N"
                print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:Y | Violation:{_viol} | SA:{_sa:.3f} | QED:{_qed:.3f}")
            else:
                print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:N")
        except:
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Error")

    time_baseline = time.time() - start_time
    results_baseline = evaluate_samples(samples_baseline, dataset_smiles, TAU)

    print(f"\n   Baseline Results:")
    print(f"   Valid: {results_baseline['valid_pct']*100:.1f}%")
    print(f"   Unique: {results_baseline['unique_pct']*100:.1f}%")
    print(f"   Novel: {results_baseline['novel_pct']*100:.1f}%")
    if results_baseline['sascores']:
        print(f"   SA Score Mean: {sum(results_baseline['sascores'])/len(results_baseline['sascores']):.3f}")
        print(f"   QED Mean: {sum(results_baseline['qeds'])/len(results_baseline['qeds']):.3f}")

    print("\n" + "=" * 60)
    print("6. Running sampling with CDD...")
    print("=" * 60)

    delta_g_fn = make_delta_g_fn(sascore_ckpt, TAU)

    samples_cdd = []
    start_time = time.time()
    for batch_idx in range(NUM_BATCHES):
        xt = sample_udlm_qm9(model, tokenizer, device, delta_g_fn=delta_g_fn, num_steps=SAMPLING_STEPS, batch_size=BATCH_SIZE)
        smi = tokenizer.decode(xt[0].cpu().tolist())
        samples_cdd.append(smi)
        try:
            mol = rdChem.MolFromSmiles(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
            if mol is not None:
                _sa = compute_sascore(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
                _qed = compute_qed(smi.replace('<bos>','').replace('<eos>','').replace('<pad>','').strip())
                _viol = "Y" if _sa > TAU else "N"
                print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:Y | Violation:{_viol} | SA:{_sa:.3f} | QED:{_qed:.3f}")
            else:
                print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:N")
        except:
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Error")

    time_cdd = time.time() - start_time
    results_cdd = evaluate_samples(samples_cdd, dataset_smiles, TAU)

    print(f"\n   CDD Results:")
    print(f"   Valid: {results_cdd['valid_pct']*100:.1f}%")
    print(f"   Unique: {results_cdd['unique_pct']*100:.1f}%")
    print(f"   Novel: {results_cdd['novel_pct']*100:.1f}%")
    print(f"   Violation Rate: {results_cdd['violation_pct']*100:.1f}%")
    if results_cdd['sascores']:
        print(f"   SA Score Mean: {sum(results_cdd['sascores'])/len(results_cdd['sascores']):.3f}")
        print(f"   QED Mean: {sum(results_cdd['qeds'])/len(results_cdd['qeds']):.3f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    sa_bl = results_baseline['sascores']
    sa_cd = results_cdd['sascores']
    sa_bl_mean = sum(sa_bl)/len(sa_bl) if sa_bl else -1
    sa_cd_mean = sum(sa_cd)/len(sa_cd) if sa_cd else -1
    sa_bl_std = (sum((s - sa_bl_mean)**2 for s in sa_bl) / len(sa_bl))**0.5 if len(sa_bl) > 1 else 0
    sa_cd_std = (sum((s - sa_cd_mean)**2 for s in sa_cd) / len(sa_cd))**0.5 if len(sa_cd) > 1 else 0

    qed_bl_mean = sum(results_baseline['qeds'])/len(results_baseline['qeds']) if results_baseline['qeds'] else -1
    qed_cd_mean = sum(results_cdd['qeds'])/len(results_cdd['qeds']) if results_cdd['qeds'] else -1

    print(f"{'Metric':<20} {'Baseline':<15} {'CDD':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Valid %':<20} {results_baseline['valid_pct']*100:.1f}{'':>10} {results_cdd['valid_pct']*100:.1f}{'':>10} {(results_cdd['valid_pct']-results_baseline['valid_pct'])*100:+.1f}")
    print(f"{'Unique %':<20} {results_baseline['unique_pct']*100:.1f}{'':>10} {results_cdd['unique_pct']*100:.1f}{'':>10} {(results_cdd['unique_pct']-results_baseline['unique_pct'])*100:+.1f}")
    print(f"{'Novel %':<20} {results_baseline['novel_pct']*100:.1f}{'':>10} {results_cdd['novel_pct']*100:.1f}{'':>10} {(results_cdd['novel_pct']-results_baseline['novel_pct'])*100:+.1f}")
    print(f"{'SA Score Mean':<20} {sa_bl_mean:.3f}{'':>12} {sa_cd_mean:.3f}{'':>12} {sa_cd_mean-sa_bl_mean:+.3f}")
    print(f"{'SA Score Std':<20} {sa_bl_std:.3f}{'':>12} {sa_cd_std:.3f}{'':>12}")
    print(f"{'QED Mean':<20} {qed_bl_mean:.3f}{'':>12} {qed_cd_mean:.3f}{'':>12} {qed_cd_mean-qed_bl_mean:+.3f}")
    print(f"{'Violation %':<20} {results_baseline['violation_pct']*100:.1f}{'':>10} {results_cdd['violation_pct']*100:.1f}{'':>10} {(results_cdd['violation_pct']-results_baseline['violation_pct'])*100:+.1f}")
    print(f"{'Time (s)':<20} {time_baseline:.2f}{'':>13} {time_cdd:.2f}{'':>13}")
    print("-" * 65)

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'cdd_eval_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'baseline': {'valid_pct': results_baseline['valid_pct'], 'unique_pct': results_baseline['unique_pct'],
                             'novel_pct': results_baseline['novel_pct'], 'sa_mean': sa_bl_mean, 'sa_std': sa_bl_std,
                             'qed_mean': qed_bl_mean, 'time': time_baseline, 'num_samples': len(samples_baseline),
                             'violation_pct': results_baseline['violation_pct']},
                'cdd': {'valid_pct': results_cdd['valid_pct'], 'unique_pct': results_cdd['unique_pct'],
                        'novel_pct': results_cdd['novel_pct'], 'sa_mean': sa_cd_mean, 'sa_std': sa_cd_std,
                        'qed_mean': qed_cd_mean, 'time': time_cdd, 'num_samples': len(samples_cdd),
                        'violation_pct': results_cdd['violation_pct']},
            },
            'baseline_samples': samples_baseline,
            'cdd_samples': samples_cdd,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()