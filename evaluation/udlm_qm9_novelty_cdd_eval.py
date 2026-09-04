#!/usr/bin/env python
"""CDD Evaluation for UDLM-QM9 with Novelty Constraint (Section 5.2).

Base model: UDLM-QM9 (kuleshov-group/udlm-qm9)
Constraint: generated SMILES must be absent from the QM9 training set (and
            from previously generated samples in this run)
Metrics:    Valid & Novel (count), QED (novel molecules only), Violation %
            (valid but not novel, as a fraction of valid generations)
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from rdkit import Chem as rdChem
from rdkit import rdBase
from transformers import AutoModelForMaskedLM, AutoTokenizer
import datasets

rdBase.DisableLog('rdApp.error')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'cdd'))

from cdd.novelty import NoveltyProjector, load_seen_smiles


def compute_qed(smiles):
    try:
        mol = rdChem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        from rdkit.Chem import QED
        return QED.qed(mol)
    except Exception:
        return 0.0


def _clean(smi):
    return smi.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()


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


def sample_udlm_qm9_novelty(
    model,
    tokenizer,
    device,
    novelty_projector: NoveltyProjector = None,
    num_steps: int = 128,
    batch_size: int = 16,
):
    """Sample from UDLM-QM9 using discrete diffusion with optional novelty CDD."""
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

            if novelty_projector is not None:
                q_xs = novelty_projector.project(q_xs)

            xt = _sample_categorical(q_xs)

    return xt


def evaluate_samples_novelty(samples, dataset_smiles):
    """Evaluate generated samples for validity, novelty and QED."""
    valids, invalids, novel, violations, qeds_novel = [], [], [], [], []

    for smi in samples:
        smi_clean = _clean(smi)
        mol = rdChem.MolFromSmiles(smi_clean)
        if mol is not None and len(smi_clean) > 0:
            valids.append(smi_clean)
            is_novel = smi_clean not in dataset_smiles
            violations.append(0.0 if is_novel else 1.0)
            if is_novel:
                novel.append(smi_clean)
                qeds_novel.append(compute_qed(smi_clean))
        else:
            invalids.append(smi_clean)

    valid_count = len(valids)
    valid_novel_count = len(novel)

    return {
        'valid': valids,
        'invalids': invalids,
        'novel': novel,
        'valid_pct': valid_count / len(samples) if samples else 0.0,
        'valid_novel_count': valid_novel_count,
        'valid_novel_pct': valid_novel_count / len(samples) if samples else 0.0,
        'qeds_novel': qeds_novel,
        'qed_mean_novel': sum(qeds_novel) / len(qeds_novel) if qeds_novel else -1,
        'violations': violations,
        'violation_pct': sum(violations) / len(violations) if violations else 0.0,
    }


TOP_K = int(os.environ.get('TOP_K', '5'))
MAX_SEARCH_NODES = int(os.environ.get('MAX_SEARCH_NODES', '20000'))


def main():
    print("=" * 60)
    print("CDD Evaluation - UDLM-QM9 with Novelty Constraint")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('yairschiff/qm9-tokenizer', trust_remote_code=True)

    print("\n2. Loading UDLM-QM9 model...")
    model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/udlm-qm9', trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print("\n3. Loading QM9 dataset (training set = novelty database)...")
    dataset_smiles = load_seen_smiles(split='train')
    print(f"   Loaded {len(dataset_smiles)} canonical SMILES")

    BATCH_SIZE = 1
    NUM_BATCHES = int(os.environ.get('NUM_BATCHES', '64'))
    SAMPLING_STEPS = int(os.environ.get('SAMPLING_STEPS', '128'))

    print("\n" + "=" * 60)
    print("Hyperparameters")
    print("=" * 60)
    print(f"  Base model:           kuleshov-group/udlm-qm9")
    print(f"  Constraint:           novelty (SMILES not in training set)")
    print(f"  Sampling steps:       {SAMPLING_STEPS}")
    print(f"  Num batches:          {NUM_BATCHES}")
    print(f"  Batch size:           {BATCH_SIZE}")
    print(f"  Total samples:        {BATCH_SIZE * NUM_BATCHES}")
    print(f"")
    print(f"  Novelty top_k:        {TOP_K}")
    print(f"  Novelty max_nodes:    {MAX_SEARCH_NODES}")
    print(f"")
    print(f"  _sample_categorical:  Gumbel-max")
    print(f"  Noise schedule:       LogLinear (eps=1e-3)")

    print("\n" + "=" * 60)
    print("4. Running sampling without CDD (baseline)...")
    print("=" * 60)

    samples_baseline = []
    start_time = time.time()
    for batch_idx in range(NUM_BATCHES):
        xt = sample_udlm_qm9_novelty(
            model, tokenizer, device, novelty_projector=None,
            num_steps=SAMPLING_STEPS, batch_size=BATCH_SIZE,
        )
        smi = tokenizer.decode(xt[0].cpu().tolist())
        samples_baseline.append(smi)
        smi_clean = _clean(smi)
        mol = rdChem.MolFromSmiles(smi_clean)
        if mol is not None:
            is_novel = smi_clean not in dataset_smiles
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:Y | Novel:{'Y' if is_novel else 'N'}")
        else:
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:N")

    time_baseline = time.time() - start_time
    results_baseline = evaluate_samples_novelty(samples_baseline, dataset_smiles)

    print(f"\n   Baseline Results:")
    print(f"   Valid: {results_baseline['valid_pct']*100:.1f}%")
    print(f"   Valid & Novel: {results_baseline['valid_novel_count']}")
    print(f"   Violation (valid but not novel): {results_baseline['violation_pct']*100:.2f}%")
    if results_baseline['qeds_novel']:
        print(f"   QED Mean (novel only): {results_baseline['qed_mean_novel']:.3f}")

    print("\n" + "=" * 60)
    print("5. Running sampling with CDD (novelty projection)...")
    print("=" * 60)

    novelty_projector = NoveltyProjector(
        seen_smiles=dataset_smiles, tokenizer=tokenizer,
        top_k=TOP_K, max_nodes=MAX_SEARCH_NODES,
    )

    samples_cdd = []
    start_time = time.time()
    for batch_idx in range(NUM_BATCHES):
        xt = sample_udlm_qm9_novelty(
            model, tokenizer, device, novelty_projector=novelty_projector,
            num_steps=SAMPLING_STEPS, batch_size=BATCH_SIZE,
        )
        smi = tokenizer.decode(xt[0].cpu().tolist())
        samples_cdd.append(smi)
        smi_clean = _clean(smi)
        mol = rdChem.MolFromSmiles(smi_clean)
        if mol is not None:
            is_novel = smi_clean not in dataset_smiles
            novelty_projector.mark_seen(smi_clean)
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:Y | Novel:{'Y' if is_novel else 'N'}")
        else:
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES} done | Valid:N")

    time_cdd = time.time() - start_time
    results_cdd = evaluate_samples_novelty(samples_cdd, dataset_smiles)

    print(f"\n   CDD Results:")
    print(f"   Valid: {results_cdd['valid_pct']*100:.1f}%")
    print(f"   Valid & Novel: {results_cdd['valid_novel_count']}")
    print(f"   Violation (valid but not novel): {results_cdd['violation_pct']*100:.2f}%")
    if results_cdd['qeds_novel']:
        print(f"   QED Mean (novel only): {results_cdd['qed_mean_novel']:.3f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"{'Metric':<25} {'Baseline':<15} {'CDD':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Valid %':<25} {results_baseline['valid_pct']*100:.1f}{'':>10} {results_cdd['valid_pct']*100:.1f}{'':>10} {(results_cdd['valid_pct']-results_baseline['valid_pct'])*100:+.1f}")
    print(f"{'Valid & Novel (count)':<25} {results_baseline['valid_novel_count']:<15} {results_cdd['valid_novel_count']:<15} {results_cdd['valid_novel_count']-results_baseline['valid_novel_count']:+d}")
    print(f"{'QED Mean (novel)':<25} {results_baseline['qed_mean_novel']:.3f}{'':>12} {results_cdd['qed_mean_novel']:.3f}{'':>12} {results_cdd['qed_mean_novel']-results_baseline['qed_mean_novel']:+.3f}")
    print(f"{'Violation %':<25} {results_baseline['violation_pct']*100:.2f}{'':>10} {results_cdd['violation_pct']*100:.2f}{'':>10} {(results_cdd['violation_pct']-results_baseline['violation_pct'])*100:+.2f}")
    print(f"{'Time (s)':<25} {time_baseline:.2f}{'':>13} {time_cdd:.2f}{'':>13}")
    print("-" * 70)

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'cdd_novelty_eval_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'baseline': {
                    'valid_pct': results_baseline['valid_pct'],
                    'valid_novel_count': results_baseline['valid_novel_count'],
                    'valid_novel_pct': results_baseline['valid_novel_pct'],
                    'qed_mean_novel': results_baseline['qed_mean_novel'],
                    'violation_pct': results_baseline['violation_pct'],
                    'time': time_baseline,
                    'num_samples': len(samples_baseline),
                },
                'cdd': {
                    'valid_pct': results_cdd['valid_pct'],
                    'valid_novel_count': results_cdd['valid_novel_count'],
                    'valid_novel_pct': results_cdd['valid_novel_pct'],
                    'qed_mean_novel': results_cdd['qed_mean_novel'],
                    'violation_pct': results_cdd['violation_pct'],
                    'time': time_cdd,
                    'num_samples': len(samples_cdd),
                },
            },
            'baseline_samples': samples_baseline,
            'cdd_samples': samples_cdd,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
