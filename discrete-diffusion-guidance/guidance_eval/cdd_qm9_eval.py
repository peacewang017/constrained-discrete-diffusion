"""CDD Evaluation using Hydra config."""

import hydra
from omegaconf import DictConfig, OmegaConf
import typing
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cdd.alm_projection import project_to_constraint
from cdd.delta_g import make_delta_g_fn


@hydra.main(version_base=None, config_path="../configs", config_name="cdd_qm9_eval")
def main(cfg: DictConfig) -> None:
    print("CDD Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup delta_g_fn with sascore checkpoint
    delta_g_fn = make_delta_g_fn(cfg.sascore_ckpt, cfg.tau)

    # Run evaluation with CDD
    print(f"\nRunning evaluation with tau={cfg.tau}, device={device}")
    # ... (evaluation code)


if __name__ == '__main__':
    main()