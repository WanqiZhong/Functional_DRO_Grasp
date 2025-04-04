import os
import sys
import warnings
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.module import ThreeDoFModule
from model.network import create_three_dof_network
from data_utils.CombineRetargetDataset import create_dataloader


@hydra.main(version_base="1.2", config_path="configs", config_name="postrain_3dof")
def main(cfg):
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    pl.seed_everything(cfg.seed)

    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        project=cfg.wandb.project
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=cfg.gpu,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.max_epochs
    )

    dataloader = create_dataloader(cfg.dataset, is_train=True)
    netwrok = create_three_dof_network(cfg.model, mode='train')
    model = ThreeDoFModule(
        cfg=cfg.training,
        network=netwrok
    )
    model.train()

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
