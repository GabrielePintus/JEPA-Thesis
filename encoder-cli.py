import torch
from lightning.pytorch.cli import LightningCLI
from src.encoder_temporal import VICRegJEPAEncoder
# from src.data.datamodule import PointMazeVICRegDataModule
from src.data.datamodule import PointMazeDataModule as DataModule

SEED = 0


if __name__ == "__main__":
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Lightning CLI
    LightningCLI(
        model_class=VICRegJEPAEncoder,
        datamodule_class=DataModule,
        seed_everything_default=SEED,
        save_config_kwargs={"overwrite": True},
    )
