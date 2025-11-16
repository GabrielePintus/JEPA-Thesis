import torch
from lightning.pytorch.cli import LightningCLI
# from src.encoder_seq import VICRegJEPAEncoder
from src.data.datamodule import PointMazeSequencesDataModule
from src.encoder_pldm import PLDMEncoder

SEED = 0


if __name__ == "__main__":
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Lightning CLI
    LightningCLI(
        model_class=PLDMEncoder,
        datamodule_class=PointMazeSequencesDataModule,
        seed_everything_default=SEED,
        save_config_kwargs={"overwrite": True},
    )
