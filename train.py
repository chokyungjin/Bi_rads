import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

# print(hydra.__version__)


# @hydra.main(config_path="configs", config_name="train", version_base="1.1")
@hydra.main(config_path="configs", config_name="train")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # print(os.getcwd())
    # print(f"Orig working directory    : {get_original_cwd()}")
    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
