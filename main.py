import hydra
from omegaconf import DictConfig
from peer_x.workflow import supervised


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    if config.get("is_training_mode"):
        supervised.train(config)
    else:
        supervised.test(config)


if __name__ == "__main__":
    main()
