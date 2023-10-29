import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="configs", version_base='')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()