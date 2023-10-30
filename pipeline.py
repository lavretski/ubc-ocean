import hydra
from omegaconf import DictConfig, OmegaConf
from simple_predictor import simple_predictor

@hydra.main(config_path="configs", config_name="configs")
def main(cfg_: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg_, resolve=True)
    simple_predictor(cfg["train_csv_file"], cfg["test_csv_file"], cfg["submission_csv_file"])

if __name__ == "__main__":
    main()