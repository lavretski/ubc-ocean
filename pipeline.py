import hydra
from omegaconf import DictConfig, OmegaConf
from simple_predictor import simple_predictor
from train.train import train


@hydra.main(config_path="configs", config_name="configs")
def main(cfg_: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg_, resolve=True)
    # simple_predictor(cfg["train_csv_file"], cfg["test_csv_file"], cfg["submission_csv_file"])
    train(cfg["train_data_dir"], cfg["test_data_dir"],
          cfg["train_csv_file"], cfg["test_csv_file"],
          cfg["image_size"], cfg["batch_size"],
          cfg["validation_split"], cfg["random_seed"],
          cfg["epochs"], cfg["lr"], cfg["num_classes"],
          cfg["save_model_path"], cfg["rescale_multiplier"])

if __name__ == "__main__":
    main()