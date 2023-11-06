import pydoc
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="configs", version_base=None)
def main(cfg_: DictConfig):
    cfg = OmegaConf.to_container(cfg_, resolve=True)

    tasks = cfg["tasks"]
    tasks_descr = cfg["tasks_descr"]

    for task in tasks:
        print(task)
        args = tasks_descr[task]
        script_id = args.pop("script_id")
        for arg_key in args:
            if not isinstance(args[arg_key], dict):
                continue
            if "type" in args[arg_key]:
                model_cfg = args[arg_key]
                model = pydoc.locate(model_cfg.pop("type"))(**model_cfg)
                args[arg_key] = model

        runner = pydoc.locate(script_id)
        if callable(runner):
            runner(**args)
        else:
            runner.main(**args)


if __name__ == '__main__':
    main()