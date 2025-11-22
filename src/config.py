import yaml

def load_config(config_path, stage="stage1"):
    '''
    Load merged config for given split and stage.

    Args:
        config_path (str): YAML path. 
        stage (str): "stage1" or "stage2".

    Returns:
        dict: Merged config (common + split/stage).
    '''
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    common = cfg.get("common", {})
    stage_cfg = cfg[stage]
    return {**common, **stage_cfg}