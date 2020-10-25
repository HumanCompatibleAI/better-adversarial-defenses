from ap_rllib.config import CONFIGS

def test_configs():
    for name, config in CONFIGS.items():
        assert isinstance(config, dict) or callable(config), f"Wrong config {name}"