from ap_rllib.config import get_config_names, get_config_by_name, get_config_attributes

def test_configs(online=False):
    """Test that can create all configs, online enables online configs."""
    configs = get_config_names()
    assert len(configs) > 0, "No configs found."
    for name in configs:
        attrs = get_config_attributes(name)
        if attrs['online'] and online:
            config = get_config_by_name(name)
            assert isinstance(config, dict)
        if not attrs['online']:
            config = get_config_by_name(name)
            assert isinstance(config, dict)
            
        
if __name__ == "__main__":
    test_configs(online=True)
    print("Test passed")