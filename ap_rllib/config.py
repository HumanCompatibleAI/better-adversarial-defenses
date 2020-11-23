# load config helpers
from ap_rllib.config_helpers import get_trainer, get_config_by_name, select_config, get_config_names
from ap_rllib.config_helpers import build_trainer_config, get_agent_config, get_config_attributes

# adding actual configuration to the CONFIGS dict
from ap_rllib.config_extra import *
from ap_rllib.config_frankenstein import *
from ap_rllib.config_performance import *
from ap_rllib.config_rllib import *