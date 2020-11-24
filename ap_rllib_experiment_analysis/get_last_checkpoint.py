"""Get last checkpoint for a given run."""

import argparse
from ap_rllib_experiment_analysis.analysis_helpers import get_last_checkpoint
from ap_rllib.config import get_config_names

# parser for main()
parser = argparse.ArgumentParser(description='Get last checkpoint for a config')
parser.add_argument('--config', type=str, help='Config to look for', default=None, required=True, choices=get_config_names())

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint = get_last_checkpoint(args.config)
    print(checkpoint)
