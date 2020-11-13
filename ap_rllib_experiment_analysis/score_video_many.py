from ap_rllib_experiment_analysis.analysis_helpers import get_videos, get_scores
import argparse
import gin
import pandas as pd
import json
import ray


parser = argparse.ArgumentParser("Make videos / compute scores for many runs")
parser.add_argument('--gin_config', type=str, required=True)
parser.add_argument('--dataframe', type=str, required=True)
parser.add_argument('--score', action="store_true")
parser.add_argument('--video', action="store_true")

if __name__ == '__main__':
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config)
    df = pd.read_csv(args.dataframe)
    
    def write_out(data, name):
        out_file = args.dataframe + f"_{name}.json"
        print(f"Output filename [{name}]: {out_file}")
        json.dump(data, open(out_file, 'w'))
    
    if args.score:
        out = get_scores(df)
        write_out(out, "scores")
    if args.video:
        out = get_videos(df)
        write_out(out, "videos")