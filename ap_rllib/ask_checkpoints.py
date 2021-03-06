import os

from dialog import Dialog
from tqdm import tqdm

from ap_rllib_experiment_analysis.analysis_helpers import get_df_from_logdir

DEFAULT_PATH = os.path.join(os.path.expanduser('~'), 'ray_results')


def get_checkpoint_list(path=DEFAULT_PATH, ask_path=True):
    """Get a list of checkpoints in a directory (with manual selection using Dialog)."""
    d = Dialog()
    if ask_path:
        code, path = d.inputbox("Where to look for checkpoints?", init=path, width=120)
        assert code == 'ok', f"Invalid response: {code} {path}"
    items = os.listdir(path)

    print("Reading data...")
    items_with_descr = []
    last_checkpoint = {}
    for item in tqdm(sorted(items)):
        try:
            df = get_df_from_logdir(os.path.join(path, item), do_tqdm=False)
            if len(df) <= 1:
                continue
            descr = f"Training iterations: {len(df)}"
            last_checkpoint[item] = df.iloc[-1].checkpoint_rllib
            items_with_descr.append((item, descr, True))
        except:
            pass

    if not items_with_descr:
        raise ValueError(f"No tune runs in this directory: {path}")

    code, tags = d.checklist("Which checkpoints do you want to use?",
                             choices=items_with_descr,
                             width=200, height=20)

    checkpoints = [last_checkpoint[t] for t in tags]

    return checkpoints


if __name__ == '__main__':
    checkpoints = get_checkpoint_list()
    print("Got checkpoints:")
    for c in checkpoints:
        print(c)
