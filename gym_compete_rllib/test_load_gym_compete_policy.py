import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import re
import pytest

from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets, difference_new_networks, pickle_path


def get_envs():
    envs = os.listdir(pickle_path)

    # environments with fully-connected networks
    envs_fcnet = ['YouShallNotPassHumans-v0']#, 'RunToGoalHumans-v0']

    envs = set(envs).intersection(envs_fcnet)
    return envs

@pytest.fixture
def envs():
    """List of supported environments (FCnet)."""
    return get_envs()

def agents_and_versions(env_name):
    """List of agents and their versions."""
    for f in os.listdir(os.path.join(pickle_path, env_name)):
        if not f.endswith('pkl'): continue
        if not f.startswith('agent'): continue
        agent, _, v = re.split('_|-', f)
        agent = agent[5:]
        v = v[1:-4]
        yield (agent, v, f)

def load_one(env_name, agent_id):
    """Load the policy for one agent."""
    nets = get_policy_value_nets(env_name, agent_id, raise_on_weight_load_failure=True)
    assert isinstance(nets, dict), "Can't load %s %d" % (env_name, agent_id)
    return nets


def test_load(envs):
    """Test that we can load policies for all agents."""
    results = []
    total_calls = 0
    errors = 0

    for e in envs:
        for (agent, v, f) in agents_and_versions(e):
            print("Agent [%s] version [%s]" % (agent, v))
            try:
                load_one('multicomp/' + e, agent)
                results.append((e, agent, v, "OK"))
            except Exception as exc:
                results.append((e, agent, v, exc))
                errors += 1
                print(e, agent, v, exc)
            total_calls += 1

    print("Total errors: %d out of %d" % (errors, total_calls))
    for (e, agent, v, exc) in results:
        print(e, agent, v, exc)
    assert not errors, "There were errors"

def test_prediction(envs, tolerance_percent=1):
    """Test that new keras network match the output of old networks with weights from .pkl files."""
    errors = []

    for env in envs:
        for (agent, v, f) in agents_and_versions(env):
            print(env, agent, v, f)
            env_full = 'multicomp/' + env
            nets = load_one(env_full, agent)
            value = nets['value']
            policy_mean_std = nets['policy_mean_logstd']
            delta = difference_new_networks(env_full, str(int(agent) - 1), value, policy_mean_std, eps=1e-10, n_test_obs=10000, verbose=False)
            print(delta)
            for k, val in delta.items():
                if val['max'] > tolerance_percent:
                    errors.append(f"Error for {env}, {agent} v{v} {f} {k} is too high: {val}")

    for e in errors:
        print(e)

    assert not errors, "There were errors."

if __name__ == '__main__':
    test_prediction(get_envs())