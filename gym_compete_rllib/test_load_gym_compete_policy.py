import os
import re

from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets


def load_one(env_name, agent_id):
    """Load the policy for one agent."""
    nets = get_policy_value_nets(env_name, agent_id, raise_on_weight_load_failure=True)
    assert isinstance(nets, dict), "Can't load %s %d" % (env_name, agent_id)


def test_load():
    """Test that we can load policies for all agents."""
    zoo_path = '../multiagent-competition/gym_compete/agent_zoo'
    envs = os.listdir(zoo_path)

    # environments with fully-connected networks
    envs_fcnet = ['YouShallNotPassHumans-v0', 'RunToGoalHumans-v0']

    envs = set(envs).intersection(envs_fcnet)

    results = []
    total_calls = 0
    errors = 0


    for e in envs:
        for f in os.listdir(zoo_path + '/' + e):
            if not f.endswith('pkl'): continue
            if not f.startswith('agent'): continue
            agent, _, v = re.split('_|-', f)
            agent = agent[5:]
            v = v[1:-4]
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


if __name__ == '__main__':
    test_load()
