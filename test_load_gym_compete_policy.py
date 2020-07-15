from load_gym_compete_policy import get_policy_value_nets, difference_new_networks
import os, re

def load_one(env_name, agent_id):
    nets = get_policy_value_nets(env_name, agent_id)
    assert isinstance(nets, dict), "Can't load %s %d" % (env_name, agent_id)

def test_load():
    zoo_path = 'multiagent-competition/gym_compete/agent_zoo'
    envs = os.listdir(zoo_path)
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

if __name__ == '__main__':
    test_load()
