def get_policies_all(config, n_policies, obs_space, act_space, policy_template="player_%d%s"):
    """Get a policy dictionary, both pretrained/from scratch."""
    which_arr = {"pretrained": "_pretrained", "from_scratch": ""}
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config,
                                                                 obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)
                for which_k, which_v in which_arr.items()
                }
    return policies


def get_policies_withnormal_sb(config, n_policies, obs_space, act_space, policy_template="player_%d%s"):
    """Get a policy dictionary, both pretrained normal and adversarial opponents."""
    which_arr = {1:
                     {"from_scratch_sb": "_pretrained_adversary_sb",
                      "pretrained": "_pretrained_sb",
                      },
                 2:
                     {"pretrained": "_pretrained_sb"}
                 }
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config,
                                                                 obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)
                for which_k, which_v in which_arr[i].items()
                }
    if config['_verbose']:
        print("Policies")
        print(policies.keys())
    return policies


def select_policy_opp_normal_and_adv_sb(agent_id, config, do_print=False):
    """Select policy at execution, normal-adversarial opponents."""
    p_normal = config['_p_normal']
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained_sb", "player_1_pretrained_adversary_sb"],
                               p=[p_normal, 1 - p_normal])
        if do_print or config['_verbose']:
            print('Chosen', out)
        return out
    elif agent_id == "player_2":
        # pretrained victim
        return "player_2_pretrained_sb"


def get_policies_pbt(config, n_policies, obs_space, act_space, policy_template="player_%d%s",
                     from_scratch_name="from_scratch"):
    """Get a policy dictionary, population-based training."""
    n_adversaries = config['_n_adversaries']
    which_arr = {1:
                     {"pretrained": ["_pretrained"],
                      from_scratch_name: ["_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)]},
                 2: {"pretrained": ["_pretrained"]}
                 }

    policies = {
        policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config, obs_space=obs_space,
                                                         act_space=act_space)
        for i in range(1, 1 + n_policies)
        for which_k, which_v_ in which_arr[i].items()
        for which_v in which_v_
    }
    return policies


def select_policy_opp_normal_and_adv_pbt(agent_id, config, do_print=False):
    """Select policy at execution, PBT."""
    p_normal = config['_p_normal']
    n_adversaries = config['_n_adversaries']

    if agent_id == "player_1":
        out = np.random.choice(
            ["player_1_pretrained"] + ["player_1_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)],
            p=[p_normal] + [(1 - p_normal) / n_adversaries for _ in range(n_adversaries)])
    elif agent_id == "player_2":
        # pretrained victim
        out = "player_2_pretrained"
    if do_print or config['_verbose']:
        print(f"Choosing {out} for {agent_id}")
    return out


def select_policy_opp_normal_and_adv(agent_id, config, do_print=False):
    """Select policy at execution, normal-adversarial opponents."""
    p_normal = config['_p_normal']
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained", "player_1"],
                               p=[p_normal, 1 - p_normal])
        if do_print or config['_verbose']:
            print('Chosen', out)
        return out
    elif agent_id == "player_2":
        # pretrained victim
        return "player_2_pretrained"