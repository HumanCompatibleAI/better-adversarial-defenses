from rock_paper_scissors.rock_paper_scissors import rewards, print_outcomes, n_act, action_to_descr
import numpy as np


def test_print_outcomes():
    print_outcomes()


def test_rps_rewards_zero_sum():
    for a1 in range(n_act):
        for a2 in range(n_act):
            r = rewards(a1, a2)
            assert np.sum(r) == 0


def test_rps_rewards_sameact_zeroreward():
    for a in range(n_act):
        r = rewards(a, a)
        assert np.allclose(r, np.zeros(2))


def test_rps_rewards_hardcodedresult():
    # reward for first opponent
    rews_reference = {'RP': -1,
                      'PS': -1,
                      'SR': -1}
    for a1a2, r in rews_reference.items():
        def test_a1a2(a1a2, reward):
            a1, a2 = a1a2
            a1 = action_to_descr.index(a1)
            a2 = action_to_descr.index(a2)
            r = rewards(a1, a2)
            assert r[0] == reward
        test_a1a2(a1a2, r)
        test_a1a2(a1a2[::-1], -r)
