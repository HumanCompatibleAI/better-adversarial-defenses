import rock_paper_scissors as rps
from rps_rl import RPSEnv, RPSAgent, Monitor, Universe


def test_create_env():
    env = RPSEnv(noise_dim=4)


def test_create_agent():
    env = RPSEnv(noise_dim=4)
    agent = RPSAgent(noise_dim=env.noise_dim, identity=0)


def test_create_monitor():
    env = RPSEnv(noise_dim=4)
    AGENTS = [RPSAgent(noise_dim=env.noise_dim, identity=i)
              for i in range(5)]
    m = Monitor(agents=AGENTS)
