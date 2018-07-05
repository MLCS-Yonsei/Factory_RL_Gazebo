import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='qlearn-v0',
    entry_point='RL_mlcs.envs.youbot:qlearnEnv',
)

register(
    id='ddpg-v0',
    entry_point='RL_mlcs.envs.youbot:ddpgEnv',
)