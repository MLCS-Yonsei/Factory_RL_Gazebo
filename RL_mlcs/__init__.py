import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='qlearn-v0',
    entry_point='RL_mlcs.envs.vehicle_v2:qlearnEnv',
)

register(
    id='ddpg-v0',
    entry_point='RL_mlcs.envs.vehicle_v2:ddpgEnv',
)