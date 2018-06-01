import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='test-v0',
    entry_point='RL_mlcs.envs.turtlebot:testEnv',
)
