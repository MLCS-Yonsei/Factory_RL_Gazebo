from setuptools import setup
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RL_mlcs'))

setup(name='RL_mlcs',
      version='0.0.2',
      install_requires=['gym>=0.2.3'],
      description='reinforcement learning agents using Gazebo and ROS.',
      url='https://github.com/erlerobot/gym',
      author='Seungchul ha',
      package_data={'RL_mlcs': ['launch/*.launch', 'worlds/*']},
)