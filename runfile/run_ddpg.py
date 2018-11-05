#!/usr/bin/env python
import gym
import RL_mlcs
import time
import numpy
import random
import time

from env_reset import env_reset
from module import ddpg, replay, liveplot

from ddpg_config import config

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    env = gym.make('ddpg-v0')
    env.action_space = 3
    
    # outdir = '/tmp/gazebo_gym_experiments'
    # env = gym.wrappers.Monitor(env, outdir, force=True)
    # plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    ddpg = ddpg.DDPG(config)
    if config.load_weight:
        try:
            ddpg.load(numpy.load('weights.npy').item())
            print ('weight loaded.')
        except:
            print ('weight does not exist')

    memory = replay.Replay(config.max_buffer, config.batch_size)
    if config.load_buffer:
        try:
            memory.buffer=numpy.load('buffer.npy').item()
            print ('replay memory loaded')
        except:
            print ('replay memory does not exist')

    initial_epsilon = ddpg.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    highest_reward = 0

    for x in range(int(config.max_episode)):
        done = False
        cumulated_reward = 0 #Should going forward give more reward then L/R ?
        state0 = env.reset()
        if ddpg.epsilon > 0.05:
            ddpg.epsilon *= epsilon_discount

        for i in range(int(config.max_step)):

            # Pick an action based on the current state
            action = ddpg.chooseAction(state0)

            # Execute the action and get feedback
            state1,reward,done,info = env.step(action)
            print("action: %s,  reset : %s" %(action,done))
            experience={
                'state_0':state0,
                'state_1':state1,
                'action':action,
                'reward':reward,
                'done':done
            }
            memory.add(experience)

            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            #nextState = ''.join(map(str, observation))

            if memory.buffersize > config.batch_size:
                batch=memory.batch()
                ddpg.learn(batch)
                batch.clear()

            # env._flush(force=True)

            if not(done):
                state0 = state1
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        if (x+1)%10==0:
            # plotter.plot(env)
            numpy.save('weights.npy',ddpg.return_variables())
            numpy.save('buffer.npy',memory.buffer)
        
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(int(config.max_episode))+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
