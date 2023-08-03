import gym
import numpy as np
import time

# env = gym.make('Pendulum-v1', render_mode="human")
env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='human')
print(env.action_space.high)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


# for i in range(10):
#     observation, info = env.reset(init_state = [np.pi/4, -1])
#     th = np.arccos(observation[0])
#     print('theta', th)
#     angl = np.rad2deg(th)
#     angle_norm = np.rad2deg(angle_normalize(th))
#     print(angl, angle_norm)
#     print("\n")
#     time.sleep(1)

# for i in range(1):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
# if terminated or truncated:
#     observation, info = env.reset()
env.close()