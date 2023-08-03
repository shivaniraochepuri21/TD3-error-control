from gym.envs.registration import register 
register(id='mypendulum-v0',entry_point='gym_mypendulum.envs:MyPendulumEnv', max_episode_steps=200) 
