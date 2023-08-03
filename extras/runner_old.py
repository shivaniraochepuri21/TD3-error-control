from Agent import Agent
# import gymnasium as gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from gym.wrappers import RescaleAction
import time

class SimpleRunner:
    def __init__(self, num_episodes=10, test_every=50, learn_every=1, noise_variance=0.1, wandb_on=0, K=1, test_atlast=1, viz=False) -> None:
        # self.env = gym.make('Pendulum-v1')
        self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='rgb_array')

        self.num_episodes = num_episodes
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.learn_every = learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.test_every = test_every
        # self.target_state = np.array([np.cos(0.0 * np.pi/180), np.sin(0.0 * np.pi/180), 0.0])
        # self.test_target_state = np.array([np.cos(170.0 * np.pi/180), np.sin(170.0 * np.pi/180), 0.0])
        # self.test_init_state = np.array([np.cos(180.0 * np.pi/180), np.sin(180.0 * np.pi/180), 0.0])

        self.target_state = np.array([0.0 * np.pi/180, 0.0])
        self.test_target_state = np.array([170.0 * np.pi/180,0.0])
        self.test_init_state = np.array([180.0 * np.pi/180, 0.0])

        self.K = K
        self.dt = 0.05 #hardcoded
        self.w1 = 1
        self.w2 = 0.0
        self.w3 = 0.0

        self.test_atlast = test_atlast
        self.wandb_on = wandb_on

        if wandb_on == 1:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "test_every": test_every,
            }
            wandb.init(project="TD3-control ", entity="shivanichepuri", config = wandb.config)   
    
    def run(self):
        steps=0
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            # print("init obs", obs, "\n")
            done = False
            episode_len = 0
            episode_reward = 0
            episode_error = 0
            ep_control_effort = 0
            self.e = self.get_reward_state(obs) - self.get_reward_state(self.target_state)
            # self.e[0] = self.angle_normalize(self.e[0])
            self.e_prev = self.e
            self.e_dot = 0

            while not done:
                action = self.agent.act(obs) + np.random.normal(0, self.noise_variance, self.env.action_space.shape)
                next_obs, reward_env, done, done_, _ = self.env.step(action)
                # print("training obs", next_obs, "\n")

                done = done or done_
                reward, step_e_train, ode = self.get_reward(next_obs, self.target_state, action)
                self.agent.memorize(obs, action, reward, next_obs, done)
                obs = next_obs
                if steps % self.learn_every == 0:
                    err_actor, err_critic = self.agent.learn()
                steps = (steps + 1) % self.learn_every
                episode_reward += reward
                episode_error += np.linalg.norm(step_e_train)
                episode_len += 1
                ep_control_effort += action

            # print(f"TrainEpisode: {episode}, TrainReward: {episode_reward}, TrainEpisodeError: {episode_error}, TrainEpisodeLength: {episode_len}")
            if self.wandb_on == 1:
                wandb.log({"TrainEpisodes":episode, "TrainEpRewards": episode_reward, "TrainEpErrors": episode_error, "TrainEpLengths": episode_len, "TrainActorLosses": err_actor, "TrainCriticLosses": err_critic, "TrainEp_control_effort": ep_control_effort })

            if episode % 1000 == 0:
                self.agent.TD.save(f"checkpoints/{episode}")
            
            if episode % self.test_every == 0:
                self.test(episode, 0)

        # test after total training is over        
        if(self.test_atlast == 1):
            self.test(episode, self.test_atlast)

        if self.wandb_on == 1:
            wandb.finish()
    
    def get_reward(self, observation, target_state, action):
        observation = self.get_reward_state(observation)
        target_state = self.get_reward_state(target_state)
        self.e = observation - target_state
        # self.e[0] = self.angle_normalize(self.e[0])
        self.e_dot = (self.e - self.e_prev)/self.dt

        ode_part = np.linalg.norm(self.e_dot + self.K*self.e)**2
        err_part = np.linalg.norm(self.e)**2
        action_part = action**2
        reward = -(self.w1*(ode_part) + self.w2*(err_part) + self.w3*(action_part))
        
        self.e_prev = self.e
        return reward, self.e, ode_part
    
    def get_reward_state(self, obs):
        # return np.array([float(np.arctan2(obs[1],obs[0])), float(obs[2])])
        return obs
    
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def test(self, episode, flag=0):
        # options = 
        if flag==1:
            # self.env = gym.make('Pendulum-v1', render_mode="human")
            self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='human')

        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_error = 0
        episode_len = 0 #steps
        ep_control_effort = 0
        
        while not done:
            action = self.agent.act(obs)
            next_obs, reward_env, done, done_, _ = self.env.step(action)
            done = done or done_
            
            if(self.test_atlast and flag):
                target_state = self.target_state
            else:
                target_state = self.target_state

            reward, step_e, ode = self.get_reward(next_obs, target_state, action)

            st = self.get_reward_state(obs)
            obs = next_obs

            if self.viz and flag:
                self.env.render()
            
            episode_reward += reward
            episode_error += np.linalg.norm(step_e)
            episode_len += 1
            ep_control_effort += action

            if self.test_atlast == 1 and flag==1 and self.wandb_on == 1:
                wandb.log({"EndTest StepErrors_0": step_e[0], "EndTest StepErrors_1": step_e[1], "EndTest StepRewards": reward, "EndTest StepStates_0": st[0], "EndTest StepStates_1": st[1], "EndTest EpSteps": episode_len, "EndTest Reward_OdePart": ode, "EndTest Actions": action})

        if self.test_atlast == 1 and flag==1:
            # print(f"EndTest EpReward: {episode_reward}, EndTest EpError: {episode_error}, EndTest EpLength: {episode_len}")
            if self.wandb_on == 1:
                wandb.log({"EndTest EpReward": episode_reward, "EndTest EpError": episode_error, "EndTest EpLength": episode_len, "EndTest_control_effort": ep_control_effort})

        else:    
            # print(f"Test Episode: {episode}, Reward: {episode_reward}, EpisodeError: {episode_error}, EpisodeLength: {episode_len}")
            if self.wandb_on == 1:
                wandb.log({"TestEpisodes":episode, "TestEpRewards": episode_reward, "TestEpErrors": episode_error, "TestEpLengths": episode_len})
    
if __name__ == "__main__":
    runner = SimpleRunner(viz = True, wandb_on=0, test_atlast=1, K=1, num_episodes=1000, test_every=50, learn_every=1)
    runner.run()