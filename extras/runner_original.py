from Agent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from noise import *

class SimpleRunner:
    def __init__(self, num_episodes=1000, learn_every=1, noise_variance=0.1, viz=False, test_every=100, wandb_on=True) -> None:
        self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='rgb_array')
        self.num_episodes = num_episodes
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.learn_every = learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.test_every = test_every
        self.wandb_on = wandb_on
        self.avg_window = self.num_episodes//20
        self.best_model_episode = 671
        self.target_observation = np.array([1.0, 0.0, 0.0])
        self.K = 1
        self.e_next = None
        self.e = None

        if self.wandb_on:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "test_every": test_every,
            }
            wandb.init(project="TD3-control-new", entity="shivanichepuri", config = wandb.config)   
    
    def run(self):
        steps=0
        # actor_noise = OUNoise(mu=np.zeros(self.env.action_space.shape[0]))
        score_history = [] # score is the sum of rewards of an episode
        best_score = self.env.reward_range[0] 
        for episode in range(self.num_episodes):
            observation, _ = self.env.reset()
            done = False
            score = 0
            episode_len = 0
            total_action = 0
            score_rewards = []
            while not done:
                actor_noise = np.random.normal(0, self.noise_variance, self.env.action_space.shape)    
                action = self.agent.act(observation) + actor_noise
                observation_, reward, done, done_, _ = self.env.step(action)

                done = done or done_
                self.agent.memorize(observation, action, reward, observation_, not done)
                observation = observation_
                if steps % self.learn_every == 0:
                     err_actor, err_critic = self.agent.learn()
                steps = (steps + 1) % self.learn_every
                if self.viz:
                    self.env.render()
                score += reward
                episode_len+=1
                total_action += float(action)
                score_rewards.append(reward)

            score_history.append(score)
            avg_score = np.mean(score_history[-self.avg_window:])
            if avg_score > best_score:
                best_score = avg_score
                best_score_rewards = score_rewards
                self.best_model_episode = episode
                self.agent.TD.save(f"checkpoints/{episode}")

            # print(f"train_episode: {episode}, train_score: {score}, train_episode_len: {episode_len}, train_actor_loss:{err_actor}, train_critic_loss:{err_critic}, train_avg_score:{avg_score}, train_episode_total_action:{total_action}")
            if self.wandb_on:
                wandb.log({"train_episode":episode, "train_score":score, "train_episode_len":episode_len, "train_actor_loss": err_actor, "train_critic_loss":err_critic, "train_avg_score":avg_score, "train_episode_total_action":total_action})

            # if episode % self.learn_every == 0:
                # self.agent.TD.save(f"checkpoints/{episode}")
            
            if episode % self.test_every == 0:
                self.test(episode)
        if self.wandb_on:
            # table = wandb.Table(columns=best_score_rewards)
            #"train_best_score_rewards":table,
            wandb.log({"train_best_score":best_score, "train_best_model_episode":self.best_model_episode})

        # self.test_after_training(self.best_model_episode)

    def test(self, episode):
        observation, _ = self.env.reset()
        done = False
        score = 0
        while not done:
            action = self.agent.act(observation)
            observation_, reward, done, done_, _ = self.env.step(action)
            done = done or done_
            observation = observation_
            if self.viz:
                self.env.render()
            self.env.render()
            score += reward
        print(f"eval_episode: {episode}, eval_reward: {score}")
        if self.wandb_on:
            wandb.log({"eval_episode":episode, "eval_reward": score})

    def test_after_training(self): # we test for one episode
        self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='human')
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        
        # self.best_model_episode = 804
        self.agent.TD.load(f"checkpoints/{self.best_model_episode}") 
        observation, _ = self.env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = self.agent.act(observation)
            observation_, reward, done, done_, _ = self.env.step(action)
            done = done or done_
            observation = observation_
            if self.viz:
                self.env.render()
            self.env.render()
            score += reward
            steps += 1  
            if self.wandb_on:
                wandb.log({"final_test_rewards": reward, "final_test_observations_theta": np.arccos(observation[0]), "final_test_observations_dtheta": observation[2], "final_test_actions":action})

        print(f"final_test_score: {score}, final_test_episode_len: {steps}")
        if self.wandb_on:
            wandb.log({"final_test_score": score, "final_test_episode_len": steps})
    
if __name__ == "__main__":
    runner = SimpleRunner(viz = False)
    # runner.run()
    runner.test_after_training()