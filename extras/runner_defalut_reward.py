from Agent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from noise import *

class SimpleRunner:
    def __init__(self, num_episodes=1000, learn_every=1, noise_variance=0.1, viz=False, eval_every=200, wandb_on=True) -> None:
        self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='rgb_array')
        self.num_episodes = num_episodes
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.learn_every = learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.eval_every = eval_every
        self.wandb_on = wandb_on
        self.avg_window = self.num_episodes//20
        self.best_model_episode = 769
        self.target_observation = np.array([1.0, 0.0, 0.0])
        self.K = 1
        self.w1 = 1.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.e_next = None
        self.e = None

        if self.wandb_on:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "eval_every": eval_every,
                "K": self.K,
                "w1": self.w1,
                "w2": self.w2,
                "w3": self.w3,
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
            episode_action = 0
            episode_error = np.array([0.0, 0.0])
            err_actor_episode = 0
            err_critic_episode = 0
            # score_rewards = []
            while not done:
                actor_noise = np.random.normal(0, self.noise_variance, self.env.action_space.shape)    
                action = self.agent.act(observation) + actor_noise
                observation_, reward, done, done_, _ = self.env.step(action)
                # reward = self.get_reward(observation, observation_, action[0], 0)

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
                episode_action += float(action)
                episode_error += self.e
                # score_rewards.append(reward)
                err_actor_episode += err_actor
                err_critic_episode += err_critic

            score_history.append(score)
            avg_score = np.mean(score_history[-self.avg_window:])
            if avg_score > best_score:
                best_score = avg_score
                # best_score_rewards = score_rewards
                self.best_model_episode = episode
                self.agent.TD.save(f"checkpoints/{episode}")

            print(f"train_episodes: {episode}, train_scores:{score}")

            if self.wandb_on:
                wandb.log({"train_episodes":episode, "train_scores":score, "train_episodes_lens":episode_len, "train_episodes_actor_loss": err_actor_episode, "train_episodes_critic_loss":err_critic_episode, "train_episodes_avg_score":avg_score, "train_episodes_total_action":episode_action, "train_episode_error_mag":np.linalg.norm(episode_error)})

                # "train_episodes_error_theta":episode_error[0], "train_episode_error_dtheta":episode_error[1],

            # if episode % self.learn_every == 0:
                # self.agent.TD.save(f"checkpoints/{episode}")
            
            if episode % self.eval_every == 0:
                self.eval(episode)
        if self.wandb_on:
            # table = wandb.Table(columns=best_score_rewards)
            #"train_best_score_rewards":table,
            wandb.log({"train_best_score":best_score, "train_best_model_episode":self.best_model_episode})

        # self.test_after_training(self.best_model_episode)

    def eval(self, episode):
        observation, _ = self.env.reset()
        done = False
        score = 0
        episode_error = np.array([0.0, 0.0])
        step = 0
        while not done:
            action = self.agent.act(observation)
            observation_, reward, done, done_, _ = self.env.step(action)
            # reward = self.get_reward(observation, observation_, action[0], 1, episode)

            done = done or done_
            observation = observation_
            if self.viz:
                self.env.render()
            self.env.render()
            score += reward
            episode_error += self.e
            step+=1
            if self.wandb_on:
                wandb.log({"eval_observations_theta" + str(episode): np.arccos(observation[0]), "eval_observations_dtheta" + str(episode): observation[2], "eval_actions" + str(episode):action, "eval_steps" :step})

        print(f"eval_episode: {episode}, eval_score: {score}")
        if self.wandb_on:
            wandb.log({"eval_episodes":episode, "eval_scores": score, "eval_episode_errors_mag": np.linalg.norm(episode_error)})

    def test_after_training(self): # we test for one episode
        self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='human')
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        
        # self.best_model_episode = 804
        self.agent.TD.load(f"checkpoints/{self.best_model_episode}") 
        observation, _ = self.env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            action = self.agent.act(observation)
            observation_, reward, done, done_, _ = self.env.step(action)
            # reward = self.get_reward(observation, observation_, action[0], 2)

            done = done or done_
            observation = observation_
            if self.viz:
                self.env.render()
            self.env.render()
            score += reward
            step += 1  
            if self.wandb_on:
                wandb.log({"test_rewards": reward, "test_observations_theta": np.arccos(observation[0]), "test_observations_dtheta": observation[2], "test_actions":action, "test_steps":step})

        print(f"test_score: {score}, test_episode_len: {step}")
        if self.wandb_on:
            wandb.log({"test_score": score, "test_episode_len": step})
    
    # train_eval_test 0 training, 1 eval, 2 testing
    def get_reward(self, observation, observation_, action, train_eval_test=0, ep=-1):
        observation = self.get_reward_obs(observation) 
        observation_ = self.get_reward_obs(observation_) 
        target_observation = self.get_reward_obs(self.target_observation) 
        self.e = target_observation - observation
        self.e_next = target_observation - observation_ 
        e_dot = (self.e_next - self.e)/self.env.dt

        ode_part = (np.linalg.norm(e_dot + self.K*self.e))**2
        err_part = (np.linalg.norm(self.e))**2
        action_part = action**2
        reward = -self.w1*(ode_part) -self.w2*(err_part) -self.w3*(action_part)
        # reward = -((np.linalg.norm(e_dot + self.K*self.e))**2)
        err_mag = np.linalg.norm(self.e)

        if self.wandb_on:
            if train_eval_test == 0:
                wandb.log({"train_error_mag": err_mag})
            elif train_eval_test == 1:
                wandb.log({"eval_error_mag" + str(ep): err_mag})
            elif train_eval_test == 2:
                wandb.log({"test_error_mag": err_mag})
        return reward
    
    def get_reward_obs(self, obs):
        return obs[:2]

if __name__ == "__main__":
    runner = SimpleRunner(viz = False)
    runner.run()
    runner.test_after_training()