from Agent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

class SimpleRunner:
    def __init__(self, num_episodes=10000, learn_every=50, noise_variance=0.1, viz=False, test_every=500) -> None:
        self.env = gym.make('Pendulum-v1', new_step_api=True)
        self.num_episodes = num_episodes
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.learn_every = learn_every
        self.learn_every=learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.test_every = test_every
        self.target_state = np.array([1.0, 0.0])
        self.K = 1


        wandb.config = {
            "num_episodes": num_episodes,
            "learn_every": learn_every,
            "noise_variance": noise_variance,
            "test_every": test_every,
        }
        wandb.init(project="Selfplay-RL", entity="gunda-boys-bad-elements", config = wandb.config)   
    
    def run(self):
        steps=0
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            self.e_prev = self.get_reward_state(state) - self.target_state
            while not done:
                action = self.agent.act(state) + np.random.normal(0, self.noise_variance, self.env.action_space.shape)
                next_state, reward_env, done, done_, _ = self.env.step(action)

                reward = self.get_reward(next_state)

                done = done or done_
                self.agent.memorize(state, action, reward, next_state, not done)
                state = next_state
                if steps % self.learn_every == 0:
                    self.agent.learn()
                steps = (steps + 1) % self.learn_every
                if self.viz:
                    self.env.render()
                episode_reward += reward

            print(f"Episode: {episode}, Reward: {episode_reward}")
            wandb.log({"Episode":episode, "Reward": episode_reward, "Env Reward": reward_env})
            if episode % 1000 == 0:
                self.agent.TD.save(f"checkpoints/{episode}")
            
            if episode % self.test_every == 0:
                self.test(episode)

        self.env.close()
    
    def test(self, episode):
        state = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = self.agent.act(state)
            next_state, reward, done, done_, _ = self.env.step(action)
            done = done or done_
            state = next_state
            if self.viz:
                self.env.render()
            self.env.render()
            episode_reward += reward
        print(f"Test Episode: {episode}, Reward: {episode_reward}")
        wandb.log({"Test Episode":episode, "Test Reward": episode_reward})
    
    def get_reward(self, state):
        state = self.get_reward_state(state)
        e = state - self.target_state
        e_dot = (e - self.e_prev)/self.env.dt
        reward = -(np.linalg.norm(e_dot + self.K*e)**2)
        self.e_prev = e
        return reward
    
    def get_reward_state(self, state):
        return state[:2]



if __name__ == "__main__":
    runner = SimpleRunner(viz = False)
    runner.run()