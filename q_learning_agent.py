import numpy as np
import pandas as pd
import typing as tt
from dataclasses import asdict
from collections import defaultdict
from microgrid_env import MicrogridEnv

State = int
Action = int
ValuesKey = tt.Tuple[State, Action]

class QLearningAgent:
    
    def __init__(self, env: MicrogridEnv, discounting_factor: float, learning_rate: float, epsilon_start: float, epsilon_end: float, decay_steps: int):
        self.env = env
        self.step = 0
        self.discounting_factor = discounting_factor
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.action_space = {0: 'charge-1500', 1: 'charge-1000', 2: 'charge-500', 3: 'do-nothing', 4: 'discharge-500', 5: 'discharge-1000', 6: 'discharge-1500'}
        self.state, self.reward, self.done = self.env.reset()
        self.values: tt.Dict[ValuesKey] = defaultdict(float)

    
    def anneal_epsilon(self):
        """
        Gradually reduce epsilon over time.
        """
        
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.step / self.decay_steps) * (self.epsilon_start - self.epsilon_end)
        )

        self.step += 1

    
    def sample_env(self, use_fixed_rule_policy: bool) -> tt.Tuple[State, Action, float, State]:

        # Select an action with e-greedy exploration / exploitation strategy

        self.anneal_epsilon()

        if use_fixed_rule_policy:
            
            action = self.env.rule_based_policy()

        else:
        
            if np.random.random() > self.epsilon:
    
                # Select best action from the q-value function for the current state
                _, action = self.best_value_and_action(state=self.state.index)
    
            else: 
    
                # Sample a random action from the action space
                action = self.env.sample_action()

        old_state = self.state

        # Execute the action against the environment
        new_state, reward, done = self.env.step(action=action)

        # Check for terminal state
        if done:

            # Reset the environment
            self.state, self.reward, self.done = self.env.reset()

        else:

            # Progress to the next state
            self.state = new_state

        return old_state, action, float(reward), new_state # -> s, a, r, s`
        

    def best_value_and_action(self, state: State) -> tt.Tuple[float, Action]:

        best_value, best_action = None, None

        for action in range( self.env.get_number_of_actions() ):

            action_value = self.values[(state, action)]

            if (best_value is None) or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_value, best_action


    def value_update(self, state: State, action: Action, reward: float, next_state: State):

        # Get the q-value for the current state
        old_val = self.values[(state, action)]

        # Get the highest/best q-value for the next state 
        best_val, _ = self.best_value_and_action(next_state)

        new_val = reward + self.discounting_factor * best_val

        # Bellman q-value function update
        self.values[(state, action)] = (1 - self.learning_rate) * old_val + self.learning_rate * new_val

        # self.values[(state, action)] = old_val + self.learning_rate * (new_val - old_val)

    
    def run_test_episode(self, env: MicrogridEnv):

        total_episode_reward = 0.0
        experience_history = []

        state, reward, done = env.reset()

        while True:

            _, action = self.best_value_and_action(state.index)

            new_state, reward, done = env.step(action=action)

            experience_history.append( (state, action, reward) )

            total_episode_reward += reward

            if done:
                
                break
                
            else:

                state = new_state

        return total_episode_reward, experience_history


    def display_action_value_table(self):

        nr_states = 168
        nr_actions = 7
        action_space = {0: 'charge-1500', 1: 'charge-1000', 2: 'charge-500', 3: 'do-nothing', 4: 'discharge-500', 5: 'discharge-1000', 6: 'discharge-1500'}

        print("\nAction Value Table: ")

        for s in range(nr_states):

            charge1500    = self.values[(s, 0)]
            charge1000    = self.values[(s, 1)]
            charge500     = self.values[(s, 2)]
            do_nothing    = self.values[(s, 3)]
            discharge500  = self.values[(s, 4)]
            discharge1000 = self.values[(s, 5)]
            discharge1500 = self.values[(s, 6)]

            best_value, best_action = self.best_value_and_action(state=s)
            best_action_name = self.action_space[best_action]
            
            print(f"""
            State: {s}, 
            Action Values -> 
               [0] charge-1500: {charge1500: .2f} vs [6] discharge-1500: {discharge1500: .2f}
               [1] charge-1000: {charge1000: .2f} vs [5] discharge-1000: {discharge1000: .2f}
               [2] charge-500:  {charge500: .2f} vs [4] discharge-500:  {discharge500: .2f}
               [3] do-nothing:  {do_nothing: .2f}

               *[{best_action}] {best_action_name}: {best_value: .2f}
            """)

    
    def save_learned_policy(self, env: MicrogridEnv):

        total_episode_reward, experience_history = self.run_test_episode(env=env)

        experience_hist_with_extra_info = []

        for state, action, reward in experience_history:
            
            experience_hist_with_extra_info.append( asdict(state) )

        df = pd.DataFrame(data=experience_hist_with_extra_info)

        print(f"\nTotal Episode Reward: {total_episode_reward: .2f}")

        print("\nSave agent's learned policy.\n")

        df.to_csv('agent_learned_policy.csv', header=True, index=True)


            