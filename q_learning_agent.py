import numpy as np
import typing as tt
from collections import defaultdict
from microgrid_env import MicrogridEnv

GAMMA = 0.99 # Discounting Factor
ALPHA = 0.3 # Learning Rate
EPSILON = 0.7 # e-greedy strategy / probability of taking a random action

State = int
Action = int
ValuesKey = tt.Tuple[State, Action]

class QLearningAgent:
    
    def __init__(self, env: MicrogridEnv):
        self.env = env
        self.state, self.reward, self.done = self.env.reset()
        self.values: tt.Dict[ValuesKey] = defaultdict(float)
        

    def sample_env(self, use_fixed_rule_policy: bool) -> tt.Tuple[State, Action, float, State]:

        # Select an action with e-greedy exploration / exploitation strategy
        # action = self.env.rule_based_policy()
        
        if np.random.random() > EPSILON:

            # Select best action from the q-value function for the current state
            _, action = self.best_value_and_action(state=self.state)

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

        new_val = reward + GAMMA * best_val

        # Update the q-value function (Bellman Update)
        # self.values[(state, action)] = old_val + ALPHA * (new_val - old_val)
        self.values[(state, action)] = (1 - ALPHA) * old_val + ALPHA * new_val

    
    def run_test_episode(self, env: MicrogridEnv) -> float:

        total_episode_reward = 0.0

        state, reward, done = env.reset()

        while True:

            _, action = self.best_value_and_action(state)

            new_state, reward, done = env.step(action=action)

            total_episode_reward += reward

            if done:
                
                break
                
            else:

                state = new_state

        return total_episode_reward



            