import re
import traceback
import logging as log
import numpy as np
import pandas as pd
from dataclasses import dataclass

np.random.seed(2025)

@dataclass
class EnvState:
    ts_hour_sin: float	
    ts_hour_cos: float	
    tou_offpeak: int	
    tou_standard: int	
    tou_peak: int	
    day_week: int	
    day_saturday: int	
    day_sunday: int	
    site_load_energy: float 	
    solar_prod_energy: float
    solar_ctlr_setpoint: float
    grid_import_energy: float

    def get_solar_vs_load_ratio(self):

        return np.round((self.solar_prod_energy / self.site_load_energy) * 100, 2)
        

class MicrogridEnv:

  def __init__(self, data: pd.DataFrame):
      self.env_data = data
      self.state = None
      self.step_idx = 0
      self.reward = 0.0
      self.discount_factor = 0.99
      self.grid_maximum_notified_demand = 2000.0
      self.bess_capacity = 3000.0
      self.bess_avail_discharge_energy = self.bess_capacity
      self.bess_step_size = [1500.0, 1000.0, 500.0, 0.0, 500.0, 1000.0, 1500.0]
      self.action_space = {0: 'charge-1500', 1: 'charge-1000', 2: 'charge-500', 3: 'do-nothing', 4: 'discharge-500', 5: 'discharge-1000', 6: 'discharge-1500'} 
      self.done = False
      self.display_data_info()

  def display_data_info(self):

      print("Environment Defaults: ")
      print(f"""
      Grid Notified Maximum Demand: {self.grid_maximum_notified_demand} kVA
      BESS Capacity: {self.bess_capacity} kWh
      BESS Actions: {', '.join( list( self.action_space.values() ) )}
      Reward Discount Factor: {self.discount_factor}
      """)
      
      print("\nData Summary: ")
      print(self.env_data.info())
      print("\n")

  def get_new_state(self) -> EnvState:

      obs = self.env_data.iloc[self.step_idx, : ]

      self.state = EnvState(**obs)
      
      print(f"""
      [{self.step_idx}] Environment State ->
              hour_sin: {self.state.ts_hour_sin: .4f}
              hour_cos: {self.state.ts_hour_cos: .4f}
              tou_offpeak: {self.state.tou_offpeak: .0f}
              tou_standard: {self.state.tou_standard: .0f}
              tou_peak: {self.state.tou_peak: .0f}
              day_week: {self.state.day_week: .0f}
              day_saturday: {self.state.day_saturday: .0f}
              day_sunday: {self.state.day_sunday: .0f}
              site_load_energy: {self.state.site_load_energy: .2f} (kWh)
              solar_prod_energy: {self.state.solar_prod_energy: .2f} (kWh)
              solar_ctlr_setpoint: {self.state.solar_ctlr_setpoint: .2f} (%)
              solar_vs_load_ratio: {self.state.get_solar_vs_load_ratio(): .2f} (%)
              grid_import_energy: {self.state.grid_import_energy: .2f} (kWh)
              bess_avail_discharge_energy: {self.bess_avail_discharge_energy: .2f} (kWh)
              done: {self.done}
      """)
      
      return self.state

  def rule_based_policy(self, state: EnvState):

      # Action space indices
      charge_1500_idx = 0
      charge_1000_idx = 1
      charge_500_idx = 2
      do_nothing_idx = 3
      discharge_500_idx = 4
      discharge_1000_idx = 5
      discharge_1500_idx = 6

      action_idx = None

      if state.tou_peak == 1:
          # Discharge when it's TOU Peak
          action_idx = discharge_1500_idx
          
      elif (state.tou_standard == 1) and (state.get_solar_vs_load_ratio() >= 70.0) and (self.bess_avail_discharge_energy < self.bess_capacity):
          # Charge when it's TOU Standard and Solar PV produces more than 70% of the site load and the BESS available energy is less than it's full capacity
          action_idx = charge_500_idx
          
      elif (state.tou_offpeak == 1) and (self.bess_avail_discharge_energy < self.bess_capacity):
          # Charge when it's TOU Off-peak and the BESS available energy is less than it's full capacity
          action_idx = charge_1500_idx
          
      else:
          # Default action is to do nothing
          action_idx = do_nothing_idx
      
      return action_idx

  def bess_charge_step(self, state: EnvState, action: int) -> float:

      final_charge_step_size = 0.0
      adjusted_charge_step_size = 0.0
      bess_step_size = self.bess_step_size[action]
      
      if (state.grid_import_energy + bess_step_size) > self.grid_maximum_notified_demand:
          
          adjusted_charge_step_size = (self.grid_maximum_notified_demand - state.grid_import_energy)

      else:

          adjusted_charge_step_size = bess_step_size

      # 2800 + 500 = 3300 > 3000 = true
      if (self.bess_avail_discharge_energy + adjusted_charge_step_size) > self.bess_capacity:

          surplus_energy = (self.bess_avail_discharge_energy + adjusted_charge_step_size) - self.bess_capacity

          final_charge_step_size = adjusted_charge_step_size - surplus_energy

          self.bess_avail_discharge_energy = self.bess_capacity

      else:

          final_charge_step_size = adjusted_charge_step_size

          self.bess_avail_discharge_energy += final_charge_step_size

      return -1*final_charge_step_size
      
  def bess_discharge_step(self, state: EnvState, action: int) -> float:

      final_discharge_step_size = 0.0
      adjusted_discharge_step_size = 0.0
      bess_step_size = self.bess_step_size[action]
      
      if (state.grid_import_energy - bess_step_size) < 0.0:
          
          adjusted_discharge_step_size = state.grid_import_energy
          
      else:
          
          adjusted_discharge_step_size = bess_step_size 
          
      # 300 - 500 = -200 < 0.0 = true
      if (self.bess_avail_discharge_energy - adjusted_discharge_step_size) < 0.0:

          final_discharge_step_size = self.bess_avail_discharge_energy
          
          self.bess_avail_discharge_energy = 0.0

      else:

          final_discharge_step_size = adjusted_discharge_step_size

          self.bess_avail_discharge_energy -= final_discharge_step_size

      return final_discharge_step_size

  def calculate_reward(self, state: EnvState, action: int) -> float:

      bess_action = self.action_space.get(action, None)
      is_charge_action = re.match(r"^charge.*", bess_action) is not None
      is_discharge_action = re.match(r"^discharge.*", bess_action) is not None

      # This tariff values are in ZAR currency
      tou_offpeak_tariff = 1.0
      tou_standard_tariff = 2.0
      tou_peak_tariff = 5.0
      solar_ppa_tariff = 1.4

      bess_step_energy = None

      # Apply selected action to BESS system
      if is_charge_action: 

          bess_step_energy = self.bess_charge_step(state=self.state, action=action)

      elif is_discharge_action:

          bess_step_energy = self.bess_discharge_step(state=self.state, action=action)

      else: # do-nothing

          bess_step_energy = 0.0
          

      monetary_saving = 0.0

      # Calculate the reward that resulted from the action
      if state.tou_peak == 1:

          if is_charge_action:
              
              monetary_saving = ( (1.0 - (state.solar_ctlr_setpoint / 100.0)) * (bess_step_energy * solar_ppa_tariff) + (state.solar_ctlr_setpoint / 100.0) * (bess_step_energy * tou_peak_tariff) ) / 1000.0

          elif is_discharge_action:

              monetary_saving = (bess_step_energy * tou_peak_tariff) / 1000.0
              
      elif state.tou_standard == 1:

          if is_charge_action:
              
              monetary_saving = ( (1.0 - (state.solar_ctlr_setpoint / 100.0)) * (bess_step_energy * solar_ppa_tariff) + (state.solar_ctlr_setpoint / 100.0) * (bess_step_energy * tou_standard_tariff) ) / 1000.0

          elif is_discharge_action:

              monetary_saving = (bess_step_energy * tou_standard_tariff) / 1000.0
          
      elif state.tou_offpeak == 1:

          if is_charge_action:
              
              monetary_saving = ( (1.0 - (state.solar_ctlr_setpoint / 100.0)) * (bess_step_energy * solar_ppa_tariff) + (state.solar_ctlr_setpoint / 100.0) * (bess_step_energy * tou_offpeak_tariff) ) / 1000.0

          elif is_discharge_action:

              monetary_saving = (bess_step_energy * tou_offpeak_tariff) / 1000.0

      self.reward = (self.discount_factor * self.reward) + monetary_saving

      return self.reward

  def terminal_state(self) -> bool:

      nr_records = self.env_data.shape[0]

      self.done = (self.step_idx + 1) > (nr_records - 1)

      if self.done:
          
          print("Episode terminal state reached !")

      return self.done

  def step(self, action: int):

      bess_action = self.action_space.get(action, None)

      print(f""" 
      [{self.step_idx}] Selected Action -> {self.action_space.get(action, None)}
      """)

      reward = self.calculate_reward(state=self.state, action=action)

      print(f""" 
      [{self.step_idx}] Reward -> {reward: .3f}
      """)

      if self.terminal_state():
          return None, self.reward, self.done

      self.step_idx += 1   
      state = self.get_new_state()
      
      return state, self.reward, self.done
      
  def reset(self):
      
      self.reward = 0.0
      self.step_idx = 0
      self.bess_avail_discharge_energy = self.bess_capacity
      self.done = False
      state = self.get_new_state()
      
      return state, self.reward, self.done