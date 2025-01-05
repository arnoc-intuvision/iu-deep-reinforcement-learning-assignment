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

    def get_solar_vs_load_ratio(self) -> float:

        return np.round((self.solar_prod_energy / self.site_load_energy) * 100, 2)

    def get_grid_surplus_energy(self, notified_maximum_demand: float) -> float:

        return np.round(notified_maximum_demand - self.grid_import_energy, 2)

    def get_solar_surplus_energy(self) -> float:
        
        return np.round( (self.solar_prod_energy / (self.solar_ctlr_setpoint / 100)) - self.solar_prod_energy, 2)


class MicrogridEnv:

  def __init__(self, data: pd.DataFrame, debug_flag: bool):
      self.env_data = data
      self.debug_flag = debug_flag
      self.state = None
      self.state_idx = 0
      self.reward = 0.0
      self.grid_notified_maximum_demand = 2000.0
      self.bess_capacity = 3000.0
      self.bess_avail_discharge_energy = self.bess_capacity
      self.bess_step_size = [1500.0, 1000.0, 500.0, 0.0, 500.0, 1000.0, 1500.0]
      self.action_space = {0: 'charge-1500', 1: 'charge-1000', 2: 'charge-500', 3: 'do-nothing', 4: 'discharge-500', 5: 'discharge-1000', 6: 'discharge-1500'} 
      self.done = False
      self.tou_offpeak_tariff = 1.0
      self.tou_standard_tariff = 2.0
      self.tou_peak_tariff = 5.0
      self.solar_ppa_tariff = 1.4
      self.display_data_info()

  def display_data_info(self):

      if self.debug_flag:

          print("Environment Defaults: ")
          print(f"""
          Grid Notified Maximum Demand: {self.grid_notified_maximum_demand} kVA
          BESS Capacity: {self.bess_capacity} kWh
          BESS Actions: {', '.join( list( self.action_space.values() ) )}
          """)

          print("\nData Summary: ")
          print(self.env_data.info())
          print("\n")

  def get_new_state(self) -> EnvState:

      obs = self.env_data.iloc[self.state_idx, : ]

      self.state = EnvState(**obs)

      if self.debug_flag:
      
          print(f"""
          [{self.state_idx}] Environment State ->
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

  def get_number_of_actions(self):

      return len(self.action_space)

  def sample_action(self):

      return np.random.choice( self.get_number_of_actions() )

  def rule_based_policy(self):

      # Action space indices
      charge_1500_idx = 0
      charge_1000_idx = 1
      charge_500_idx = 2
      do_nothing_idx = 3
      discharge_500_idx = 4
      discharge_1000_idx = 5
      discharge_1500_idx = 6

      action_idx = None
      state = self.state

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

      charge_step_size = self.bess_step_size[action]

      grid_surplus_energy = state.get_grid_surplus_energy(notified_maximum_demand=self.grid_notified_maximum_demand)
      solar_surplus_energy = state.get_solar_surplus_energy()
      
      required_charge_from_grid_import_energy = charge_step_size - solar_surplus_energy

      charge_from_solar_energy = 0.0
      charge_from_grid_import_energy = 0.0

      # Determine the possible bess charge energy split between solar and grid
      if (required_charge_from_grid_import_energy > 0) and (grid_surplus_energy > required_charge_from_grid_import_energy):
          
          charge_from_grid_import_energy = required_charge_from_grid_import_energy
          charge_from_solar_energy = solar_surplus_energy

      elif (required_charge_from_grid_import_energy > 0) and (grid_surplus_energy < required_charge_from_grid_import_energy):

          charge_from_grid_import_energy = grid_surplus_energy
          charge_from_solar_energy = solar_surplus_energy

      else:

          charge_from_grid_import_energy = 0.0
          charge_from_solar_energy = charge_step_size

      adjusted_charge_step_size = charge_from_solar_energy + charge_from_grid_import_energy
      charge_from_solar_prop = 1.0 if charge_from_grid_import_energy == 0.0 else (charge_from_solar_energy / adjusted_charge_step_size)
      charge_from_grid_import_prop = 1.0 - charge_from_solar_prop

      final_charge_step_size = adjusted_charge_step_size
      final_charge_from_solar_energy = charge_from_grid_import_energy
      final_charge_from_grid_import_energy = charge_from_solar_energy

      # Apply the bess charge energy to the battery storage, and adjust for over charging condition
      if (self.bess_avail_discharge_energy + adjusted_charge_step_size) > self.bess_capacity:

          surplus_charge_energy = (self.bess_avail_discharge_energy + adjusted_charge_step_size) - self.bess_capacity

          final_charge_step_size = adjusted_charge_step_size - surplus_charge_energy
          final_charge_from_solar_energy = final_charge_step_size * charge_from_solar_prop
          final_charge_from_grid_import_energy = final_charge_step_size * charge_from_grid_import_prop

          # BESS fully charged
          self.bess_avail_discharge_energy = self.bess_capacity

      else:

          # BESS busy charging
          self.bess_avail_discharge_energy += final_charge_step_size

      # Calculate the total cost for charging from solar and grid energy sources
      charge_from_grid_cost = 0.0
      charge_from_solar_cost = final_charge_from_solar_energy * ( -1 * self.solar_ppa_tariff + (self.tou_offpeak_tariff + self.tou_standard_tariff + self.tou_peak_tariff) / 3.0 )

      if state.tou_peak == 1:
          
          charge_from_grid_cost = final_charge_from_grid_import_energy * ( -1 * self.tou_peak_tariff + (self.tou_offpeak_tariff + self.tou_standard_tariff) / 2.0 )

      elif state.tou_standard == 1:
          
          charge_from_grid_cost = final_charge_from_grid_import_energy * ( -1 * self.tou_standard_tariff + (self.tou_offpeak_tariff + self.tou_peak_tariff) / 2.0 )
          
      elif state.tou_offpeak == 1:
          
          charge_from_grid_cost = final_charge_from_grid_import_energy * ( -1 * self.tou_offpeak_tariff + (self.tou_standard_tariff + self.tou_peak_tariff) / 2.0 )

      return (charge_from_solar_cost + charge_from_grid_cost) / 1000.0

  def bess_discharge_step(self, state: EnvState, action: int) -> float:

      discharge_step_size = self.bess_step_size[action]

      adjusted_discharge_step_size = 0.0

      # Determine the possible discharge energy from the grid import energy
      if (state.grid_import_energy - discharge_step_size) < 0.0:
          
          adjusted_discharge_step_size = state.grid_import_energy
          
      else:
          
          adjusted_discharge_step_size = discharge_step_size 
          
      final_discharge_step_size = 0.0

      # Apply the bess discharge energy to the battery storage, and adjust for over discharging condition
      if (self.bess_avail_discharge_energy - adjusted_discharge_step_size) < 0.0:

          final_discharge_step_size = self.bess_avail_discharge_energy

          # BESS fully discharged
          self.bess_avail_discharge_energy = 0.0

      else:

          final_discharge_step_size = adjusted_discharge_step_size

          # BESS busy discharging
          self.bess_avail_discharge_energy -= final_discharge_step_size

      # Calculate the total cost saving for discharging the battery storage into the load
      discharge_into_load_cost_saving = 0.0
      
      if state.tou_peak == 1:
          
          discharge_into_load_cost_saving = final_discharge_step_size * ( self.tou_peak_tariff - (self.tou_offpeak_tariff + self.tou_standard_tariff) / 2 )

      elif state.tou_standard == 1:
          
          discharge_into_load_cost_saving = final_discharge_step_size * ( self.tou_standard_tariff - (self.tou_offpeak_tariff + self.tou_peak_tariff) / 2 )
          
      elif state.tou_offpeak == 1:
          
          discharge_into_load_cost_saving = final_discharge_step_size * ( self.tou_offpeak_tariff - (self.tou_standard_tariff + self.tou_peak_tariff) / 2 )

      return discharge_into_load_cost_saving / 1000.0

  def calculate_reward(self, state: EnvState, action: int) -> float:

      bess_action = self.action_space.get(action, None)
      is_charge_action = re.match(r"^charge.*", bess_action) is not None
      is_discharge_action = re.match(r"^discharge.*", bess_action) is not None

      # Apply the selected charge/discharge action to battery storage
      if is_charge_action: 

          self.reward = self.bess_charge_step(state=self.state, action=action)

      elif is_discharge_action:

          self.reward = self.bess_discharge_step(state=self.state, action=action)

      else: # do-nothing

          self.reward = 0.0

      return self.reward

  def terminal_state(self) -> bool:

      nr_records = self.env_data.shape[0]

      self.done = (self.state_idx + 1) > (nr_records - 1)

      if self.done and self.debug_flag:
          
          print("Episode terminal state reached !")

      return self.done

  def step(self, action: int):

      bess_action = self.action_space.get(action, None)

      if self.debug_flag:

          print(f""" 
          [{self.state_idx}] Selected Action -> {self.action_space.get(action, None)}
          """)

      reward = self.calculate_reward(state=self.state, action=action)

      if self.debug_flag:

          print(f""" 
          [{self.state_idx}] Reward -> {reward: .3f}
          """)

      if self.terminal_state():
          return None, self.reward, self.done

      self.state_idx += 1   
      state = self.get_new_state()
      
      return self.state_idx, self.reward, self.done
      
  def reset(self):
      
      self.reward = 0.0
      self.state_idx = 0
      self.bess_avail_discharge_energy = self.bess_capacity
      self.done = False
      state = self.get_new_state()
      
      return self.state_idx, self.reward, self.done