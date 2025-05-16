import torch
import numpy as np

from sklearn.metrics.pairwise import haversine_distances

from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
  SubstepObservation,
  SubstepAction,
  SubstepTransition,
)

from utilities import *

@Registry.register_substep("gather_solar_inputs", "observation")
class GatherSolarAndHouseholdData(SubstepObservation):
  def forward(self, state):
    facility_latitudes = read_var(state, self.input_variables['facility/latitude'])
    facility_longitudes = read_var(state, self.input_variables['facility/longitude'])
    facility_solar_productions = read_var(state, self.input_variables['facility/solar_production_kwh'])
    facility_energy_demands = read_var(state, self.input_variables['facility/energy_demand_kwh'])
    household_latitudes = read_var(state, self.input_variables['household/latitude'])
    household_longitudes = read_var(state, self.input_variables['household/longitude'])
    household_has_solar = read_var(state, self.input_variables['household/has_solar'])
    household_adoption_propensities = read_var(state, self.input_variables['household/adoption_propensity'])

    current_month = int(
      read_var(state, self.input_variables['environment/current_month']).item() % 12,
    )

    facility_monthly_production = facility_solar_productions[:, current_month]
    facility_monthly_demand = facility_energy_demands[:, current_month]
    # facility_excess_energies = torch.clamp(
    #   facility_monthly_production - facility_monthly_demand,
    #   min=0,
    # )
    facility_excess_energies = facility_monthly_production 

    return {
      'facility_latitudes': facility_latitudes,
      'facility_longitudes': facility_longitudes,
      'facility_excess_energies': facility_excess_energies,
      'household_latitudes': household_latitudes,
      'household_longitudes': household_longitudes,
      'household_has_solar': household_has_solar,
      'household_adoption_propensities': household_adoption_propensities,
    }

@Registry.register_substep("diffuse_and_decide", "policy")
class DiffuseSolarExcessAndDecide(SubstepAction):
  def forward(self, state, observation):
    grid_price = read_var(state, self.input_variables['environment/grid_price']).item()
    installation_cost = read_var(state, self.input_variables['environment/installation_cost']).item()
    maintenance_cost = read_var(state, self.input_variables['environment/maintenance_cost']).item()
    diffusion_radius = read_var(state, self.input_variables['environment/diffusion_radius']).item()
    distance_epsilon = read_var(state, self.input_variables['environment/distance_epsilon']).item()

    facility_lat = observation['facility_latitudes'].numpy().reshape(-1, 1)
    facility_lon = observation['facility_longitudes'].numpy().reshape(-1, 1)
    facility_excess = observation['facility_excess_energies']
    household_lat = observation['household_latitudes'].numpy().reshape(-1, 1)
    household_lon = observation['household_longitudes'].numpy().reshape(-1, 1)
    household_has_solar = observation['household_has_solar'].squeeze().bool()
    household_propensity = observation['household_adoption_propensities'].squeeze()

    facility_coords = np.hstack([facility_lat, facility_lon]) * (np.pi / 180.0)
    household_coords = np.hstack([household_lat, household_lon]) * (np.pi / 180.0)
    distances = haversine_distances(facility_coords, household_coords) * 6371.0
    distance_matrix = torch.from_numpy(distances).float()
    within_radius = (distance_matrix <= diffusion_radius).float()

    weights = within_radius / (distance_matrix + distance_epsilon)
    normalization = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    normalized_weights = weights / normalization

    allocation = facility_excess.unsqueeze(1) * normalized_weights
    received_energy = (allocation * grid_price)
    annual_savings = received_energy.sum(dim=0)
    return_on_investment = (annual_savings - maintenance_cost / 12) / installation_cost
    random_values = torch.rand_like(household_propensity)
    new_adoption = (
      (return_on_investment > 0)
      & (random_values < household_propensity)
      & (~household_has_solar)
    )

    return {
      'annual_savings': annual_savings.unsqueeze(1),
      'adoption_decision': new_adoption,
    }

@Registry.register_substep("update_household_solar_state", "transition")
class UpdateHouseholdSolarAdoption(SubstepTransition):
  def forward(self, state, action):
    current_solar_status = read_var(state, self.input_variables['household/has_solar']).bool().squeeze()
    adoption_events = action['household']['adoption_decision']
    annual_savings = action['household']['annual_savings']

    updated_solar_status = current_solar_status | adoption_events

    return {
      'household/has_solar': updated_solar_status.unsqueeze(1),
      'household/annual_savings': annual_savings,
    }
