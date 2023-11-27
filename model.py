import mesa

#our SIRDV Agent
import agent

import random
import numpy as np
from scipy import stats
import pandas as pd
#import matplotlib.pyplot as plt

####################################################

State = agent.State


# Modified model class to initiate vaccination after `vaccination_delay` to simulate supply chain issues 
# Vaccination starts after a specified delay and continues at each step. Agents receive one shot at a time. 
# Once an agent has received the required number of shots, their state changes to 'VACCINATED'.

class GridInfectionModel(mesa.Model):
    def __init__(self, seed=101, N=10, width=10, height=10, ptrans=0.02, 
                 progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                 recovery_sd=7, vaccination_rate=0.02, vaccination_delay=0, shots_needed=2):
        self.seed = seed

        self.num_agents = N
        self.initial_outbreak_size = 1
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.ptrans = ptrans
        self.death_rate = death_rate
        self.vaccination_rate = vaccination_rate
        self.vaccination_delay = vaccination_delay
        self.shots_needed = shots_needed

        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        
        self.dead_agents = []
        # Create agents
        for i in range(self.num_agents):
            a = agent.MyAgent(i, self)
            self.schedule.add(a)
            
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            #make some agents infected at start
            infected = np.random.choice([0,1], p=[0.98,0.02])
            if infected == 1:
                a.state = State.INFECTED
                a.recovery_time = self.get_recovery_time()
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Total Susceptible": lambda m: self.count_type(m, State.SUSCEPTIBLE),
                             "Total Infected": lambda m: self.count_type(m, State.INFECTED),
                             "Total Recovered": lambda m: self.count_type(m, State.RECOVERED),
                             "Total Dead": lambda m: self.count_type(m, State.DEAD),
                             "Total Vaccinated": lambda m: self.count_type(m, State.VACCINATED)},
            agent_reporters={"State": "state"})

    def get_recovery_time(self):
        """Returns the recovery time for an infected individual"""
        return int(self.random.normalvariate(self.recovery_days,self.recovery_sd))
    
    def count_type(self, model, state):
        """Counts the number of agents in a given state"""
        count = 0
        for agent in model.schedule.agents:
            if agent.state is state:
                count += 1
        return count
    
    def step(self):
        """This method performs one step in the model's schedule. 
        Vaccination starts after a specified delay and continues at each step. Agents receive one shot at a time. 
        Once an agent has received the required number of shots, their state changes to 'VACCINATED'."""
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.time >= self.vaccination_delay:
            susceptible_agents = [a for a in self.schedule.agents if a.state is State.SUSCEPTIBLE]
            for a in self.random.sample(susceptible_agents, min(len(susceptible_agents), int(self.vaccination_rate * len(susceptible_agents)))):
                a.shots += 1
                if a.shots >= self.shots_needed:
                    a.state = State.VACCINATED