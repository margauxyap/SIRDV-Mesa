import mesa
import enum

import numpy as np

####################################################

class State(enum.IntEnum):  
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DEAD = 3
    VACCINATED = 4

####################################################
    
class MyAgent(mesa.Agent):
    """An agent in a rudimentary SIRDV model"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)   
        self.state = State.SUSCEPTIBLE  
        self.infection_time = 0

        #Add number of days before a newly-infected person becomes symptomatic
        self.incubation_time = np.random.normal(5, 2)  
        # Add number of days before a newly-infected person becomes infectious
        self.latency_time = np.random.normal(5, 2)  
        
        self.shots = 0  # Keep track of the number of vaccination shots
        
    def move(self):
        """Moves the agent to a new position in the grid"""
        # Dead agents do not move
        if self.state == State.DEAD:
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


    def update_status(self):
        """
        This method simulates the contact between agents in a grid. 
        Checks the infection status of the agent and updates its state based on the following rules:
        - If the agent is SUSCEPTIBLE: it may become INFECTED if there is an INFECTED cell-mate agent that has been exposed for longer its latency time (time to become infectious) at p(model.ptrans)
        - If the agent is VACCINATED: it may become INFECTED if there is an INFECTED cell-mate agent that has been exposed for longer its latency time (time to become infectious), but at a reduced transmission rate (based on the number of shots the VACCINATED agent has received)
        - If the agent is INFECTED:
            - if they have been exposed for <= their (individual) Recovery period, they may Die at each step at p(model.death_rate).
            - if they have been exposed for > their Individual Recovery period, they move to RECOVERED.
        - If an agent is DEAD, it does not come into contact with other agents. 
        - If an agent is RECOVERED, it can no longer become INFECTED, and just moves around the board. 

        Note: Looks from the perspective of what may happen (or others may do) to me -- NOT what I may do to others!  (Transmission view orientation is Receive not Send)
        """

        #Dead: stays dead! No zombies!
        if self.state == State.DEAD:
            return

        #Recovered: stays Recovered (assume full immunity)
        elif self.state == State.RECOVERED:
            return   

        # Infected: can Die (only if has become symptomatic), Recover (if meets or exceeds recovery_time), or stay Infected
        elif self.state == State.INFECTED and self.model.schedule.time - self.infection_time > self.incubation_time:     
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate,1-drate])
            if alive == 0:
                self.state = State.DEAD
            else:
                t = self.model.schedule.time-self.infection_time
                if t >= self.recovery_time:          
                    self.state = State.RECOVERED
            return

        
        #only need to examine cellmates if get this far...
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # Vaccinated: if comes into contact with an Infected (and infectious) agent, there's still a (diminished -- assume partial immunity) chance they could become infected. Otherwise, remain Vax'd
        if self.state == State.VACCINATED:  
            if len(cellmates) > 1:
                for other in cellmates:
                    if other.state is State.INFECTED and other.infection_time > other.latency_time and self.random.random() < self.model.ptrans * (0.5 if self.shots == 1 else 0.25 if self.shots >= 2 else 1):
                        self.state = State.INFECTED
                        self.infection_time = self.model.schedule.time
                        self.recovery_time = self.model.get_recovery_time()

        # Susceptible: Can contract through contact with Infected (and infectious) agent in same cell; otherwise, remains Susceptible  
        # [note: transition to VACCINATED happens in Model.step() not Agent.step()]
        elif self.state == State.SUSCEPTIBLE:
            if len(cellmates) > 1:
                for other in cellmates:
                    if other.state is State.INFECTED and other.infection_time > other.latency_time and self.random.random() < self.model.ptrans:
                        self.state = State.INFECTED
                        self.infection_time = self.model.schedule.time
                        self.recovery_time = self.model.get_recovery_time()

        


    def step(self):
        """
        Performs one step of the agent's life by calling status_update() then move() in order.
        """
        self.update_status()
        self.move()