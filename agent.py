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
        # Add number of days before a newly-infected person becomes infectious
        self.exposed_time = np.random.normal(5, 2)  
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


        
    def status(self):
        """
        Checks the infection status of the agent and updates its state based on the following rules:
        - If the agent is infected and has been exposed for longer than their exposed time, they may either die or recover.
        - If the agent is vaccinated, they may become infected if they come into contact with an infected agent.
        """
        if self.state == State.INFECTED and self.model.schedule.time - self.infection_time > self.exposed_time:     
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate,1-drate])
            if alive == 0:
                self.state = State.DEAD
            else:
                t = self.model.schedule.time-self.infection_time
                if t >= self.recovery_time:          
                    self.state = State.RECOVERED
        elif self.state == State.VACCINATED:
            # If a vaccinated agent comes into contact with an infected agent, there's a chance they could become infected.
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            if len(cellmates) > 1:
                for other in cellmates:
                    if other.state is State.INFECTED and self.random.random() < self.model.ptrans:
                        self.state = State.INFECTED
                        self.infection_time = self.model.schedule.time
                        self.recovery_time = self.model.get_recovery_time()

    def contact(self):
        """
        This method simulates the contact between agents in a grid. If an agent is dead, it does not come into contact with other agents.
        If an agent is infected and comes into contact with a susceptible agent, the susceptible agent may become infected based on the transmission rate.
        """
        if self.state == State.DEAD:
            return
        cellmates = self.model.grid.get_cell_list_contents([self.pos])       
        if len(cellmates) > 1:
            for other in cellmates:
                # Agents with one shot get partial immunity (reduced transmission rate)
                # while the transmission rates of agents with two shots are cut by 75%
                if self.random.random() > self.model.ptrans * (0.5 if self.shots == 1 else 0.25 if self.shots >= 2 else 1):
                    continue
                if self.state is State.INFECTED and other.state is State.SUSCEPTIBLE:                    
                    other.state = State.INFECTED
                    other.infection_time = self.model.schedule.time
                    other.recovery_time = self.model.get_recovery_time()

    def step(self):
        """
        Performs one step of the agent's life by calling the status, move, and contact methods in order.
        """
        self.status()
        self.move()
        self.contact()