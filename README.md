# Simple SIRDV Contagion Model Using Mesa

This repository contains a simple agent-based model to simulate the spread of disease in a population and model the effects of vaccination. It uses the [Mesa](https://github.com/projectmesa/mesa/) Python library to build an SIRDV model

## Overview

The model consists of agents moving randomly on a multigrid. Agents can be in one of several states:

- Susceptible - can become infected  
- Infected - can infect susceptible agents
- Recovered - had the disease but recovered
- Dead - died from the disease 
- Vaccinated - received vaccine shots and has (partial) immunity

Infected agents go through an incubation period where they are exposed but not yet infectious. Once infectious, they can transmit to susceptibles within their grid cell based on a transmission probability.

The model allows configuring vaccination to start after a delay. Vaccination is applied to random susceptible agents in each step. Agents receive vaccine shots one at a time, gaining partial immunity after the first shot. Once the required number of shots is reached, the agent is fully vaccinated.

## Key Files

- `agent.py` - Defines the agent class and logic
- `model.py` - Implements the overall SIRDV model
- `run.py` - Example script to run a model simulation

## Running the Model

To run a simulation:

1. Install requirements (check requirements.txt)
2. Modify `agent.py` and `model.py` parameters
3. Run `model.py` code. See sample code below:
4. Run `final_simulation_run.py` for multiple replications

```python
model2 = GridInfectionModel(N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)

for i in range(100):
    model2.step()

model_data2 = model2.datacollector.get_model_vars_dataframe()
```

Key parameters:

- `N`: The number of agents in the model. The default is 10,000 agents.
- `width` and `height`: The size of the grid that the agents move around on. The default is 10x10.
- `ptrans`: The probability of transmission of the infection when a susceptible and infected agent come into contact. The default value is 0.002.
- `exposed_time`: controls the number of days that an infected agent is exposed before becoming infectious. When an agent first becomes infected, they are exposed to the disease but not yet able to transmit it to others. The exposed_time represents this incubation period.
- `incubation_time`: The time between infection and symptom onset drawn from a normal distribution with mean 5 and standard deviation 2. 
- `death_rate`: The probability that an infected agent dies each step after exceeding the incubation time. The default value is 0.00904. 
- `recovery_days` and `recovery_sd`: The mean and standard deviation of the normal distribution used to determine the recovery time of infected agents. In this case, the mean recovery time is 14 days and the standard deviation is 7 days.
- `vaccination_rate`: The proportion of susceptible individuals who get vaccinated each step. The default is 0.05.
- `vaccination_delay` - Steps before vaccination starts. The default is 0 (no delay)
- `shots_needed` - 1 or 2 shots for immunity

## Future Work

Possible extensions:

- Age-based mortality rates
- Different transmission rates based on age
- Vaccine efficacy levels instead of full/partial immunity
- Contact Network Grid instead of a Multigrid

## References

Kazil, J., Masad, D., & Crooks, A. (2020). Utilizing Python for Agent-Based modeling: the MESA Framework. In Lecture Notes in Computer Science (pp. 308–317). https://doi.org/10.1007/978-3-030-61255-9_30
