#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mesa')


# In[2]:


import enum
import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy import stats
import pandas as pd


import model 
import time
pd.options.display.max_rows = 150


# Parameters:
# 
# - `N`: The number of agents in the model. The default is 10,000 agents.
# - `width` and `height`: The size of the grid that the agents move around on. The default is 10x10.
# - `ptrans`: The probability of transmission of the infection when a susceptible and infected agent come into contact. The default value is 0.002.
# - `exposed_time`: controls the number of days that an infected agent is exposed before becoming infectious. When an agent first becomes infected, they are exposed to the disease but not yet able to transmit it to others. The exposed_time represents this incubation period.
# - `incubation_time`: The time between infection and symptom onset drawn from a normal distribution with mean 5 and standard deviation 2. 
# - `death_rate`: The probability that an infected agent dies each step. The default is 0.00904.
# - `recovery_days` and `recovery_sd`: The mean and standard deviation of the normal distribution used to determine the recovery time of infected agents. In this case, the mean recovery time is 14 days and the standard deviation is 7 days.
# - `vaccination_rate`: The proportion of susceptible individuals who get vaccinated each step. The default is 0.05.
# 

# **Note:** We start with Model 2 since Model 1 was the base model without an incubation period, which is not what’s required for this project.

# ## Defining Model Parameters 

# In[3]:


# Define the number of simulation steps
k_steps = 100

# Create the initial model instance
my_model = model.GridInfectionModel(seed=101, N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)


# # Sample Script for 1 Run

# In[4]:


# Record the start time of the simulation
start_time = time.time()

# Run the simulation for 'k_steps' steps
for i in range(k_steps):
    my_model.step()
    
# Record the end time of the simulation
end_time = time.time()

# Print the time taken for the simulation
print(f"This model run of {k_steps} steps was completed in {round(end_time-start_time,1)} secs")

# Get data from the model
my_model_data = my_model.datacollector.get_model_vars_dataframe()
my_model_data


# ## Model 2 - with Incubation Period, NO vaccination delay

# In[5]:


# Define the number of replications
num_reps = 50
model2_rep_data = []

# Run multiple replications of the model with different random seeds
for i in range(num_reps):
    model2_rep = model.GridInfectionModel(seed = random.seed(i), N=10000, width=10, height=10, ptrans=0.002,  
                              progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                              recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)

    for j in range(100):
        model2_rep.step()
    
    model2_rep_data.append(model2_rep.datacollector.get_model_vars_dataframe())

    print(f"Model 2 Replication {i+1} done")
    
# Record the end time of the simulation
end_time = time.time()

# Print the time taken for the simulation
print(f"This model run of {num_reps} replications was completed in {round(end_time-start_time,1)} secs")


# # Calculating mean states across replicatons (Model 2)

# In[6]:


# Convert the list to a numpy array
data_array2 = np.array(model2_rep_data)

# Calculate the mean across the first dimension
mean_data2 = data_array2.mean(axis=0)

mean_data2_df = pd.DataFrame(mean_data2, columns=[col for col in model2_rep_data[0].columns])
mean_data2_df


# In[7]:


plt.figure(figsize=(10, 8))
plt.plot(mean_data2_df['Total Susceptible'], label='Susceptible')
plt.plot(mean_data2_df['Total Infected'], label='Infected')
plt.plot(mean_data2_df['Total Recovered'], label='Recovered')
plt.plot(mean_data2_df['Total Dead'], label='Dead')
plt.plot(mean_data2_df['Total Vaccinated'], label='Vaccinated')
plt.axvline(mean_data2_df['Total Infected'].idxmax(), color='gray', linestyle='--')
plt.text(mean_data2_df['Total Infected'].idxmax(), mean_data2_df['Total Infected'].max(), f"Peak Infection: {mean_data2_df['Total Infected'].idxmax()}", color='red', fontsize=10, ha='center', va='bottom')
plt.text(mean_data2_df['Total Dead'].idxmax(), mean_data2_df['Total Dead'].max(), f"Total Dead: {mean_data2_df['Total Dead'].max()}", color='red', fontsize=10, ha='center', va='bottom')
plt.title('No Vaccination Delay - 50 Reps')
plt.legend()
plt.savefig("Model2.png")
plt.show()


# # Model 3 - with 30-Day Vaccination Delay, Double Vaccination Rate

# In[8]:


# Record the start time of the simulation
start_time = time.time()

# Initialize a list to store data from model3 replications
model3_rep_data = []

# Run replications of model3 with different random seeds (note vaccination rate of 0.10)
for i in range(num_reps):
    
    model3_rep = model.GridInfectionModel(seed = random.seed(i), N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.10, vaccination_delay=30, shots_needed=2)

    for j in range(100):
        model3_rep.step()
    
    model3_rep_data.append(model3_rep.datacollector.get_model_vars_dataframe())

    print(f"Model 3 Replication {i+1} done")
    
# Record the end time of the simulation
end_time = time.time()

# Print the time taken for the simulation
print(f"This model run of {num_reps} replications was completed in {round(end_time-start_time,1)} secs")


# # Calculating mean states across replicatons (Model 3)

# In[9]:


# Convert the list to a numpy array
data_array3 = np.array(model3_rep_data)

# Calculate the mean across the first dimension
mean_data3 = data_array3.mean(axis=0)

mean_data3_df = pd.DataFrame(mean_data3, columns=[col for col in model3_rep_data[0].columns])
mean_data3_df


# In[10]:


plt.figure(figsize=(10, 8))
plt.plot(mean_data3_df['Total Susceptible'], label='Susceptible')
plt.plot(mean_data3_df['Total Infected'], label='Infected')
plt.plot(mean_data3_df['Total Recovered'], label='Recovered')
plt.plot(mean_data3_df['Total Dead'], label='Dead')
plt.plot(mean_data3_df['Total Vaccinated'], label='Vaccinated')
plt.axvline(mean_data3_df['Total Infected'].idxmax(), color='gray', linestyle='--')
plt.text(mean_data3_df['Total Infected'].idxmax(), mean_data3_df['Total Infected'].max(), f"Peak Infection: {mean_data3_df['Total Infected'].idxmax()}", color='red', fontsize=10, ha='center', va='bottom')
plt.text(mean_data3_df['Total Dead'].idxmax(), mean_data3_df['Total Dead'].max(), f"Total Dead: {mean_data3_df['Total Dead'].max()}", color='red', fontsize=10, ha='center', va='bottom')
plt.title('With 30-Day Vaccination Delay, Double Vaccination Rate - 50 Reps')
plt.legend()
plt.savefig("Model3.png")
plt.show()


# ## Calculating Difference in Means between 2 Scenarios

# In[11]:


# Get 'Total Dead' at the last step for model2 (no vaccination delay)
sample_vals2 = mean_data2_df['Total Dead'].values
sample_mean_death2 = sample_vals2.max()
sample_mean_death2


# In[12]:


# Get 'Total Dead' at the last step for model3 (with 30-day delay, but double vaccination rate)
sample_vals3 = mean_data3_df['Total Dead'].values
sample_mean_death3 = sample_vals3.max()
sample_mean_death3


# In[13]:


# Calculate the difference in means between model2 and model3
diff_in_means = mean_data2_df['Total Dead'].mean() - mean_data3_df['Total Dead'].mean()

print(f"The mean difference in the Total number of deaths between the 2 scenarios is {diff_in_means}")


# ## Calculating the Approximate Confidence Interval for Difference between 2 Means

# In[15]:


# Calculate the sample variance and sample size for model2
sample_var_2 = mean_data2_df['Total Dead'].var()
n_2 = len(mean_data2_df['Total Dead'])

# Calculate the sample variance and sample size for model3
sample_var_3 = mean_data3_df['Total Dead'].var()
n_3 = len(mean_data3_df['Total Dead'])

# Calculate the estimated standard deviation of the difference in means
diff_estd_sample_sd = np.sqrt(sample_var_2/n_2 + sample_var_3/n_3)

# Calculate the approximate degrees of freedom
nu = (sample_var_2/n_2 + sample_var_3/n_3)**2 / ((sample_var_2/n_2)**2 / n_2+1 + (sample_var_3/n_3)**2 / n_3+1) - 2

# Calculate the t-score for the difference
t_score_for_diff = stats.t.ppf(0.975, nu)

# Calculate the confidence interval for the difference in means
CI_diff = (diff_in_means - t_score_for_diff*diff_estd_sample_sd , diff_in_means + t_score_for_diff*diff_estd_sample_sd)
CI_diff

print(f"The confidence interval at 0.05 alpha between the 2 scenarios is {CI_diff}")


# Since the lower and upper bounds of the confidence interval fall below zero, we can be reasonably sure that the mean of `Total Dead` for **Model 2** (no vaccination delays) is lower than that of **Model 3** (30-day delay, double vaccination rate).  Therefore, **Model 2** is better in terms of controlling the spread of disease, despite the increased vaccination rate for **Model 3**.
