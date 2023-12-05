###########################################
## Import necessary libraries and modules

#Mesa-based
import model 

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

###########################################
## Parameters
k_steps = 100   # K steps per replication
num_reps = 50   # N Replications

######################################################################################
#### STEP 1: RUN THE REPLICATIONS 

###########################################
# Model 1 Replication Runs

# Initialize a list to store data from model1 replications
model1_rep_data = []

# Run multiple replications of the model with different random seeds
for i in range(num_reps):
    model1_rep = model.GridInfectionModel(seed = random.seed(i), N=10000, width=10, height=10, ptrans=0.002,  
                              progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                              recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)
    
    # Record the start time of the simulation
    start_time = time.time()
    
    for j in range(100):
        model1_rep.step()
    
    # Record the end time of the simulation
    end_time = time.time()
    
    model1_rep_data.append(model1_rep.datacollector.get_model_vars_dataframe())

    print(f"Model 1 Replication {i+1} of {num_reps} (of {k_steps} steps each) was completed in {round(end_time-start_time,1)} secs")
 
    
###########################################
# Model 2 (30-day delay, but double vax rate) Replication Runs

# Initialize a list to store data from model2 replications 
model2_rep_data = []

# Run replications of model2 with different random seeds
for i in range(num_reps):    
    model2_rep = model.GridInfectionModel(seed = random.seed(i), N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.10, vaccination_delay=30, shots_needed=2)

    # Record the start time of the simulation
    start_time = time.time()
     
    for j in range(100):
        model2_rep.step()
    
    # Record the end time of the simulation
    end_time = time.time()
    
    model2_rep_data.append(model2_rep.datacollector.get_model_vars_dataframe())

    print(f"Model 2 Replication {i+1} of {num_reps} (of {k_steps} steps each) was completed in {round(end_time-start_time,1)} secs")
    
 

######################################################################################
#### STEP 2: ANALYZE THE REPLICATIONS

###########################################
## Model 1 Summary
# Get number of replications
num_reps = np.shape(model1_rep_data)[0]


# Calculate summary stats (here, max of each) for each variable across replications 
summary_data_per_rep__model1_list = []

for i in range(num_reps):
    summary_data_per_rep__model1 = []
    
    for col in model1_rep_data[i].columns:
        # We're using max to get the total number of agents per state throughout the simulation
        summary_i = model1_rep_data[i][col].values.max()
        summary_data_per_rep__model1.append(summary_i)
    
    summary_data_per_rep__model1_list.append(summary_data_per_rep__model1)

summary_data_per_rep__model1_df = pd.DataFrame(summary_data_per_rep__model1_list, columns=[col for col in model1_rep_data[0].columns])
summary_data_per_rep__model1_df


###########################################
## Model 2 Summary
# Get the number of replications for model2
num_reps = np.shape(model2_rep_data)[0]


# Calculate mean of each variable across replications 
summary_data_per_rep__model2_list = []

for i in range(num_reps):
    summary_data_per_rep__model2 = []
    
    for col in model2_rep_data[i].columns:
        # We're using max to get the total number of agents per state throughout the simulation
        summary_i = model2_rep_data[i][col].values.max()
        summary_data_per_rep__model2.append(summary_i)
    
    summary_data_per_rep__model2_list.append(summary_data_per_rep__model2)

summary_data_per_rep__model2_df = pd.DataFrame(summary_data_per_rep__model2_list, columns=[col for col in model2_rep_data[0].columns])
summary_data_per_rep__model2_df




###########################################
## CALCULATE STATS TO COMPARE MODEL 1 vs MODEL 2
# Calculate the mean of 'Total Dead' variable for model1
sample_vals2 = summary_data_per_rep__model1_df['Total Dead'].values
sample_mean_death2 = sample_vals2.mean()
sample_mean_death2

# Calculate the mean of 'Total Dead' variable for model2
sample_vals3 = summary_data_per_rep__model2_df['Total Dead'].values
sample_mean_death3 = sample_vals3.mean()
sample_mean_death3

# Calculate the difference in means between model1 and model2
diff_in_means = summary_data_per_rep__model1_df['Total Dead'].mean() - summary_data_per_rep__model2_df['Total Dead'].mean()

# Calculate the sample variance and sample size for model1
sample_var_1 = summary_data_per_rep__model1_df['Total Dead'].var()
n_1 = len(summary_data_per_rep__model1_df['Total Dead'])

# Calculate the sample variance and sample size for model1
sample_var_2 = summary_data_per_rep__model2_df['Total Dead'].var()
n_2 = len(summary_data_per_rep__model2_df['Total Dead'])

# Calculate the estimated standard deviation of the difference in means
diff_estd_sample_sd = np.sqrt(sample_var_1/n_1 + sample_var_2/n_2)

# Calculate the degrees of freedom
nu = (sample_var_1/n_1 + sample_var_2/n_2)**1 / ((sample_var_1/n_1)**1 / n_1+1 + (sample_var_2/n_2)**1 / n_2+1) - 1

# Calculate the t-score for the difference
t_score_for_diff = stats.t.ppf(0.975, nu)

# Calculate the confidence interval for the difference in means
CI_diff = (diff_in_means - t_score_for_diff*diff_estd_sample_sd , diff_in_means + t_score_for_diff*diff_estd_sample_sd)







######################################################################################
#### STEP 3: VIZ

###########################################
## VIZ PREP

# Convert the list of model1 rep data to a numpy array
data_array1 = np.array(model1_rep_data)

# Calculate the mean across the first dimension for model1
# that is, the average number of Agents in each State at each step in the run -- summarized across the K replications
summary_data_per_rep_1_whole = data_array1.mean(axis=0)
summary_data_per_rep_1_whole_df = pd.DataFrame(summary_data_per_rep_1_whole, columns=[col for col in model1_rep_data[0].columns])


# Convert the list of model2 rep data to a numpy array
data_array2 = np.array(model2_rep_data)

# Calculate the mean across the first dimension for model2
# that is, the average number of Agents in each State at each step in the run -- summarized across the K replications
summary_data_per_rep_2_whole = data_array2.mean(axis=0)
summary_data_per_rep_2_whole_df = pd.DataFrame(summary_data_per_rep_2_whole, columns=[col for col in model1_rep_data[0].columns])


###########################################
# Now: Plot the results for model1
plt.figure(figsize=(10, 8))
plt.plot(summary_data_per_rep_1_whole_df['Total Susceptible'], label='Susceptible')
plt.plot(summary_data_per_rep_1_whole_df['Total Infected'], label='Infected')
plt.plot(summary_data_per_rep_1_whole_df['Total Recovered'], label='Recovered')
plt.plot(summary_data_per_rep_1_whole_df['Total Dead'], label='Dead')
plt.plot(summary_data_per_rep_1_whole_df['Total Vaccinated'], label='Vaccinated')
plt.legend()
plt.title('No Vaccination Delay - Average Run')
plt.xlabel('Days')  # Add x-axis label
plt.ylabel('Population')  # Add y-axis label
plt.show()

# Plot the results for model2
plt.figure(figsize=(10, 8))
plt.plot(summary_data_per_rep_2_whole_df['Total Susceptible'], label='Susceptible')
plt.plot(summary_data_per_rep_2_whole_df['Total Infected'], label='Infected')
plt.plot(summary_data_per_rep_2_whole_df['Total Recovered'], label='Recovered')
plt.plot(summary_data_per_rep_2_whole_df['Total Dead'], label='Dead')
plt.plot(summary_data_per_rep_2_whole_df['Total Vaccinated'], label='Vaccinated')
plt.legend()
plt.title('30-Day Vaccination Delay, Double Vaccination Rate - Average Run')
plt.xlabel('Days')  # Add x-axis label
plt.ylabel('Population')  # Add y-axis label
plt.show()