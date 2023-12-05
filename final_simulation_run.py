# Import necessary libraries and modules
import model 
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Define the number of simulation steps
k_steps = 100

# Create the initial model instance
my_model = model.GridInfectionModel(seed=101, N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)

# Record the start time of the simulation
start_time = time.time()

# Run the simulation for 'k_steps' steps
for i in range(k_steps):
    my_model.step()

# Get data from the model
my_model_data = my_model.datacollector.get_model_vars_dataframe()
my_model_data

# Record the end time of the simulation
end_time = time.time()

# Print the time taken for the simulation
print(f"This model run of {k_steps} steps was completed in {round(end_time-start_time,1)} secs")

# Running 50 independent replications for scenario 1 (no vaccination delay)

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

    
# Running 50 independent replications for scenario 2 (30-day delay, double vaccination rate)    
    
# Initialize a list to store data from model3 replications
model3_rep_data = []

# Run replications of model3 with different random seeds
for i in range(num_reps):
    
    model3_rep = model.GridInfectionModel(seed = random.seed(i), N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.10, vaccination_delay=30, shots_needed=2)

    for j in range(100):
        model3_rep.step()
    
    model3_rep_data.append(model3_rep.datacollector.get_model_vars_dataframe())

    print(f"Model 3 Replication {i+1} done")
    
# Get number of replications
num_reps = np.shape(model2_rep_data)[0]


# Calculate mean of each variable across replications 
mean_data2_list = []

for i in range(num_reps):
    mean_data2 = []
    
    for col in model2_rep_data[i].columns:
        # We're using max to get the total number of agents per state throughout the simulation
        mean_i = model2_rep_data[i][col].values.max()
        mean_data2.append(mean_i)
    
    mean_data2_list.append(mean_data2)

mean_data2_df = pd.DataFrame(mean_data2_list, columns=[col for col in model2_rep_data[0].columns])
mean_data2_df

# Get the number of replications for model3
num_reps = np.shape(model3_rep_data)[0]


# Calculate mean of each variable across replications 
mean_data3_list = []

for i in range(num_reps):
    mean_data3 = []
    
    for col in model3_rep_data[i].columns:
        mean_i = model3_rep_data[i][col].values.max()
        mean_data3.append(mean_i)
    
    mean_data3_list.append(mean_data3)

mean_data3_df = pd.DataFrame(mean_data3_list, columns=[col for col in model3_rep_data[0].columns])
mean_data3_df

# Calculate the mean of 'Total Dead' variable for model2
sample_vals2 = mean_data2_df['Total Dead'].values
sample_mean_death2 = sample_vals2.mean()
sample_mean_death2

# Calculate the mean of 'Total Dead' variable for model3
sample_vals3 = mean_data3_df['Total Dead'].values
sample_mean_death3 = sample_vals3.mean()
sample_mean_death3

# Calculate the difference in means between model2 and model3
diff_in_means = mean_data2_df['Total Dead'].mean() - mean_data3_df['Total Dead'].mean()

# Calculate the sample variance and sample size for model2
sample_var_2 = mean_data2_df['Total Dead'].var()
n_2 = len(mean_data2_df['Total Dead'])

# Calculate the sample variance and sample size for model3
sample_var_3 = mean_data3_df['Total Dead'].var()
n_3 = len(mean_data3_df['Total Dead'])

# Calculate the estimated standard deviation of the difference in means
diff_estd_sample_sd = np.sqrt(sample_var_2/n_2 + sample_var_3/n_3)

# Calculate the degrees of freedom
nu = (sample_var_2/n_2 + sample_var_3/n_3)**2 / ((sample_var_2/n_2)**2 / n_2+1 + (sample_var_3/n_3)**2 / n_3+1) - 2

# Calculate the t-score for the difference
t_score_for_diff = stats.t.ppf(0.975, nu)

# Calculate the confidence interval for the difference in means
CI_diff = (diff_in_means - t_score_for_diff*diff_estd_sample_sd , diff_in_means + t_score_for_diff*diff_estd_sample_sd)


# Plot the results for model2
plt.figure(figsize=(10, 8))
plt.plot(mean_data2_whole_df['Total Susceptible'], label='Susceptible')
plt.plot(mean_data2_whole_df['Total Infected'], label='Infected')
plt.plot(mean_data2_whole_df['Total Recovered'], label='Recovered')
plt.plot(mean_data2_whole_df['Total Dead'], label='Dead')
plt.plot(mean_data2_whole_df['Total Vaccinated'], label='Vaccinated')
plt.axvline(mean_data2_df['Total Infected'].idxmax(), color='gray', linestyle='--')
plt.text(mean_data2_df['Total Infected'].idxmax(), mean_data2_df['Total Infected'].max(), f"Peak Infection: {mean_data2_df['Total Infected'].idxmax()}", color='red', fontsize=10, ha='center', va='bottom')
plt.text(mean_data2_df['Total Dead'].idxmax(), mean_data2_df['Total Dead'].max(), f"Total Dead: {mean_data2_df['Total Dead'].max()}", color='red', fontsize=10, ha='center', va='bottom')
plt.legend()
plt.title('No Vaccination Delay - Average Run')
plt.xlabel('Days')  # Add x-axis label
plt.ylabel('Population')  # Add y-axis label
plt.show()

# Plot the results for model3
plt.figure(figsize=(10, 8))
plt.plot(mean_data3_whole_df['Total Susceptible'], label='Susceptible')
plt.plot(mean_data3_whole_df['Total Infected'], label='Infected')
plt.plot(mean_data3_whole_df['Total Recovered'], label='Recovered')
plt.plot(mean_data3_whole_df['Total Dead'], label='Dead')
plt.plot(mean_data3_whole_df['Total Vaccinated'], label='Vaccinated')
plt.axvline(mean_data3_df['Total Infected'].idxmax(), color='gray', linestyle='--')
plt.text(mean_data3_df['Total Infected'].idxmax(), mean_data3_df['Total Infected'].max(), f"Peak Infection: {mean_data3_df['Total Infected'].idxmax()}", color='red', fontsize=10, ha='center', va='bottom')
plt.text(mean_data3_df['Total Dead'].idxmax(), mean_data3_df['Total Dead'].max(), f"Total Dead: {mean_data3_df['Total Dead'].max()}", color='red', fontsize=10, ha='center', va='bottom')
plt.legend()
plt.title('30-Day Vaccination Delay, Double Vaccination Rate - Average Run')
plt.xlabel('Days')  # Add x-axis label
plt.ylabel('Population')  # Add y-axis label
plt.show()


