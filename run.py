import model #rename later to be more specific, a la "SIRDV_model"
import time

#############################

k_steps = 100

my_model = model.GridInfectionModel(seed=101, N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)

start_time = time.time()

for i in range(k_steps):
    my_model.step()

my_model_data = my_model.datacollector.get_model_vars_dataframe()
my_model_data

end_time = time.time()

print(f"This model run of {k_steps} steps was completed in {round(end_time-start_time,1)} secs")