model2 = GridInfectionModel(seed=101, N=10000, width=10, height=10, ptrans=0.002, 
                           progression_period=5, progression_sd=2, death_rate=0.00904, recovery_days=14,
                           recovery_sd=7, vaccination_rate=0.05, vaccination_delay=0, shots_needed=2)

for i in range(100):
    model2.step()

model_data2 = model2.datacollector.get_model_vars_dataframe()
model_data2