from learning_fc.callbacks import ParamSchedule

STEPS=20

scalar_schedule = ParamSchedule(
    var_name="scalar",
    start=0.3,
    stop=0.7,
    first_value=0.0,
    final_value=10.0,
    total_timesteps=STEPS-1
)

list_schedule = ParamSchedule(
    var_name="list",
    start=0.0, 
    stop=1.0,
    first_value=[-0.1, 0.3],
    final_value=[0.3, -0.5],
    total_timesteps=STEPS-1
)

for i in range(STEPS): print(f"SCALAR[{i}]", scalar_schedule.get_value(i))
for i in range(STEPS): print(f"LIST[{i}]", list_schedule.get_value(i))