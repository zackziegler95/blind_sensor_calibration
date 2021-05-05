# Initial exploration of the data to see what's up
# This is for the first set of data we ended up using in the project

import sys
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'data_tscorrect'
plot_dir = 'plots'
num_sensors = 8
size = 'PM1'

plt.figure()

# First, read the ground truth sensor, taking into account its peculiarities
start_time = '04/11/2021 01:21:17 PM'
start_time = pd.to_datetime(start_time, infer_datetime_format=True)
data = pd.read_csv(f'{data_dir}/GroundTruth.csv')
times = pd.to_timedelta(data['Elapsed Time [s]'], unit='s')
times = start_time + times

particle_density = data[f'{size} [mg/m3]']*333 # 1000 should be right, but something is weird
particle_density_gt = particle_density.rolling(20).mean()
times = times - start_time # convert to seconds since start
times_gt = times.dt.total_seconds()/3600

size = size.replace('.', '')

# Next, read the cheap sensor readings
for i in range(1, num_sensors+1):
    data = pd.read_csv(f'{data_dir}/Sensor{i}.csv')
    
    # times are written as timestamps instead of seconds since starting
    times = data['DeviceTimeStamp']
    times = pd.to_datetime(times, infer_datetime_format=True)
    times = times - pd.DateOffset(hours=4)
    times = times - start_time
    times = times.dt.total_seconds()/3600

    particle_density = data[size]
    particle_density = particle_density.rolling(20).mean()

    plt.plot(times, particle_density, label=f'Sensor{i}')

plt.plot(times_gt, particle_density_gt, label='GroundTruth')

# Misc plotting things
plt.xlabel('Time (hours)')
plt.ylabel(f'{size} (ug/m3)')
plt.xlim([0, 11.5])
plt.ylim([0, 50])
plt.legend()
plt.savefig(f'{plot_dir}/all_data_{size}.png')

plt.ylim([0, 500])
plt.savefig(f'{plot_dir}/all_data_{size}_largescale.png')
