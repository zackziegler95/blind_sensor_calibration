# Same plotting as analyze.py, but for the second set of data we didn't end up using

import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'data_tscorrect_new'
plot_dir = 'plots_new'
num_sensors = 8
size = 'PM1'

os.makedirs(plot_dir, exist_ok=True)

# Here the ground truth is separated into two separate captures
start_time = '04/15/2021 07:48:12 PM'
start_time = pd.to_datetime(start_time, infer_datetime_format=True)
data = pd.read_csv(f'{data_dir}/BaselineGroundTruth.csv')
times = pd.to_timedelta(data['Elapsed Time [s]'], unit='s')
times = start_time + times

particle_density_gt = data[f'{size} [mg/m3]']*1000 # 1000 should be right
particle_density_gt = particle_density_gt.rolling(20).mean()
times = times - start_time
times_gt = times.dt.total_seconds()/3600


start_time2 = '04/15/2021 08:50:27 PM'
start_time2 = pd.to_datetime(start_time2, infer_datetime_format=True)
data = pd.read_csv(f'{data_dir}/CandleLeftGroundTruth.csv')
times = pd.to_timedelta(data['Elapsed Time [s]'], unit='s')
times = start_time + times

particle_density_gt2 = data[f'{size} [mg/m3]']*1000 # 1000 should be right
particle_density_gt2 = particle_density_gt2.rolling(20).mean()
times = times - start_time
time_offset = start_time2 - start_time
times = times + time_offset
times_gt2 = times.dt.total_seconds()/3600


size = size.replace('.', '')

plt.figure()
plt.plot(times_gt, particle_density_gt, 'blue', label='GroundTruth')
plt.plot(times_gt2, particle_density_gt2, 'blue', label='GroundTruth')

# The cheap sensors have a single capture for the full time period
for i in range(1, num_sensors+1):
    fname = glob.glob(f'{data_dir}/Sensor{i}*.csv')[0]
    print(fname)
    data = pd.read_csv(fname)

    times = data['DeviceTimeStamp']
    times = pd.to_datetime(times, infer_datetime_format=True)
    times = times - pd.DateOffset(hours=4)
    times = times - start_time
    times = times.dt.total_seconds()/3600

    particle_density = data[size]
    particle_density = particle_density.rolling(20).mean()

    plt.plot(times, particle_density, label=f'Sensor{i}')


plt.xlabel('Time (hours)')
plt.ylabel(f'{size} (ug/m3)')
#plt.xlim([0, 11.5])
plt.ylim([0, 50])
plt.legend()
plt.savefig(f'{plot_dir}/all_data_{size}.png')

plt.ylim([0, 500])
plt.savefig(f'{plot_dir}/all_data_{size}_largescale.png')
