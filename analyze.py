import sys
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'data_tscorrect'
plot_dir = 'plots'
num_sensors = 8
size = 'PM1'

plt.figure()

start_time = '04/11/2021 01:21:17 PM'
start_time = pd.to_datetime(start_time, infer_datetime_format=True)
data = pd.read_csv(f'{data_dir}/GroundTruth.csv')
times = pd.to_timedelta(data['Elapsed Time [s]'], unit='s')
times = start_time + times

particle_density = data[f'{size} [mg/m3]']*1000
times = times - start_time
times = times.dt.total_seconds()/3600
plt.plot(times, particle_density, label='GroundTruth')

size = size.replace('.', '')

for i in range(1, num_sensors+1):
    data = pd.read_csv(f'{data_dir}/Sensor{i}.csv')

    times = data['DeviceTimeStamp']
    print(times)
    times = pd.to_datetime(times, infer_datetime_format=True)
    times = times - pd.DateOffset(hours=4)
    times = times - start_time
    times = times.dt.total_seconds()/3600

    particle_density = data[size]
    particle_density = particle_density.rolling(20).mean()

    plt.plot(times, particle_density, label=f'Sensor{i}')

plt.xlabel('Time (hours)')
plt.ylabel(f'{size} (ug/m3)')
plt.xlim([0, 11.5])
plt.ylim([0, 50])
plt.legend()
plt.savefig(f'{plot_dir}/all_data_{size}.png')

plt.ylim([0, 500])
plt.savefig(f'{plot_dir}/all_data_{size}_largescale.png')
