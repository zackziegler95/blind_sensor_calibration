import sys
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, data_dir, num_sensors, size, start_time, delta=1, max_sec=-1):
        # delta is the step time in seconds
        self.delta = delta

        self.all_data = []
        for i in range(1, num_sensors+1):
            data = pd.read_csv(f'{data_dir}/Sensor{i}.csv')

            times = data['DeviceTimeStamp']
            times = pd.to_datetime(times, infer_datetime_format=True)
            times = times - pd.DateOffset(hours=4)
            times = times - start_time
            times = times.dt.total_seconds()

            cond = times > 0
            if max_sec > 0:
                cond = cond & (times < max_sec)
            times = times[cond]


            particle_density = data[size]
            particle_density = particle_density[cond]
            #particle_density = particle_density.rolling(20).mean()
            self.all_data.append((times, particle_density))

    def step(self, t):
        endt = t + self.delta

        for i in range(len(self.all_data)):
            if self.done[i]:
                continue

            times_i = self.all_data[i][0]
            vals_i = self.all_data[i][1]
            new_data = vals_i[(times_i > t) & (times_i <= endt)]

            if len(new_data) > 0:
                self.values[i] = new_data.iloc[0]
                if self.values[i] < 100: # 100 ug/m3
                    self.confidences[i] = 11.25
                elif self.values[i] < 1000: # 1000 ug/m3
                    self.confidences[i] = self.values[i]*0.1125
                else:
                    self.confidences[i] = 999999 # no confidence at all!

            remaining_times = sum(times_i > endt)
            if remaining_times == 0:
                self.done[i] = True

        t = endt

        return t

    def check_done(self):
        for done in self.done:
            if not done:
                return False
        return True

    def simulate(self):
        # all_data is a list of pandas dataframes
        t = 0
        self.values = [0 for _ in range(len(self.all_data))]
        self.confidences = [1 for _ in range(len(self.all_data))]
        self.done = [False for _ in range(len(self.all_data))]

        while True:
            if t % (60*10) == 0:
                minutes = t//60
                print(f'{minutes} minutes elapsed')

            t = self.step(t)

            ret_vals = deepcopy(self.values)
            ret_confs = deepcopy(self.confidences)

            yield t, ret_vals, ret_confs

            if self.check_done():
                break

def main():
    data_dir = 'data_tscorrect'
    plot_dir = 'plots_experiment'
    num_sensors = 4
    size = 'PM25'
    max_sec = 60*630

    os.makedirs(plot_dir, exist_ok=True)

    start_time = '04/11/2021 01:21:17 PM'
    start_time = pd.to_datetime(start_time, infer_datetime_format=True)

    sim = Simulator(data_dir, num_sensors, size, start_time, max_sec=max_sec)
    
    results = list(sim.simulate())
    times = [res[0] for res in results]
    values = [res[1] for res in results]
    confs = [res[2] for res in results]

    plt.figure()
    for i in range(num_sensors):
        vals_i = np.array([val[i] for val in values])
        confs_i = np.array([conf[i] for conf in confs])
        plt.plot(times, vals_i, linewidth=0.8, label=f'Sensor{i+1}')
        plt.fill_between(times, vals_i-confs_i, vals_i+confs_i, alpha=0.2)

    plt.xlabel('Time (seconds)')
    plt.ylabel(f'{size} (ug/m3)')
    plt.ylim([0, 50])
    plt.legend()
    plt.savefig(f'{plot_dir}/all_data_{size}.pdf')

    plt.ylim([0, 500])
    plt.savefig(f'{plot_dir}/all_data_{size}_largescale.pdf')

if __name__ == '__main__':
    main()
