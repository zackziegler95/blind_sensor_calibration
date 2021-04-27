import sys
import pickle
import os
from copy import deepcopy
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

def contains_all_indices(l, n):
    sorted_l = []
    for sub_l in l:
        sorted_l.extend(sub_l)
    sorted_l = list(sorted(sorted_l))
    return sorted_l == list(range(n))

class Simulator():
    def __init__(self, data_dir, num_sensors, size, start_time, delta=1, max_sec=-1, subsample_freq=1, sensor_proximity_events_file=None):
        # delta is the step time in seconds
        self.delta = delta
        self.num_sensors = num_sensors

        # Read proximity events
        prox_events = pd.read_csv(sensor_proximity_events_file, converters={1:ast.literal_eval})
        times = prox_events['minutes_from_start']*60
        assert times[0] == 0, 'groups should be defined from time 0'
        groups = prox_events['sensorids']
        self.prox_times = times
        self.prox_groups = groups

        self.all_data = []
        for i in range(1, num_sensors+1):
            data = pd.read_csv(f'{data_dir}/Sensor{i}.csv')
            data = data.iloc[::subsample_freq, :]

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
            self.all_data.append((times, particle_density))


    def step(self, t):
        endt = t + self.delta

        # update neighbors based on prox_events
        new_prox_groupings = self.prox_groups[(self.prox_times >= t) & (self.prox_times < endt)]
        
        if len(new_prox_groupings) > 0:
            new_prox_groupings = new_prox_groupings.iloc[-1]
            assert contains_all_indices(new_prox_groupings, self.num_sensors)

            for i in range(self.num_sensors):
                for subgroup in new_prox_groupings:
                    if i in subgroup:
                        self.neighbors[i] = [subgroup_item for subgroup_item in subgroup if subgroup_item != i]
                        break
        
        for i in range(self.num_sensors):
            if self.done[i]:
                continue


            # update values and confidences based on data

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
        self.values = [0 for _ in range(self.num_sensors)]
        self.confidences = [1 for _ in range(self.num_sensors)]
        self.done = [False for _ in range(self.num_sensors)]
        self.neighbors = [[] for _ in range(self.num_sensors)]

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

colorlist = ['b', 'y', 'r', 'g', 'm', 'c', 'indigo', 'darkorange']
def get_colors_from_group(prox_group, num_sensors):
    colors = ['' for _ in range(num_sensors)]
    for j in range(len(prox_group)):
        for sensorid in prox_group[j]:
            colors[sensorid] = colorlist[j]
    return colors

def group_results_by_proximity(results, prox_times, prox_groups):
    times = [res[0] for res in results]
    values = [res[1] for res in results]
    confs = [res[2] for res in results]
    num_sensors = len(values[0])

    last_prox_index = 0
    no_more_prox_events = len(prox_times) == last_prox_index + 1

    all_groupings = [] # each element of all_groupings is a list of details at each time, each detail is a list with the time, value, conf
    all_colors = [] # each element of all_colors is a list of the color of each sensor for the duration of that grouping
    current_grouping = []

    for i in range(len(times)):
        # prox event triggered
        if not no_more_prox_events and times[i] >= prox_times[last_prox_index+1]:
            all_groupings.append(current_grouping)
            all_colors.append(get_colors_from_group(prox_groups[last_prox_index], num_sensors))
            current_grouping = []
            last_prox_index += 1
            no_more_prox_events = len(prox_times) == last_prox_index + 1

        current_grouping.append([times[i], values[i], confs[i]])

    all_groupings.append(current_grouping)
    all_colors.append(get_colors_from_group(prox_groups[last_prox_index], num_sensors))           

    return all_groupings, all_colors


def main():
    # parameters
    runname = 'baseline'
    data_dir = 'data_tscorrect'
    plot_dir = 'plots_experiment'
    sim_results_dir = 'saved_sims'
    num_sensors = 4
    size = 'PM25'
    subsample_freq = 1 # i.e. only take every <value>th sample from each sensor
    max_sec = 60*630 #60*630
    sensor_proximity_events_file = 'proximity_events/4_occasional_drop.csv'
    plot_only = False
    color_by_group = False

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(sim_results_dir, exist_ok=True)
    
    # Run simulation
    if plot_only:
        with open(f'{sim_results_dir}/{runname}.pkl', 'rb') as f:
            combined_results, prox_times, prox_groups = pickle.load(f)
    else:
        start_time = '04/11/2021 01:21:17 PM'
        start_time = pd.to_datetime(start_time, infer_datetime_format=True)

        sim = Simulator(data_dir, num_sensors, size, start_time, max_sec=max_sec, subsample_freq=subsample_freq, sensor_proximity_events_file=sensor_proximity_events_file)
        
        combined_results = list(sim.simulate())
        prox_times = sim.prox_times
        prox_groups = sim.prox_groups
        with open(f'{sim_results_dir}/{runname}.pkl', 'wb') as f:
            pickle.dump([combined_results, prox_times, prox_groups], f)

    # Plot results
    if color_by_group:
        result_groups, color_groups = group_results_by_proximity(combined_results, prox_times, prox_groups)
    else:
        result_groups = [combined_results]
        color_groups = [None]

    plt.figure()
    for groupnum, (results, colors) in enumerate(zip(result_groups, color_groups)):
        times = [res[0] for res in results]
        values = [res[1] for res in results]
        confs = [res[2] for res in results]

        for i in range(num_sensors):
            vals_i = np.array([val[i] for val in values])
            confs_i = np.array([conf[i] for conf in confs])

            color = colors[i] if color_by_group else colorlist[i]
            plt.plot(times, vals_i, linewidth=0.8, color=color, label=f'Sensor{i+1}')
            plt.fill_between(times, vals_i-confs_i, vals_i+confs_i, alpha=0.2, color=color)

    plt.xlabel('Time (seconds)')
    plt.ylabel(f'{size} (ug/m3)')
    plt.ylim([0, 50])

    if not color_by_group:
        plt.legend()
    plt.savefig(f'{plot_dir}/{runname}_all_data_{size}.pdf')
    plt.savefig(f'{plot_dir}/{runname}_all_data_{size}.png')

    plt.ylim([0, 500])
    plt.savefig(f'{plot_dir}/{runname}_all_data_{size}_largescale.pdf')
    plt.savefig(f'{plot_dir}/{runname}_all_data_{size}_largescale.png')

if __name__ == '__main__':
    main()
