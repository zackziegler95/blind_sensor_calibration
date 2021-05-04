import sys
import pickle
import os
from copy import deepcopy
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from random import choice

def contains_all_indices(l, n):
    sorted_l = []
    for sub_l in l:
        sorted_l.extend(sub_l)
    sorted_l = list(sorted(sorted_l))
    return sorted_l == list(range(n))

class Simulator():
    def __init__(self, algorithm, data_dir, num_sensors, particle_size, start_time, delta=1, max_sec=-1, subsample_freq=1, sensor_proximity_events_file=None):
        # delta is the step time in seconds
        self.delta = delta
        self.num_sensors = num_sensors
        self.algorithm = algorithm

        # Read proximity events
        prox_events = pd.read_csv(sensor_proximity_events_file, converters={1:ast.literal_eval})
        times = prox_events['minutes_from_start']*60
        assert times[0] == 0, 'groups should be defined from time 0'
        groups = prox_events['sensorids']
        self.prox_times = times
        self.prox_groups = groups

        self.all_data = []
        # populate all_data from CSVs
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


            particle_density = data[particle_size]
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
            # skip sensors which have run out of data
            if self.done[i]:
                continue

            # update values and times based on newly arrived data
            times_i = self.all_data[i][0]
            vals_i = self.all_data[i][1]
            new_data = vals_i[(times_i > t) & (times_i <= endt)]

            # update confidences: "Simple MW update"
            if self.algorithm == 'simple':
                if len(new_data) > 0:
                    adjustment_rate = 0.1
                    self.values[i] = new_data.iloc[0]
                    # randomly choose other sensor that isn't done
                    other_live_sensor_indices = [j for j in range(self.num_sensors) if not j==i and not self.done[j]]
                    if len(other_live_sensor_indices) == 0:
                        continue
                    j = choice(other_live_sensor_indices)
                    # take difference in values
                    val_difference = self.values[i] - self.values[j]
                    # update both (just the one?) confidences accordingly
                    multiplicative_adjustment = (1 + adjustment_rate * (abs(val_difference)/self.confidences[i] - 1))
                    self.confidences[i] = self.confidences[i] * multiplicative_adjustment

            # update confidences: "Complex" MW update
            if self.algorithm == 'complex':
                if len(new_data) > 0:
                    adjustment_rate = 0.1
                    self.values[i] = new_data.iloc[0]
                    # randomly choose other sensor that isn't done
                    other_live_sensor_indices = [j for j in range(self.num_sensors) if not j==i and not self.done[j]]
                    if len(other_live_sensor_indices) == 0:
                        continue
                    j = choice(other_live_sensor_indices)
                    # take difference in values
                    val_difference = self.values[i] - self.values[j]
                    # update both (just the one?) confidences accordingly
                    sum_of_confidences = self.confidences[i] + self.confidences[j]
                    multiplicative_adjustment = (1 + adjustment_rate * (abs(val_difference)/sum_of_confidences - 1))
                    self.confidences[i] = self.confidences[i] * multiplicative_adjustment
                    self.confidences[j] = self.confidences[j] * multiplicative_adjustment

            # "Baseline" approach
            if self.algorithm == 'baseline':
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
        # stores current vals and confidences:
        self.values = [0 for _ in range(self.num_sensors)]
        self.confidences = [1 for _ in range(self.num_sensors)]
        # keeps track of whether sensor has run out of data:
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

def calculate_metric_crossval(results, uncertainty_scale=1, val_smoothing=0):
    num_sensors = len(results[0][1])
    
    total_in_bounds = 0
    total_in_bounds_options = 0

    total_dist_when_in_bounds = 0
    total_dist_when_in_bounds_options = 0

    prev_value = np.zeros([num_sensors])
    for time, value, conf in results:
        value = np.array(value)
        smooth_value = prev_value*val_smoothing + value*(1-val_smoothing)
        conf = np.array(conf)*uncertainty_scale

        upper_bounds = value + conf
        lower_bounds = value - conf
        
        # number of sensors that are in bounds of the other sensors
        val_ext = smooth_value[:, np.newaxis]
        ub_ext = upper_bounds[np.newaxis, :]
        lb_ext = lower_bounds[np.newaxis, :]
        in_bounds = np.logical_and(val_ext < ub_ext, val_ext > lb_ext) # [N, N]
        np.fill_diagonal(in_bounds, 0)
        total_in_bounds += in_bounds.sum()
        total_in_bounds_options += (num_sensors-1)*num_sensors

        # distance between sensor values and confidence bounds
        dist_to_ub = np.maximum(ub_ext - val_ext, 0)
        dist_to_lb = -np.minimum(lb_ext - val_ext, 0)
        min_dist = np.minimum(dist_to_ub, dist_to_lb)
        np.fill_diagonal(min_dist, 0)

        total_dist_when_in_bounds += min_dist.sum()
        total_dist_when_in_bounds_options += in_bounds.sum()
        prev_value = smooth_value
    
    avg_in_bounds = total_in_bounds/total_in_bounds_options
    avg_dist_when_in_bounds = total_dist_when_in_bounds/total_dist_when_in_bounds_options

    return avg_in_bounds, avg_dist_when_in_bounds

def smooth_values(results, smoothing=0):
    num_sensors = len(results[0][1])

    prev_value = np.zeros([num_sensors])
    new_results = []
    for time, value, conf in results:
        value = np.array(value)
        smooth_value = prev_value*smoothing + value*(1-smoothing)
        new_results.append([time, smooth_value.tolist(), conf])
        prev_value = smooth_value

    return new_results

def main():
    # parameters
    runname = 'final_4sensors'
    algorithm = 'baseline'
    data_dir = 'data_tscorrect'
    plot_dir = 'plots_experiment'
    sim_results_dir = 'saved_sims'
    num_sensors = 4
    particle_size = 'PM25'
    subsample_freq = 1 # i.e. only take every <value>th sample from each sensor
    max_sec = 60*630 #60*630
    sensor_proximity_events_file = 'proximity_events/4_constant.csv'
    plot_only = True
    color_by_group = False
    val_smoothing = 0
    notitle = True

    runname = runname+'_'+algorithm

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(sim_results_dir, exist_ok=True)

    # Run simulation
    if plot_only:
        with open(f'{sim_results_dir}/{runname}.pkl', 'rb') as f:
            combined_results, prox_times, prox_groups = pickle.load(f)
    else:
        start_time = '04/11/2021 01:21:17 PM'
        start_time = pd.to_datetime(start_time, infer_datetime_format=True)

        sim = Simulator(algorithm, data_dir, num_sensors, particle_size, start_time, max_sec=max_sec, subsample_freq=subsample_freq, sensor_proximity_events_file=sensor_proximity_events_file)

        combined_results = list(sim.simulate())
        prox_times = sim.prox_times
        prox_groups = sim.prox_groups
        with open(f'{sim_results_dir}/{runname}.pkl', 'wb') as f:
            pickle.dump([combined_results, prox_times, prox_groups], f)

    # Calculate "correctness" metrics
    avg_in_bounds, avg_dist_when_in_bounds = calculate_metric_crossval(combined_results)
    print(f'Average number of readings inside co-located sensor confidence intervals: {avg_in_bounds:0.3f}')
    print(f'Average distance from sensor readings to co-located sensor confidence bound when inside bound: {avg_dist_when_in_bounds:0.3f}')

    # Plot results
    if val_smoothing > 0:
        assert not color_by_group
        smooth_combined_results = smooth_values(combined_results, val_smoothing)
    else:
        smooth_combined_results = combined_results

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
        smooth_vals = [res[1] for res in smooth_combined_results]

        for i in range(num_sensors):
            vals_i = np.array([val[i] for val in values])
            confs_i = np.array([conf[i] for conf in confs])
            smooth_vals_i = np.array([val[i] for val in smooth_vals])

            color = colors[i] if color_by_group else colorlist[i]
            plt.plot(times, smooth_vals_i, linewidth=0.8, color=color, label=f'Sensor{i+1}')
            plt.fill_between(times, vals_i-confs_i, vals_i+confs_i, alpha=0.2, color=color)

    plt.xlabel('Time (seconds)')
    plt.ylabel(f'{particle_size} (ug/m3)')
    plt.ylim([0, 50])

    if not notitle:
        plt.title(f'Metric 1: {avg_in_bounds:0.3f}   Metric 2: {avg_dist_when_in_bounds:0.3f}')

    if not color_by_group:
        plt.legend()
    plt.savefig(f'{plot_dir}/{runname}_all_data_{particle_size}_smooth{val_smoothing}.pdf')
    plt.savefig(f'{plot_dir}/{runname}_all_data_{particle_size}_smooth{val_smoothing}.png')

    plt.ylim([0, 500])
    plt.savefig(f'{plot_dir}/{runname}_all_data_{particle_size}_smooth{val_smoothing}_largescale.pdf')
    plt.savefig(f'{plot_dir}/{runname}_all_data_{particle_size}_smooth{val_smoothing}_largescale.png')

if __name__ == '__main__':
    main()
