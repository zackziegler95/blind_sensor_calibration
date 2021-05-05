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
    # l is a list of lists
    # n is an integer
    # returns True if the sublists combined contain exactly range(0, n)
    sorted_l = []
    for sub_l in l:
        sorted_l.extend(sub_l)
    sorted_l = list(sorted(sorted_l))
    return sorted_l == list(range(n))

class Simulator():
    '''
        Simulate the process of querying each sensor at a fixed interval
        Reads data from a folder data_dir to get the sensor reading events
        Optionally reads data from a proximity events file to simulate sensors changing
            their neighborhood
        At a high level, the simulator takes small discrete time steps and updates the global
            confidences at each step. If a sensor reading happened during the time step,
            the global sensor value is updated accordingly.
        The simulator is stateful, as values and confidences are updated globally
    '''

    def __init__(self, algorithm, data_dir, num_sensors, particle_size, start_time, delta=1, ghosts=False, max_sec=-1, subsample_freq=1, sensor_proximity_events_file=None):
        # delta is the step time in seconds
        self.delta = delta
        self.ghosts = ghosts
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

            # need to adjust the times so the format becomes seconds since a fixed time
            times = data['DeviceTimeStamp']
            times = pd.to_datetime(times, infer_datetime_format=True)
            times = times - pd.DateOffset(hours=4)
            times = times - start_time
            times = times.dt.total_seconds()

            cond = times > 0
            if max_sec > 0:
                cond = cond & (times < max_sec)
            times = times[cond]

            # no need to modify the particle density
            particle_density = data[particle_size]
            particle_density = particle_density[cond]
            self.all_data.append((times, particle_density))


    def step(self, t):
        # This is the main function of the simulator, which takes a step and updates the state
        # Updates are different depending on the algorithm
        endt = t + self.delta

        # always update neighbors based on prox_events
        new_prox_groupings = self.prox_groups[(self.prox_times >= t) & (self.prox_times < endt)]
        if len(new_prox_groupings) > 0:
            new_prox_groupings = new_prox_groupings.iloc[-1]
            assert contains_all_indices(new_prox_groupings, self.num_sensors)

            for i in range(self.num_sensors):
                for subgroup in new_prox_groupings:
                    if i in subgroup:
                        self.neighbors[i] = [subgroup_item for subgroup_item in subgroup if subgroup_item != i]
                        break

        # actually do the update for each sensor
        for i in range(self.num_sensors):
            # skip sensors which have run out of data
            if self.done[i]:
                continue

            # update values and times based on newly arrived data
            times_i = self.all_data[i][0]
            vals_i = self.all_data[i][1]
            new_data = vals_i[(times_i > t) & (times_i <= endt)]

            # update confidences: "Simple" multiplicative update
            if self.algorithm == 'simple':
                if len(new_data) > 0:
                    adjustment_rate = 0.1
                    self.values[i] = new_data.iloc[0]
                    # randomly choose other sensor that isn't done
                    other_live_sensor_indices = [j for j in self.neighbors[i] if not self.done[j]]
                    measurement_differences = [abs(self.values[i] - self.values[j]) for j in other_live_sensor_indices]
                    if self.ghosts:
                        baseline_uncertainty = min(11.25, self.values[i]*0.1125)
                        measurement_differences.append(baseline_uncertainty)
                    if len(measurement_differences) == 0:
                        continue
                    val_difference = choice(measurement_differences)
                    # update both (just the one?) confidences accordingly
                    multiplicative_adjustment = (1 + adjustment_rate * (val_difference/self.confidences[i] - 1))
                    self.confidences[i] = self.confidences[i] * multiplicative_adjustment

            # update confidences: "Complex" multiplicative update
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

            # the sensor is done if there are no more reading events in the data
            remaining_times = sum(times_i > endt)
            if remaining_times == 0:
                self.done[i] = True

        t = endt

        return t

    def check_done(self):
        # check if all the sensors indicate done
        for done in self.done:
            if not done:
                return False
        return True

    def simulate(self):
        # this function is an iterator, yields the sensor values and confidences at each step

        t = 0

        # initialize the values, confidences, and sets for each state
        self.values = [0 for _ in range(self.num_sensors)]
        self.confidences = [1 for _ in range(self.num_sensors)]
        self.done = [False for _ in range(self.num_sensors)]
        self.neighbors = [[] for _ in range(self.num_sensors)] # guarenteed to update at the first step

        # Iterate until all sensors indicate they have no more readings
        while True:
            if t % (60*10) == 0:
                minutes = t//60
                print(f'{minutes} minutes elapsed')

            # all the work happens here
            t = self.step(t)

            # save the state to return
            ret_vals = deepcopy(self.values)
            ret_confs = deepcopy(self.confidences)

            yield t, ret_vals, ret_confs

            # check if everyone is done
            if self.check_done():
                break

colorlist = ['b', 'y', 'r', 'g', 'm', 'c', 'indigo', 'darkorange']
def get_colors_from_group(prox_group, num_sensors):
    # return a flat list of colors for each sensor, determined by the group
    # prox_group is a list of lists, indicating the groupings of the sensor
    # num_sensors is an integer
    colors = ['' for _ in range(num_sensors)]
    for j in range(len(prox_group)):
        for sensorid in prox_group[j]:
            colors[sensorid] = colorlist[j]
    return colors

def group_results_by_proximity(results, prox_times, prox_groups):
    # convert a single set of results into a list of sets of results, one set of results per group
    # output:
    #   all_groupings - a list of results, one per group
    #   all_colors - a list of lists, one per group. The inner list has the color of each sensor during the given grouping
    times = [res[0] for res in results]
    values = [res[1] for res in results]
    confs = [res[2] for res in results]
    num_sensors = len(values[0])

    # current state
    last_prox_index = 0
    no_more_prox_events = len(prox_times) == last_prox_index + 1

    all_groupings = [] # each element of all_groupings is a list of details at each time, each detail is a list with the time, value, conf
    all_colors = [] # each element of all_colors is a list of the color of each sensor for the duration of that grouping
    current_grouping = []

    # loop through the times in order. When the sensor grouping changes, that indicates
    # the end of the previous result set and the start of the next set
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
    # calculates the InBounds and DistFromBound metrics
    # uncertainty scale is the conservativeness scaling metric
    # val_smoothing controls how much smoothing to apply to the value series
    num_sensors = len(results[0][1])

    total_in_bounds = 0
    total_in_bounds_options = 0

    total_dist_when_in_bounds = 0
    total_dist_when_in_bounds_options = 0

    prev_value = np.zeros([num_sensors])
    for time, value, conf in results:
        value = np.array(value)
        smooth_value = prev_value*val_smoothing + value*(1-val_smoothing) # standard EMA
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
        total_dist_when_in_bounds_options += in_bounds.sum() # denominator is sensors within the bounds
        prev_value = smooth_value

    avg_in_bounds = total_in_bounds/total_in_bounds_options
    avg_dist_when_in_bounds = total_dist_when_in_bounds/total_dist_when_in_bounds_options

    return avg_in_bounds, avg_dist_when_in_bounds

def smooth_values(results, smoothing=0):
    # Apply an EMA to the values component of a given results set
    num_sensors = len(results[0][1])

    prev_value = np.zeros([num_sensors])
    new_results = []
    for time, value, conf in results:
        value = np.array(value)
        smooth_value = prev_value*smoothing + value*(1-smoothing) # standard EMA
        new_results.append([time, smooth_value.tolist(), conf])
        prev_value = smooth_value

    return new_results


def main():
    # parameters
    runname = 'drop'
    algorithm = 'simple' # [baseline, simple, complex]
    data_dir = 'data_tscorrect'
    plot_dir = 'plots_experiment'
    sim_results_dir = 'saved_sims'
    num_sensors = 4 # up to 8 for this dataset
    particle_size = 'PM25'
    subsample_freq = 1 # i.e. only take every <value>th sample from each sensor
    max_sec = 60*630 # time in seconds
    sensor_proximity_events_file = 'proximity_events/4_occasional_drop.csv'
    plot_only = False
    color_by_group = False
    val_smoothing = 0
    notitle = True

    runname = runname+'_'+algorithm

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(sim_results_dir, exist_ok=True)

    # Handle either loading from a saved simulation or running the simulation
    if plot_only:
        with open(f'{sim_results_dir}/{runname}.pkl', 'rb') as f:
            combined_results, prox_times, prox_groups = pickle.load(f)
    else:
        # start_time is needed to anchor all sensors
        start_time = '04/11/2021 01:21:17 PM'
        start_time = pd.to_datetime(start_time, infer_datetime_format=True)

        sim = Simulator(algorithm, data_dir, num_sensors, particle_size, start_time, ghosts=False, max_sec=max_sec, subsample_freq=subsample_freq, sensor_proximity_events_file=sensor_proximity_events_file)

        combined_results = list(sim.simulate()) # all the work happens here
        prox_times = sim.prox_times
        prox_groups = sim.prox_groups

        # Save results to make changing the plots/analysis easier later
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

    # to ease plotting, break results into group by proximity events if coloring by group
    if color_by_group:
        result_groups, color_groups = group_results_by_proximity(combined_results, prox_times, prox_groups)
    else:
        result_groups = [combined_results] # only a single proximity grouping
        color_groups = [None]

    # Actually do the plotting
    plt.figure()
    for groupnum, (results, colors) in enumerate(zip(result_groups, color_groups)):
        times = [res[0] for res in results]
        values = [res[1] for res in results]
        confs = [res[2] for res in results]
        smooth_vals = [res[1] for res in smooth_combined_results]

        # Plot the full time series for each sensor at once
        for i in range(num_sensors):
            vals_i = np.array([val[i] for val in values])
            confs_i = np.array([conf[i] for conf in confs])
            smooth_vals_i = np.array([val[i] for val in smooth_vals])

            color = colors[i] if color_by_group else colorlist[i]
            plt.plot(times, smooth_vals_i, linewidth=0.8, color=color, label=f'Sensor{i+1}')
            plt.fill_between(times, vals_i-confs_i, vals_i+confs_i, alpha=0.2, color=color)

    # Misc plot labels
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
