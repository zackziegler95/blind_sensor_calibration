# Plot the ROC curve for multiple simulations on the same plot
# Simply calculates the two competing metrics at multiple values of the uncertainty_scale

import pickle
import os
import matplotlib.pyplot as plt

from testbed import calculate_metric_crossval

def calc_roc_curve(uncertainty_scales, combined_results, val_smoothing=0):
    # uncertainty_scales is a list of floats
    # combined_results is a single set of results
    # val_smoothing is passed to the calculator

    all_avg_in_bounds = []
    all_avg_dist_when_in_bounds = []
    for scale in uncertainty_scales:
        avg_in_bounds, avg_dist_when_in_bounds = calculate_metric_crossval(combined_results, uncertainty_scale=scale, val_smoothing=val_smoothing)
        all_avg_in_bounds.append(avg_in_bounds)
        all_avg_dist_when_in_bounds.append(avg_dist_when_in_bounds)

    return all_avg_in_bounds, all_avg_dist_when_in_bounds

def main():
    plot_dir = 'plots_experiment'
    sim_results_dir = 'saved_sims'
    val_smoothing = 0
    uncertainty_scales = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.7, 2]

    os.makedirs(plot_dir, exist_ok=True)
    
    # The runs to be included in the plot, and names to show in the legend
    runnames = ['final_4sensors_baseline', 'final_4sensors_simple']
    plotnames = ['Baseline', 'Ours']

    plt.figure()
    for plotname, runname in zip(plotnames, runnames):
        print(runname)

        # Read the results from the usual file
        with open(f'{sim_results_dir}/{runname}.pkl', 'rb') as f:
            combined_results, prox_times, prox_groups = pickle.load(f)

        # Perform the full calculation and add to the plot
        all_avg_in_bounds, all_avg_dist_when_in_bounds = calc_roc_curve(uncertainty_scales, combined_results, val_smoothing=val_smoothing)
        plt.plot(all_avg_dist_when_in_bounds, all_avg_in_bounds, label=plotname)
        
    # Misc plotting things
    plt.xlabel('DistFromBound')
    plt.ylabel('InBounds')
    plt.legend()
    plt.savefig(f'{plot_dir}/roc_smooth{val_smoothing:0.2f}.pdf')
    plt.savefig(f'{plot_dir}/roc_smooth{val_smoothing:0.2f}.png')

if __name__ == '__main__':
    main()
