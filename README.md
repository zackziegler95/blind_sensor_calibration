# CS 262 Final Project: Estimating Uncertainty in a Low-Cost ParticulateMatter Sensor Network via Rendezvous Consensus
Alex Cabral, Greg Kehne, Zack Ziegler

## Overview

This repository contains the data, code, and plots for the project. There are two datasets included, the second with the suffix `_new`, although we only ended up using the first because it had examples of all the phenomina we were interested in. For completeness, we've included the second dataset as well. Additionally, each dataset includes readings from a single high quality sensor designated as the "ground truth," and 8 smaller and cheaper sensors which make up the distributed network. Initial experiments indicated that the ground truth was not comparable with the other sensors, so in the end we only used data from the distributed sensor network.

`correct_timestamps(_new).py` - data preprocessing (see below)
`analyze(_new).py` - initial plots exploring the data
`testbed.py` - main simulation, algorithm and plotting code
`plot_roc_curves.py` - plotting roc curves from multiple simulation runs on a single plot

## Data processing

The raw data as output by the sensors is stored in `data`. There appears to be an issue with the sensor data logging code as it doesn't indicate AM or PM, so as a first step we run `correct_timestamps(_new).py` to correct the timesteps. The output is stored in a new directory called `data_tscorrect`.

Additionally, we initially explore the data with `analyze(_new).py`. Some care is needed here to align the data properly because the two types of sensors report the data in a different format, with different units, and different time zones.

## testbed.py

### Simulator design

The main chunk of working code is in `testbed.py`. We decided to frame the uncertainty estimation problem as a simulation of time, during the course of which sensor readings appear. That is, instead of looping through sensor reading events (which happen every 20 seconds for each sensor but do not happen at exactly the same time for the different sensors), we step through time at 1 second intervals. At each interval, we see for each sensor if there is a new reading available and if so we update the current value for that sensor. At each interval the simulation needs to report an uncertainty for each sensor, and this is what is decided by the algorithm. We decided to frame the problem this way instead of an event-centric way so that we wouldn't need to worry about the fact that sensors report readings at different times, and because originally we were imagining there might be an algorithm which changes uncertainty over time, even without any sensor readings being recorded.

The `Simulator` class, acting as a true simulator, is stateful. It maintains a set of current sensor readings and current uncertainties, to mimic the real-world use case where one would want to query at any given point the value and uncertainty of each sensor.

To simulate sensors moving around to be in contact with different sets of neighbors at different times, we introduce a notion of proximity events. The first proximity event has to start at time 0, and any number of events can occur after that. Each event is defined as a set of lists, where each list contains a grouping of sensors that are in close proximity to each other. Each sensor may only appear in one neighborhood, and each sensor must appear in at least one neighborhood. The proximity event files are stored in `proximity_events`. The case where all sensors are in a single neighborhood throughout the simulation is `proximity_events/4_constant.csv`, and an example case where the sensors move around is `proximity_events/4_occasional_drop.csv`. 

At the end of a simulation, the `results` object containing the set of observation times and the values and uncertainties at each time is saved in `saved_sims` for future analysis. A plot of the simulation run is also generated, showing the values and confidences at each point for all sensors.

### Algorithms

Three algorithms are implemented:

`baseline` - The factory default baseline that simply reports the uncertainty according to the sensor documentation.
`complex` - An initial pass at our algorithm, that uses a more complicated update rule than what we ended up with. This is not reported in the paper
`simple` - The algorithm we ended up discussing in the paper. It uses a simpler update rule than `comlex`.

## ROC curve creation

The ROC curves are generated using `plot_roc_curves.py`, based on saved data from multiple simulations.
