# Gradual-Change Trace Analysis

This Python script provides a method for analyzing a gradual-change trace to identify sudden changes in the data. It filters the data, calculates the first derivative, detects change-points, reconstructs the fitted data, and visualizes the results.

## Overview

The method consists of the following steps:

1. Generate a gradual-change trace with simulated data.
2. Filter the data using a moving window filter.
3. Detect the change-points in the filtered data.
4. Calculate the first derivative of the filtered data.
5. Reconstruct the fitted data based on the determined change-points.
6. Plot the results.
7. Export the data to CSV files.
8. Calculate change sizes, changing rates, burst durations, and processivity.

## Methodology

### Gradual-Change Trace Generation

The `gradual_change_trace` function generates a gradual-change trace based on the given intervals for x and y values and a noise standard deviation. The function creates a trace with linear segments and simulates the gradual change by adding Gaussian noise to the y values.

### Data Filtering

The `moving_window_filter` function filters the y values using a moving window of a specified size. The function pads the y values at the beginning and end to maintain the original length of the data. The filtered data is less sensitive to noise and provides a smoother representation of the underlying process.

### Change-Point Detection

The `detect_steps` function detects change-points in the filtered data using a combination of the first derivative and a scaling factor. The first derivative is calculated using the `first_derivative` function, which approximates the derivative using finite differences. The scaling factor is determined using the `optimal_scaling_factor` function, which calculates an appropriate scaling factor based on the estimated noise standard deviation of the y values.

The change-point detection method identifies the locations of sudden changes in the first derivative by comparing the absolute value of the first derivative to a threshold defined by the scaling factor. The identified change-points correspond to the positions of sudden changes in the original data.

### Fitted Data Reconstruction

The `reconstruct_fitted_data` function reconstructs the fitted data based on the filtered data and the detected change-points. The function creates linear segments between consecutive change-points by interpolating the y values. The reconstructed fitted data represents the underlying process, approximating the gradual change with linear segments.

### Data Visualization

The `plot_data` function creates two subplots:

1. The first subplot shows the original data, filtered data, and the first derivative with a double y-axis. This plot illustrates the effect of filtering on the data and highlights the locations of the detected change-points.
2. The second subplot shows the original data and the fitted data, which demonstrates how well the linear segments approximate the gradual change in the trace.

### Data Export

The script exports the original data, filtered data, and fitted data to a CSV file. It also creates a DataFrame with the detected changing-points, changing rates, pausing durations, and change sizes, and exports it to a separate CSV file.

### Change Sizes, Changing Rates, Burst Durations, and Processivity Calculation

The script calculates the change sizes based on the fitted data and detected change-points. It also calculates the changing rates by dividing the change sizes by the corresponding differences in x values, avoiding division by zero. The burst durations are calculated as the differences in x values between consecutive change-points. The script outputs a DataFrame containing the changing points, changing rates, burst durations, and processivity.

## Usage

To use the script, you need to provide simulated data or real data for the gradual-change trace. You may need to adjust the constants in the `optimal_window_size` and `optimal_scaling_factor` functions to better suit your specific dataset. Run the script to generate the gradual-change trace, filter the data, detect the change-points, reconstruct the fitted data, visualize the results, and export the data to CSV files.

## Dependencies

The script requires the following libraries:

- NumPy
- Matplotlib
- Pandas

Make sure you have these libraries installed in your Python environment before running the script.

## Examples

An example of the simulated data is provided in the script:

```
x = [[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1]]
y = [[0, 0.5], [0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 1], [1, 1]]
noise_stddev = 0.02
x_values, y_values = gradual_change_trace(x, y, noise_stddev)
```

You can replace the simulated data with your own dataset and adjust the constants in the `optimal_window_size` and `optimal_scaling_factor` functions as needed.

After running the script, you will obtain the filtered data, first derivative values, and fitted data, as well as the change-points and corresponding changing rates, burst durations, and processivity. The script will also generate plots to visualize the original data, filtered data, first derivative, and fitted data. The data will be exported to CSV files for further analysis.

## Conclusion

This gradual-change trace analysis script provides a comprehensive method for identifying sudden changes in a gradual-change trace by filtering the data, calculating the first derivative, detecting change-points, and reconstructing the fitted data. The script also includes data visualization and export functionalities, making it a valuable tool for analyzing traces with sudden changes in various fields, such as biophysics, finance, and climate research.