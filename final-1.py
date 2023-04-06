import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from step_detection import detect_steps

def gradual_change_trace(x, y, noise_stddev):
    """
    Generates a gradual changing trace with pausing events.
    x: A list of lists, each sublist contains the start and end x-coordinate of the segment.
    y: A list of lists, each sublist contains the start and end y-coordinate of the segment.
    noise_stddev: Standard deviation of the Gaussian noise added to the y values.
    """
    x_values = []
    y_values = []

    for i, (x_start, x_end) in enumerate(x):
        y_start, y_end = y[i]
        num_points = int((x_end - x_start) * 1000)
        segment_x = np.linspace(x_start, x_end, num_points)
        segment_y = np.linspace(y_start, y_end, num_points)
        noise = np.random.normal(0, noise_stddev, num_points)
        x_values.extend(segment_x)
        y_values.extend(segment_y + noise)

    return np.array(x_values), np.array(y_values)

# using the np.pad function to pad the data before applying the moving window filter. 
# This way, the window size remains the same throughout the filtering process, 
# but we avoid extreme values by padding the data.
def moving_window_filter(y_values, window_size):
    padding_size = window_size // 2
    padded_y_values = np.pad(y_values, (padding_size, padding_size), mode='edge')
    filtered_y_values = np.zeros(len(y_values))

    for i in range(len(y_values)):
        left = i
        right = i + window_size
        filtered_y_values[i] = np.mean(padded_y_values[left:right])

    return filtered_y_values

# Function to calculate the first derivative while handing the edge effect
def first_derivative(x, y, window_size):
    first_derivative_values = np.zeros(len(y))
    
    for i in range(len(y)):
        if i < window_size:
            dy = y[i + window_size] - y[0]
            dx = x[i + window_size] - x[0]
        elif i > len(y) - window_size - 1:
            dy = y[-1] - y[i - window_size]
            dx = x[-1] - x[i - window_size]
        else:
            dy = y[i + window_size] - y[i - window_size]
            dx = x[i + window_size] - x[i - window_size]
        
        first_derivative_values[i] = dy / dx
    
    return first_derivative_values

# To estimate the noise level of the input data
def estimate_noise_std(data, scaling_factor=1.4826):
    # Calculate the difference between consecutive data points
    diff_data = np.diff(data)
    
    # Calculate the median absolute deviation (MAD) of the difference data
    mad = np.median(np.abs(diff_data - np.median(diff_data)))
    
    # Estimate the standard deviation using the scaling factor
    estimated_std = mad * scaling_factor
    return estimated_std

# Based on simulated data, we determine the optimal window size with a base_window_size and a noise_level_muliplier
def optimal_window_size(estimated_noise_y_value):
    # You can adjust the constants to better suit your specific dataset
    # These constants can be determined experimentally
    base_window_size = 21
    noise_level_multiplier = 171

    window_size = base_window_size + int(noise_level_multiplier * estimated_noise_y_value)
    
    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    return window_size

# # Based on simulated data, we determine the optimal scaling_factor with a base_scaling_factor and a noise_level_muliplier
# def optimal_scaling_factor(estimated_noise_y_value):
#     # You can adjust the constants to better suit your specific dataset
#     # These constants can be determined experimentally
#     base_scaling_factor = 0.8655
#     noise_level_multiplier = 2.7

#     scaling_factor = base_scaling_factor + noise_level_multiplier * estimated_noise_y_value
#     scaling_factor = round(scaling_factor,4)

#     return scaling_factor

# Main function to obtain the fitted y-data while properly handle the edge of the data
def detect_changing_point(x_values, y_values, filter_window_sg=5, filter_polyorder_sg=3,scaling_factor=0.9,distance_fraction=0.3,step_size_min_fraction=0.3):
    # Calculate the optimal window size and scaling factor based on the estimated noise standard deviation
    window_size = optimal_window_size(estimate_noise_std(y_values, scaling_factor=1.3))
    # scaling_factor = optimal_scaling_factor(estimate_noise_std(y_values, scaling_factor=0.9))

    # filter the y_values and determine the first derivative
    filtered_y_values = moving_window_filter(y_values, window_size)
    first_derivative_values = first_derivative(x_values, filtered_y_values, window_size)

    # detect the step_loc of first derivative of y_values, by calling the Detect_Steps function
    x_values, fitted_steps_first_derivative,optimal_step_locs, sorted_residuals = detect_steps(x_values, first_derivative_values,filter_window=filter_window_sg, filter_polyorder=filter_polyorder_sg, scaling_factor=scaling_factor, distance_fraction=distance_fraction,step_size_min_fraction=step_size_min_fraction)
    
    fitted_y_values = np.zeros_like(filtered_y_values)
    sorted_step_locs = sorted(optimal_step_locs)
    n_step_locs = len(sorted_step_locs)

    for i in range(n_step_locs + 1):
        if i == 0:
            x_start, y_start = x_values[0], filtered_y_values[0]
            x_end, y_end = x_values[sorted_step_locs[i]], filtered_y_values[sorted_step_locs[i]]
            segment_length = sorted_step_locs[i]
        elif i == n_step_locs:
            x_start, y_start = x_values[sorted_step_locs[i - 1]], filtered_y_values[sorted_step_locs[i - 1]]
            x_end, y_end = x_values[-1], filtered_y_values[-1]
            segment_length = len(filtered_y_values) - sorted_step_locs[i - 1]
        else:
            x_start, y_start = x_values[sorted_step_locs[i - 1]], filtered_y_values[sorted_step_locs[i - 1]]
            x_end, y_end = x_values[sorted_step_locs[i]], filtered_y_values[sorted_step_locs[i]]
            segment_length = sorted_step_locs[i] - sorted_step_locs[i - 1]
        
        segment_y = np.linspace(y_start, y_end, segment_length)
        fitted_y_values[sorted_step_locs[i - 1] if i > 0 else 0:sorted_step_locs[i] if i < n_step_locs else len(filtered_y_values)] = segment_y
            
    return x_values,filtered_y_values,first_derivative_values,optimal_step_locs,fitted_y_values,fitted_steps_first_derivative

# Plot all the results out
def plot_data(x_values, y_values, filtered_y_values, first_derivative_values, fitted_y_values,fitted_steps_first_derivative):
    """
    The first subplot shows the original data, filtered data, and the first derivative with a double y-axis.
    The second subplot shows the original data and the fitted data. 
    The fitted data are linear segments with x and y both from the filtered_y_values.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # First plot: original data, filtered data, and first derivative with double y-axis
    ax1.plot(x_values, y_values, label='Original Data')
    ax1.plot(x_values[:len(filtered_y_values)], filtered_y_values, label='Filtered Data')
    ax1.set_ylabel('Y Values')
    ax1.legend(loc='upper left')
    
    ax1b = ax1.twinx()
    ax1b.plot(x_values[:len(first_derivative_values)], first_derivative_values, color='g', label='First Derivative')
    ax1b.plot(x_values[:len(fitted_steps_first_derivative)], fitted_steps_first_derivative, color='r', label='First Derivative-steps')
    ax1b.set_ylabel('First Derivative')
    ax1b.legend(loc='upper right')

    # Second plot: original data and fitted data
    ax2.plot(x_values, y_values, label='Original Data')
    ax2.plot(x_values[:len(fitted_y_values)], fitted_y_values, label='Fitted Data')
    ax2.set_ylabel('Y Values')
    ax2.legend()


# Example of the simulated data
x = [[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1]]
y = [[0, 0.5], [0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 1], [1, 1]]
noise_stddev = 0.03
x_values, y_values = gradual_change_trace(x, y, noise_stddev)

# calling the main function
x_values,filtered_y_values,first_derivative_values,optimal_step_locs,fitted_y_values,fitted_steps_first_derivative = detect_changing_point(x_values, y_values, filter_window_sg=5, filter_polyorder_sg=3,scaling_factor=1.2,distance_fraction=0.6,step_size_min_fraction=0.6)

# Plot all the data out
plot_data(x_values, y_values, filtered_y_values, first_derivative_values, fitted_y_values,fitted_steps_first_derivative)
plt.show()

"""
# Export original data, filtered data, and fitted data to a CSV file
data_export = pd.DataFrame({
    "X": x_values[:len(filtered_y_values)],
    "Original Data": y_values[:len(filtered_y_values)],
    "Filtered Data": filtered_y_values,
    "Fitted Data": fitted_y_values
})
data_export.to_csv("data_export.csv", index=False)

# Calculate the change sizes based on fitted data
fitted_change_sizes = np.diff(fitted_y_values[np.array(optimal_step_locs_first_dirivative)])
# calculate the fitted changing_rates while avoiding divided by zero
x_diffs = np.diff(x_values[optimal_step_locs_first_dirivative])
non_zero_diffs_mask = x_diffs != 0
fitted_changing_rates = np.zeros_like(x_diffs, dtype=float)
fitted_changing_rates[non_zero_diffs_mask] = fitted_change_sizes[non_zero_diffs_mask] / x_diffs[non_zero_diffs_mask]
# Calculate the burst durations
burst_durations = np.diff(x_values[optimal_step_locs_first_dirivative])

# Create a DataFrame with detected changing-points, changing rates, pausing durations, and change sizes
change_points_export = pd.DataFrame({
    "Changing Point": x_values[optimal_step_locs_first_dirivative[:-1]],
    "Changing Rate": fitted_changing_rates,
    "Burst Duration(in time)": burst_durations,
    "Processivity": fitted_change_sizes,
})

# Export change points data to a separate CSV file
change_points_export.to_csv("change_points_export.csv", index=False)

# Plot the distributions of Changing Rates, Burst Durations, and Processivity
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
fig.suptitle("Distributions of Changing Rates, Burst Durations, and Processivity")

# Plot histograms for changing rates, burst durations, and processivity
hist_params = {
    'bins': 30,
    'density': True,
    'alpha': 0.6,
    'color': 'b'
}

# When plotting, only use the non-zero differences
axes[0, 0].hist(fitted_changing_rates[non_zero_diffs_mask], **hist_params)
axes[0, 0].set_title("Histogram of Changing Rates")
axes[0, 0].set_xlabel("Changing Rate")
axes[0, 0].set_ylabel("Density")

axes[0, 1].hist(burst_durations, **hist_params)
axes[0, 1].set_title("Histogram of Burst Durations")
axes[0, 1].set_xlabel("Burst Duration")
axes[0, 1].set_ylabel("Density")

axes[0, 2].hist(fitted_change_sizes, **hist_params)
axes[0, 2].set_title("Histogram of Processivity")
axes[0, 2].set_xlabel("Processivity")
axes[0, 2].set_ylabel("Density")

# Plot box plots for changing rates, burst durations, and processivity
box_data = [fitted_changing_rates, burst_durations, fitted_change_sizes]
box_labels = ["Changing Rate", "Burst Duration", "Processivity"]

axes[1, 1].boxplot(box_data, labels=box_labels)
axes[1, 1].set_title("Box Plots of Changing Rates, Burst Durations, and Processivity")
axes[1, 1].set_ylabel("Value")

# Hide unused subplots
axes[1, 0].axis('off')
axes[1, 2].axis('off')

# Show the plots
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Print the values 
print("The estimated noise standard deviation of y_value is {:.2f}".format(estimate_noise_std(y_values, scaling_factor=1.3)))
print("The optimal window size is {}".format(window_size))
print("The optimal scaling factor is {}".format(scaling_factor))
"""



"""
# Main function to detect the change-point of the gradual-changing trace
def detect_steps(x_values, y_values, window_size, scaling_factor):
    filtered_y_values = moving_window_filter(y_values, window_size)
    first_derivative_values = first_derivative(x_values, filtered_y_values, window_size)

    filtered_first_derivative = savgol_filter(first_derivative_values, window_length=5, polyorder=3)
    estimated_noise_std = estimate_noise_std(filtered_first_derivative, scaling_factor=scaling_factor)

    optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_first_derivative, step_size_threshold=estimated_noise_std,min_distance=20)
    recalculated_step_sizes = recalculate_step_sizes(filtered_first_derivative, optimal_step_locs)
    fitted_steps = reconstruct_fitted_steps(x, filtered_first_derivative, optimal_step_locs, recalculated_step_sizes)

    # detect step again on the fitted steps of the first derivative, so as to decrease the artifical small steps
    optimal_step_locs_first_dirivative, _, _ = find_optimal_steps(fitted_steps, step_size_threshold=0.001)
    recalculated_step_sizes_first_dirivative = recalculate_step_sizes(fitted_steps, optimal_step_locs_first_dirivative)
    fitted_steps_first_derivative = reconstruct_fitted_steps(x, fitted_steps, optimal_step_locs_first_dirivative, recalculated_step_sizes_first_dirivative)
    return optimal_step_locs_first_dirivative,fitted_steps,fitted_steps_first_derivative
"""