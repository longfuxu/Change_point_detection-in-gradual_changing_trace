import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

# Generate step-like data with Gaussian noise using the provided function
def generate_step_data(n_points, step_locs, step_sizes, noise_std):
    x = np.linspace(0, 1, n_points)
    y = np.zeros(n_points)
    for loc, size in zip(step_locs, step_sizes):
        y += size * (x > loc)
    y += np.random.normal(0, noise_std, n_points)
    return x, y

# Fit a single step to the data
def fit_single_step(data):
    n_points = len(data)

    if n_points <= 5:
        return None, None, np.inf
    
    chi2 = np.zeros(n_points)
    for i in range(1, n_points):
        left_mean = np.mean(data[:i])
        right_mean = np.mean(data[i:])
        chi2[i] = np.sum((data[:i] - left_mean) ** 2) + np.sum((data[i:] - right_mean) ** 2)
    best_loc = np.argmin(chi2[1:-1]) + 1
    step_size = np.mean(data[best_loc:]) - np.mean(data[:best_loc])
    return best_loc, step_size, chi2[best_loc]

#  By prioritizing fitting the longer remaining sides first, 
#  the method would be more balanced in terms of detecting steps in both the right and left sides of the data.
def find_steps(data, max_steps=400):
    step_locs = []
    step_sizes = []
    residuals = []
    remaining_data = np.copy(data)

    data_segments = [(0, len(data))]

    for _ in range(max_steps):
        best_loc = -1
        best_step_size = 0
        best_chi2 = np.inf
        best_segment_index = -1

        # Iterate over the data segments and find the best step for each segment
        for segment_index, (start, end) in enumerate(data_segments):
            segment_data = remaining_data[start:end]
            loc, step_size, chi2 = fit_single_step(segment_data)

            if chi2 < best_chi2:
                best_loc = start + loc
                best_step_size = step_size
                best_chi2 = chi2
                best_segment_index = segment_index

        # Update the data_segments by splitting the best_segment at the best_loc
        start, end = data_segments.pop(best_segment_index)
        data_segments.append((start, best_loc))
        data_segments.append((best_loc, end))
        data_segments.sort(key=lambda x: x[0])

        step_locs.append(best_loc)
        step_sizes.append(best_step_size)
        residuals.append(best_chi2)
        remaining_data[best_loc:] -= best_step_size

    return step_locs, step_sizes, residuals

# def find_optimal_steps(data, step_size_threshold, min_distance=None, step_size_min=None, return_unfiltered=False):
#     """
#     Find the optimal steps that satisfy both the minimum distance and minimum step size conditions.
#     If return_unfiltered is True, also return the step_locations and sorted_residuals for unfiltered steps.
#     """
#     step_locations_filtered = []
#     step_locations_unfiltered = []
#     sorted_residuals_filtered = []
#     sorted_residuals_unfiltered = []

#     for i in range(1, len(data) - 1):
#         step_size = abs(data[i + 1] - data[i - 1])
#         if step_size >= step_size_threshold:
#             step_distance_condition = (min_distance is None) or (len(step_locations_filtered) == 0) or (i - step_locations_filtered[-1] >= min_distance)
#             step_size_condition = (step_size_min is None) or (step_size >= step_size_min)

#             if step_distance_condition and step_size_condition:
#                 step_locations_filtered.append(i)
#                 sorted_residuals_filtered.append(step_size)

#             if return_unfiltered:
#                 if len(step_locations_unfiltered) == 0 or i != step_locations_unfiltered[-1]:
#                     step_locations_unfiltered.append(i)
#                     sorted_residuals_unfiltered.append(step_size)

#     sorted_residuals_filtered.sort(reverse=True)
#     if return_unfiltered:
#         sorted_residuals_unfiltered.sort(reverse=True)
#         return step_locations_filtered, len(step_locations_filtered), sorted_residuals_filtered, step_locations_unfiltered, len(step_locations_unfiltered), sorted_residuals_unfiltered
#     else:
#         return step_locations_filtered, len(step_locations_filtered), sorted_residuals_filtered

def find_optimal_steps(data, step_size_threshold, min_distance=None, step_size_min=None, return_unfiltered=False):
    """
    Find the optimal steps that satisfy both the minimum distance and minimum step size conditions.
    If return_unfiltered is True, also return the step_locations and sorted_residuals for unfiltered steps.
    """
    step_locations_filtered = []
    step_locations_unfiltered = []
    sorted_residuals_filtered = []
    sorted_residuals_unfiltered = []

    for i in range(1, len(data) - 1):
        step_size = abs(data[i + 1] - data[i])  # Modified this line
        if step_size >= step_size_threshold:
            step_distance_condition = (min_distance is None) or (len(step_locations_filtered) == 0) or (i - step_locations_filtered[-1] >= min_distance)
            step_size_condition = (step_size_min is None) or (step_size >= step_size_min)

            if step_distance_condition and step_size_condition:
                step_locations_filtered.append(i)
                sorted_residuals_filtered.append(step_size)

            if return_unfiltered:
                if len(step_locations_unfiltered) == 0 or i != step_locations_unfiltered[-1]:
                    step_locations_unfiltered.append(i)
                    sorted_residuals_unfiltered.append(step_size)

    sorted_residuals_filtered.sort(reverse=True)
    if return_unfiltered:
        sorted_residuals_unfiltered.sort(reverse=True)
        return step_locations_filtered, len(step_locations_filtered), sorted_residuals_filtered, step_locations_unfiltered, len(step_locations_unfiltered), sorted_residuals_unfiltered
    else:
        return step_locations_filtered, len(step_locations_filtered), sorted_residuals_filtered


# Function to recalculate step sizes based on the mean values between adjacent step locations
def recalculate_step_sizes(data, step_locs):
    step_sizes = []
    n_steps = len(step_locs)
    for i in range(n_steps):
        if i == 0:
            left_data = data[:step_locs[i]]
        else:
            left_data = data[step_locs[i-1]:step_locs[i]]
        
        if i == n_steps - 1:
            right_data = data[step_locs[i]:]
        else:
            right_data = data[step_locs[i]:step_locs[i+1]]
        
        step_sizes.append(np.mean(right_data) - np.mean(left_data))
    return step_sizes

# Function to reconstruct the fitted curve
def reconstruct_fitted_curve(x, y, step_locs, step_sizes):
    """
    Reconstruct the fitted curve from the optimal steps.
    """
    reconstructed_data = np.zeros_like(y)
    current_value = y[0]
    current_index = 0

    for loc, size in zip(step_locs, step_sizes):
        # Use the filtered_data for better alignment
        reconstructed_data[current_index:loc] = current_value
        current_value += size
        current_index = loc

    # Fill the rest of the curve
    reconstructed_data[current_index:] = current_value

    return reconstructed_data

# To estimate the noise level of the input data
def estimate_noise_std(data, scaling_factor=1.4826):
    # Calculate the difference between consecutive data points
    diff_data = np.diff(data)
    
    # Calculate the median absolute deviation (MAD) of the difference data
    mad = np.median(np.abs(diff_data - np.median(diff_data)))
    
    # Estimate the standard deviation using the scaling factor
    estimated_std = mad * scaling_factor
    return estimated_std

def estimate_parameters(step_locs, step_sizes, min_distance_fraction=0.3, step_size_min_fraction=0.3):
    # Estimate the min_distance
    step_differences = np.diff(step_locs)
    avg_distance = np.mean(step_differences)
    min_distance = int(min_distance_fraction * avg_distance)

    # Estimate the step_size_min
    avg_step_size = np.mean(step_sizes)
    step_size_min = step_size_min_fraction * avg_step_size

    return min_distance, step_size_min

def detect_steps(x, y, filter_window=5, filter_polyorder=3, scaling_factor=1.1, min_distance=None, step_size_min=None, distance_fraction=0.3, step_size_min_fraction=0.3):
    # Apply the Savitzky-Golay filter
    filtered_data = savgol_filter(y, filter_window, filter_polyorder)

    # Estimate the noise standard deviation
    estimated_noise_std = estimate_noise_std(filtered_data, scaling_factor)

    # Find the optimal steps and step sizes
    _,_,_,optimal_step_locs_unfiltered, _, sorted_residuals_unfiltered = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std,return_unfiltered=True)
    step_size_before_estimation = recalculate_step_sizes(filtered_data,optimal_step_locs_unfiltered)

    # Estimate the min_distance and step_size_min
    # min_distance, step_size_min = estimate_parameters(optimal_step_locs, step_size_before_estimation, distance_fraction, step_size_min_fraction)

    if min_distance is None or step_size_min is None:
        # Estimate the min_distance and step_size_min
        min_distance_est, step_size_min_est = estimate_parameters(optimal_step_locs_unfiltered, step_size_before_estimation, distance_fraction, step_size_min_fraction)
        if min_distance is None:
            min_distance = min_distance_est
        if step_size_min is None:
            step_size_min = step_size_min_est

    optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std, min_distance=min_distance, step_size_min=step_size_min)

    # Recalculate step sizes based on the optimal step locations
    recalculated_step_sizes = recalculate_step_sizes(filtered_data, optimal_step_locs)

    # Reconstruct the fitted curve
    fitted_steps = reconstruct_fitted_curve(x, filtered_data, optimal_step_locs, recalculated_step_sizes)

    # Export original data, filtered data, and fitted data to a CSV file
    data_export = pd.DataFrame({
    "X": x[:len(filtered_data)],
    "Original Data": y[:len(filtered_data)],
    "Filtered Data": filtered_data,
    "Fitted Data": fitted_steps
    })
    data_export.to_csv("/Users/longfu/Desktop/data_export.csv", index=False)

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # axes[0, 0]
    axes[0, 0].plot(x, y, label='Original data')
    axes[0, 0].plot(x, filtered_data, label='Filtered data')
    axes[0, 0].plot(x, fitted_steps, label='Fitted data')
    axes[0, 0].legend()
    axes[0, 0].set_title("Original, Filtered, and Fitted Data")

    # axes[0, 1]
    unfiltered_step_locations, _, unfiltered_residuals = find_optimal_steps(y, step_size_threshold=estimated_noise_std, return_unfiltered=True)[:3]  # Update this line to include unfiltered results
    axes[0, 1].plot(unfiltered_step_locations, unfiltered_residuals, label='Unfiltered residuals')
    axes[0, 1].axhline(y=estimated_noise_std, color='r', linestyle='--', label='Threshold')
    axes[0, 1].legend()
    axes[0, 1].set_title("Residuals vs. Fitting Steps (Unfiltered)")

    # axes[1, 0]
    step_sizes_filtered = [abs(filtered_data[i + 1] - filtered_data[i]) for i in range(len(filtered_data) - 1)]
    axes[1, 0].hist(step_sizes_filtered, bins='auto', alpha=0.7)
    axes[1, 0].axvline(x=step_size_min, color='r', linestyle='--', label='Threshold')
    axes[1, 0].legend()
    axes[1, 0].set_title("Histogram of Step Sizes (Filtered)")

    # axes[1, 1]
    pause_durations_filtered = [optimal_step_locs[i + 1] - optimal_step_locs[i] for i in range(len(optimal_step_locs) - 1)]
    axes[1, 1].hist(pause_durations_filtered, bins='auto', alpha=0.7)
    axes[1, 1].axvline(x=min_distance, color='r', linestyle='--', label='Threshold')
    axes[1, 1].legend()
    axes[1, 1].set_title("Histogram of Pause Durations (Filtered)")

    plt.tight_layout()


"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # [0, 0] Plot the original data, filtered data, and fitted steps
    axes[0, 0].plot(x, y, label="Original Data", linewidth=0.8)
    axes[0, 0].plot(x, filtered_data, label="Filtered Data", linewidth=1)
    axes[0, 0].plot(x, fitted_steps, label="Fitted Steps", linewidth=1.5)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('X-axis label')
    axes[0, 0].set_ylabel('Y-axis label')

    # [0, 1] Plot the quality check
    axes[0, 1].plot(range(len(sorted_residuals_unfiltered)), sorted_residuals_unfiltered, label="Sorted Residuals vs Iteration Steps", linewidth=1, marker='o', markersize=2)
    axes[0, 1].axvline(x=len(optimal_step_locs), color='b', linestyle='--', label="Threshold")
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Iteration Steps')
    axes[0, 1].set_ylabel('Residuals')

    # [1, 0] Histogram distribution of step sizes before filtering
    bins_step_sizes = 80  # Adjust this value to change the bin size of the step sizes histogram
    _, _, _, unfiltered_step_locs, _, _ = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std, min_distance=min_distance, step_size_min=step_size_min, return_unfiltered=True)
    all_step_sizes = recalculate_step_sizes(filtered_data, unfiltered_step_locs)
    axes[1, 0].hist(all_step_sizes, bins=bins_step_sizes, label="Step Sizes Distribution", alpha=0.75)
    if step_size_min is not None:
        axes[1, 0].axvline(x=step_size_min, color='r', linestyle='--', label=f"Min Step Size: {step_size_min}")
        axes[1, 0].axvline(x=-step_size_min, color='r', linestyle='--')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Step Sizes')
    axes[1, 0].set_ylabel('Frequency')

    # [1, 1] Histogram distribution of pause durations (plateau lengths) before filtering
    bins_pause_durations = 80  # Adjust this value to change the bin size of the pause durations histogram
    unfiltered_step_locs, _, _ = find_optimal_steps(filtered_data, step_size_threshold=estimated_noise_std, min_distance=None,step_size_min=None)
    # unfiltered_pause_durations = np.diff(np.concatenate(([0], unfiltered_step_locs, [len(y)])))
    unfiltered_pause_durations = np.diff(unfiltered_step_locs)
    axes[1, 1].hist(unfiltered_pause_durations, bins=bins_pause_durations, label="Pause Durations Distribution", alpha=0.75)
    if min_distance is not None:
        axes[1, 1].axvline(x=min_distance, color='r', linestyle='--', label=f"Min Distance: {min_distance}")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Pause Durations')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("/Users/longfu/Desktop/plot.svg", format='svg', dpi=300, bbox_inches='tight')

"""