# Change-point detection in gradual-changing trace

In this section, we present a detailed method for generating a gradual-change trace, filtering the data, detecting change-points, and plotting the results. The method also includes an introduction and discussion on the theory behind the approach. To better understand how we detect the step-location of the first derivative of y-data, we have also included a method section on how to determine the step-location.

## Introduction

The purpose of this method is to analyze a gradual-change trace with noise and identify the locations and characteristics of the changes in the data. By applying filtering, first derivative calculation, and change-point detection techniques, we can extract valuable information about the underlying process that generated the trace.

## Method

1. **Generate a gradual-change trace with Gaussian noise:**

   - Define the x and y intervals for the trace, and the noise standard deviation.
   - Use the `gradual_change_trace` function to create a trace with simulated data. This function generates a gradual change in y-values between the given x intervals, adds Gaussian noise to the data, and returns the x and y values of the simulated data.

2. **Estimate the noise level:**

   - Estimate the standard deviation of the Gaussian noise present in the data using the `estimate_noise_std` function. This function calculates the median absolute deviation (MAD) of the differences between consecutive data points and scales it with a factor (e.g., 1.4826) to estimate the standard deviation.

3. **Filter the data using a moving window filter:**

   - Calculate the optimal window size based on the estimated noise standard deviation using the `optimal_window_size` function. This function determines the window size for the moving window filter by adding a base window size to the product of the noise level multiplier and the estimated noise standard deviation.
   - Filter the y-values using the moving window filter function `moving_window_filter`. This function applies a moving average filter to the y-values using a specified window size. It pads the y_values with the window size divided by two on both sides, calculates the mean of the values within the window, and stores the result in an array of filtered y-values.

4. **Calculate the first derivative of the filtered data:**

   - Use the `first_derivative` function to calculate the first derivative of the filtered data with respect to the x_values. This function computes the first-order differences between consecutive filtered y-values and divides them by the differences between the corresponding x_values to obtain the first derivative.

5. **Detect the change-points in the filtered data:**

   - Calculate the optimal scaling factor based on the estimated noise standard deviation using the `optimal_scaling_factor` function. This function determines the scaling factor by adding a base scaling factor to the product of the noise level multiplier and the estimated noise standard deviation.
   - Use the `detect_steps` function to determine the step locations in the filtered data based on the window size and scaling factor. This function applies the Savitzky-Golay filter to the first derivative values and calculates the residuals for each possible step location. It sorts the residuals and finds a threshold based on the estimated noise level. Finally, it selects the step locations that meet the threshold.

6. **Reconstruct the fitted data based on the determined change-points:**

   - Use the `reconstruct_fitted_data` function to create the fitted data based on the filtered data and the detected change-points. This function iterates through the change-points and calculates the linear segments between each pair of change-points. It then fills an array of fitted y-values with the calculated segments.

7. **Plot the results:**

   - Use the 

     ```
     plot_data
     ```

      function to create two subplots:

     - The first subplot shows the original data, filtered data, and the first derivative with a double y-axis.
     - The second subplot shows the original data and the fitted data. The fitted data are linear segments with x and y both from the filtered_y_values.

8. **Export the data and change-points:**

   - Create a DataFrame containing the original data, filtered data, and fitted data, and export it to a CSV file.
   - Calculate the change sizes based on the fitted data, the changing rates, burst durations, and processivity. Create a DataFrame containing this information and export it to a separate CSV file.
   - Plot histograms for changing rates, burst durations, and processivity.

## Discussion

**Advantages:**

1. This method is robust to noise, as it uses a moving window filter to smooth the data and a Savitzky-Golay filter to calculate the first derivative. Both of these filtering techniques help to reduce the impact of noise on the analysis.
2. The method is adaptable to different noise levels and data characteristics, as it uses the estimated noise standard deviation to determine the optimal window size and scaling factor for the filters.
3. The change-point detection algorithm is based on the step locations in the first derivative of the filtered data, which allows for a more accurate identification of change-points compared to methods that only analyze the original data.

**Disadvantages:**

1. The method relies on several user-defined parameters, such as the base window size, noise level multipliers, and scaling factors. These parameters may need to be adjusted for different datasets to obtain optimal results.
2. The moving window filter may not be suitable for all types of data, as it can distort the shape of the original signal, particularly near the edges of the data. Alternative filtering techniques, such as wavelet-based denoising or total variation denoising, may be more appropriate in some cases.
3. The method assumes that the changes in the data are linear between change-points. This assumption may not hold true for all types of data, and more complex models might be required to accurately describe the underlying process.

In conclusion, this method provides a comprehensive approach for analyzing gradual-change traces with noise and detecting change-points in the data. By applying filtering, first derivative calculation, and change-point detection techniques, the method can extract valuable information about the underlying process that generated the trace. While there are some limitations, the method offers a robust and adaptable solution for a wide range of datasets.