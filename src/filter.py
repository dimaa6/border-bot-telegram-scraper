import numpy as np

def extrapolate_trend_proxy(history_list, anomaly_value=0):
    """
    history_list: List of the last 3-4 valid data points, e.g., [180, 180, 143, 120]
    """
    if len(history_list) < 3:
        return history_list[-1] if history_list else anomaly_value

    # Use the last 4 points to establish the current momentum vector
    recent_trend = history_list[-4:]
    x = np.arange(len(recent_trend))
    y = np.array(recent_trend)

    # Calculate the slope (m) and intercept (c) of the trend line: y = mx + c
    slope, intercept = np.polyfit(x, y, 1)

    # Predict where the NEXT point (index len(recent_trend)) should realistically land
    next_index = len(recent_trend)
    predicted_queue = int(slope * next_index + intercept)

    # Security check: Queue can't be negative. 
    # If the trend was crashing fast, cap the minimum floor at a realistic low value (e.g., 5-10 cars)
    return max(predicted_queue, 10)
