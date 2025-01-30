"""Create boxplots for the results of the RL and A* algorithms."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_boxplot_values(data_column):
    """Calculate the boxplot values for the given data column."""
    q1 = np.percentile(data_column, 25)
    q3 = np.percentile(data_column, 75)
    iqr = q3 - q1
    lower_whisker = max(min(data_column), q1 - 1.5 * iqr)
    upper_whisker = min(max(data_column), q3 + 1.5 * iqr)
    outliers = data_column[
        (data_column < lower_whisker) | (data_column > upper_whisker)
    ]
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers,
    }


def process_results(file_path):
    """Process results for both RL and A* algorithms."""
    # Load data
    data = pd.read_csv(file_path)

    # Determine if it is RL or A* results based on columns
    if "total_reward" in data.columns:
        # RL results processing
        print("Processing RL algorithm results...")
        reward_column = "total_reward"
        column_name = "timesteps"

        # Ensure required columns exist
        if column_name not in data.columns or reward_column not in data.columns:
            raise ValueError("Required columns for RL results are missing.")

        # Filter rows where maximum reward is achieved
        max_reward = data[reward_column].max()
        filtered_data = data[data[reward_column] == max_reward]

        # Calculate success rate; ToDo: make it flexible regarding different number of agents
        success_rate = len(filtered_data[filtered_data[reward_column] == 6]) / len(data)

        # Drop NaN values in the relevant column
        data_column = filtered_data[column_name].dropna()
    elif "max_steps" in data.columns:
        # A* results processing
        print("Processing A*/CBS algorithm results...")
        column_name = "max_steps"

        # Exclude the first episode with id 0
        data = data[data["episode"] != 0]
        print("Number of evaluated episodes:", len(data))

        # Calculate success rate
        success_rate = len(data[column_name].dropna()) / len(data)

        # Drop NaN values in the relevant column
        data_column = data[column_name].dropna()
    else:
        raise ValueError(
            "Unknown data format. Ensure the file contains either RL or A* results."
        )

    # Ensure the column has valid numeric data
    if data_column.empty:
        raise ValueError(f"Column '{column_name}' contains no valid data.")
    data_column = data_column.astype(float)

    # Calculate boxplot values
    boxplot_values = calculate_boxplot_values(data_column)

    # Print the calculated statistics
    print(f"    median={np.median(data_column)},")
    print(f"    upper quartile={boxplot_values['Q3']},")
    print(f"    lower quartile={boxplot_values['Q1']},")
    print(f"    upper whisker={boxplot_values['upper_whisker']},")
    print(f"    lower whisker={boxplot_values['lower_whisker']},")
    print("] coordinates {")
    print(
        "    ",
        " ".join(
            f"({i},{val})" for i, val in enumerate(boxplot_values["outliers"], start=1)
        ),
    )

    # Print the success rate
    print(f"Success rate: {success_rate:.2%}")

    # Plot the boxplot
    plt.boxplot(data_column, vert=True)
    plt.title(f"Boxplot for {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Values")
    plt.show()


# Process the provided file with flexibility for both RL and A* results
process_results(r"experiments\results\CBS_12agents_2025-01-29_13-07-37.csv")
