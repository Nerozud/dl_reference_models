import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_boxplot_values(data_column):
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


def main():
    file_path = r"experiments\results\ReferenceModel-2-1_PPO_2024-10-30_15-55-55.csv"  # Update with your CSV file path
    data = pd.read_csv(file_path)
    column_name = "timesteps"  # Update with your column name
    reward_column = "total_reward"  # Update with your reward column name

    # Filter data where the maximum reward is reached
    max_reward = data[reward_column].max()
    filtered_data = data[data[reward_column] == max_reward]

    data_column = filtered_data[column_name].dropna()
    boxplot_values = calculate_boxplot_values(data_column)

    print(f"    median={np.median(data_column)},")
    print(f"    upper quartile={boxplot_values['Q3']},")
    print(f"    lower quartile={boxplot_values['Q1']},")
    print(f"    upper whisker={boxplot_values['upper_whisker']},")
    print(f"    lower whisker={boxplot_values['lower_whisker']},")
    print("] coordinates {{")
    print(
        "    ",
        " ".join(
            f"({i},{val})" for i, val in enumerate(boxplot_values["outliers"], start=1)
        ),
    )

    plt.boxplot(data_column, vert=True)
    plt.title(f"Boxplot for {column_name}")
    plt.show()


if __name__ == "__main__":
    main()
