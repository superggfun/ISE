import pandas as pd
import numpy as np
import random

# 1. Load Dataset
data = pd.read_csv("lab3/datasets/7z.csv")  # Replace with your actual dataset path

# 2. Set Sampling Ratio (e.g., 5%)
sample_ratio = 0.05
budget = 100  # Number of search iterations
factor = 1.05  # Histogram update weight factor

# 3. Sample 5% of the Data
sampled_data = data.sample(frac=sample_ratio, random_state=42)

# 4. Find the Best Performing Configuration in the 5% Sample
best_from_sampled = sampled_data.loc[sampled_data['performance'].idxmin()]  # Find the minimum performance
best_sampled_config = best_from_sampled.drop(labels=['performance']).to_dict()
best_sampled_performance = best_from_sampled['performance']

print("Best Configuration from Sampled Data:")
print("Best Sampled Configuration:", best_sampled_config)
print("Best Sampled Performance:", best_sampled_performance)

# 5. Construct Histogram Based on 5% Sampled Data
def build_histogram(data):
    histograms = {}
    for col in data.columns:
        if col != 'performance':
            value_counts = data[col].value_counts(normalize=True)
            histograms[col] = value_counts
    return histograms

histograms = build_histogram(sampled_data)

# 6. Histogram-Based Random Search
def update_histogram(histograms, sampled_config, factor):
    """
    Dynamically update the histogram to favor high-performance areas.
    """
    for col, value in sampled_config.items():
        if value in histograms[col]:
            histograms[col][value] *= factor  # Increase the weight of this value
    for col in histograms:
        total = np.sum(histograms[col])
        histograms[col] = histograms[col] / total  # Normalize the probabilities
    return histograms

def histogram_random_search(data, histograms, budget):
    """
    Perform histogram-based random search.
    """
    best_config = None
    best_performance = float('inf')

    for _ in range(budget):
        # Sample a configuration based on the histogram
        random_config = {
            col: random.choices(
                population=histograms[col].index,
                weights=histograms[col].values,
                k=1
            )[0] for col in histograms
        }

        # Query the dataset for the corresponding performance value
        query = data
        for col, value in random_config.items():
            query = query[query[col] == value]

        if not query.empty:
            performance = query['performance'].iloc[0]
            if performance < best_performance:
                best_performance = performance
                best_config = random_config

        # Dynamically update the histogram
        histograms = update_histogram(histograms, random_config, factor)

    return best_config, best_performance

# 7. Run Histogram-Based Random Search
best_config_hist, best_performance_hist = histogram_random_search(data, histograms, budget)

print("Best Configuration from Histogram Random Search:")
print("Best Histogram-Based Configuration:", best_config_hist)
print("Best Histogram-Based Performance:", best_performance_hist)

# 8. Select the Best Overall Solution (Compare Sampled Data vs. Histogram Search Results)
if best_sampled_performance < best_performance_hist:
    print("The best solution from the 5% sampled data is better. Using the sampled data's best configuration!")
    final_config = best_sampled_config
    final_performance = best_sampled_performance
else:
    print("The histogram-based random search found a better solution. Using this configuration!")
    final_config = best_config_hist
    final_performance = best_performance_hist

# 9. Print the Final Best Result
print("Final Best Configuration:", final_config)
print("Final Best Performance:", final_performance)
