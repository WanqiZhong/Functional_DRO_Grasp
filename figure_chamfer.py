import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

result_path = os.path.join(ROOT_DIR, "results_with_predict_q_val.pkl")

def load_results(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

results = load_results(result_path)['results']

chamfer_values = [result['chamfer_value'] for result in results]

plt.figure(figsize=(10, 5))

sns.histplot(chamfer_values, bins=50, kde=True, color='blue', alpha=0.6)
plt.axvline(x=1, color='red', linestyle='--', label='Threshold = 1')
plt.axvline(x=3, color='black', linestyle='--', label='Threshold = 3')

plt.xlabel("Chamfer Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Chamfer Distance")
plt.legend()
# plt.show()
plt.savefig(os.path.join(ROOT_DIR, "chamfer_distribution.png"))