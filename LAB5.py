import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


df = pd.read_excel("ecg_eeg_features.csv.xlsx")  

# Encode target for regression (numerical mapping of ECG)
df["target_numeric"] = df["signal_type"].map({"ECG": 0, "EEG": 1})

# Features (X) and target (y)
X = df.drop(columns=["signal_type", "target_numeric"])
y = df["target_numeric"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  A1: Linear Regression (1 attribute) 
def linear_regression_one_feature(X_train, y_train, X_test, y_test, feature):
    model = LinearRegression().fit(X_train[[feature]], y_train)
    y_train_pred = model.predict(X_train[[feature]])
    y_test_pred = model.predict(X_test[[feature]])
    return model, y_train_pred, y_test_pred

# A2: Metrics 
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

#  A3: Multiple Features 
def linear_regression_all_features(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return model, y_train_pred, y_test_pred

#  A4: K-Means Clustering
def perform_kmeans(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
    return kmeans

#  A5: Clustering Metrics 
def clustering_scores(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels)
    }

# A6: Different k values 
def kmeans_evaluation(X_train, k_values):
    scores_dict = {}
    for k in k_values:
        kmeans = perform_kmeans(X_train, k)
        scores_dict[k] = clustering_scores(X_train, kmeans.labels_)
    return scores_dict

# A7: Elbow Plot (optimized for speed) 
def elbow_method(X_train, k_range, sample_size=1000):
    # Reduce dataset size for faster computation
    if len(X_train) > sample_size:
        X_sample = X_train.sample(sample_size, random_state=42)
    else:
        X_sample = X_train.copy()
    
    distortions = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sample)
        distortions.append(kmeans.inertia_)
    plt.plot(k_range, distortions, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Method")
    plt.show()


if _name_ == "_main_":
    # A1 & A2
    feature = "mean_val"
    model_one, y_train_pred1, y_test_pred1 = linear_regression_one_feature(X_train, y_train, X_test, y_test, feature)
    metrics_train1 = calculate_metrics(y_train, y_train_pred1)
    metrics_test1 = calculate_metrics(y_test, y_test_pred1)
    print("\nA1 & A2 - One Feature Metrics:")
    print("Train:", metrics_train1)
    print("Test:", metrics_test1)

    # A3
    model_all, y_train_pred_all, y_test_pred_all = linear_regression_all_features(X_train, y_train, X_test, y_test)
    metrics_train_all = calculate_metrics(y_train, y_train_pred_all)
    metrics_test_all = calculate_metrics(y_test, y_test_pred_all)
    print("\nA3 - All Features Metrics:")
    print("Train:", metrics_train_all)
    print("Test:", metrics_test_all)

    # A4 & A5
    kmeans2 = perform_kmeans(X_train, 2)
    cluster_metrics = clustering_scores(X_train, kmeans2.labels_)
    print("\nA4 & A5 - Clustering Metrics (k=2):", cluster_metrics)

    # A6
    k_values = range(2, 6)
    k_scores = kmeans_evaluation(X_train, k_values)
    print("\nA6 - KMeans Scores for different k:")
    for k, scores in k_scores.items():
        print(f"k={k}:", scores)

    # A7
    elbow_method(X_train, range(2, 10))
