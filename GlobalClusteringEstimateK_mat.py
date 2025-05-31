import pandas as pd
import numpy as np
import re
import time
import warnings
import traceback
import logging
import os 
import matplotlib.pyplot as plt 
import seaborn as sns          
from umap import UMAP
import plotly.io as pio
import hdbscan
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering,
    AgglomerativeClustering,OPTICS, Birch
)    #removed DBSCAN

# import hdbscan 
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score 
)
from tqdm.auto import tqdm 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

LABEL_COLUMN = 'status' 
FEATURE_PATTERN = r'.*-\d+(\.\d+)?$' 
FINAL_RESULTS_CSV = "clustering_results_natural_clusters.csv"
CONFUSION_MATRIX_DIR = "confusion_matrices_natural"
CREATE_PLOTS = True 

ESTIMATE_K_FOR = ['KMeans','SpectralClustering','AgglomerativeClustering', 'BIRCH']

RANDOM_STATE = 42 
KMEANS_N_INIT = 10
DBSCAN_EPS = 0.5 
DBSCAN_MIN_SAMPLES = 5 

SPECTRAL_AFFINITY = 'nearest_neighbors'
SPECTRAL_ASSIGN_LABELS = 'kmeans'
SPECTRAL_N_NEIGHBORS = 10
AGGLOMERATIVE_LINKAGE = 'ward'

HDBSCAN_MIN_CLUSTER_SIZE = 5
OPTICS_MIN_SAMPLES = 5
BIRCH_THRESHOLD = 0.5
BIRCH_BRANCHING_FACTOR = 50
AFFINITY_PROPAGATION_DAMPING = 0.5

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
ALGORITHMS_TO_TEST = [
    {
        'name': 'KMeans',
        'model': KMeans,
        'params': {'n_init': KMEANS_N_INIT, 'random_state': RANDOM_STATE},
        'needs_n_clusters': True
    },
    {
        'name': 'AffinityPropagation',
        'model': AffinityPropagation,
        'params': {'damping': AFFINITY_PROPAGATION_DAMPING, 'random_state': RANDOM_STATE, 'max_iter': 300, 'convergence_iter': 25},
        'needs_n_clusters': False
    },
    {
        'name': 'MeanShift',
        'model': MeanShift,
        'params': {'bandwidth': None, 'bin_seeding': True, 'cluster_all': False, 'n_jobs': 1},
        'needs_n_clusters': False
    },
    {
        'name': 'SpectralClustering',
        'model': SpectralClustering,
        'params': {
            'affinity': SPECTRAL_AFFINITY,
            'assign_labels': SPECTRAL_ASSIGN_LABELS,
            'random_state': RANDOM_STATE,
        },
        'needs_n_clusters': True,
    },
    {
        'name': 'AgglomerativeClustering',
        'model': AgglomerativeClustering,
        'params': {'linkage': AGGLOMERATIVE_LINKAGE},
        'needs_n_clusters': True
    },
    {
        'name': 'HDBSCAN',
        'model': hdbscan.HDBSCAN,
        'params': {
            'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
            'min_samples': None,
            'allow_single_cluster': True,
            'gen_min_span_tree': False,
            'core_dist_n_jobs': 1
        },
        'needs_n_clusters': False
    },
    {
        'name': 'OPTICS',
        'model': OPTICS,
        'params': {'min_samples': OPTICS_MIN_SAMPLES, 'cluster_method': 'xi', 'n_jobs': 1},
        'needs_n_clusters': False
    },
    {
        'name': 'BIRCH',
        'model': Birch,
        'params': {'threshold': BIRCH_THRESHOLD, 'branching_factor': BIRCH_BRANCHING_FACTOR},
        'needs_n_clusters': True
    }
]



def plot_status_distribution(labels, status_column, output_path="plots/status_distribution.png"):
   
    df = pd.DataFrame({
        'cluster': labels,
        'status': status_column
    })

    status_distribution = df.groupby(['cluster', 'status']).size().unstack(fill_value=0)

    ax = status_distribution.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Status Distribution within Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Projects')
    plt.xticks(rotation=0)
    plt.legend(title='Status')
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150)
        print(f"Status distribution plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving status distribution plot: {e}")
    
    return 


def plot_umap_2d_best_cluster(data, pred, status, algorithm_name, output_path="plots/umap_output"):
   
    umap_2d = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    proj_2d = umap_2d.fit_transform(data)

    df = pd.DataFrame({
        'x': proj_2d[:, 0],
        'y': proj_2d[:, 1],
        'cluster': pred,
        'status': status.astype(str)
    })

    fig = px.scatter(
        df, x='x', y='y',
        color=df['cluster'].astype(str),
        symbol='status',
        labels={'color': 'Cluster', 'symbol': 'Status'}
    )

    fig.update_layout(
        title=f"UMAP 2D Visualization - {algorithm_name}",
        xaxis_title="x",
        yaxis_title="y",
        legend_title="Cluster / Status"
    )

    try:
        pio.write_image(fig, output_path + ".png", format="png", scale=2)
        print(f"[Saved] UMAP PNG to: {output_path}")
    except Exception as e:
        print(f"[Error] Failed to save UMAP PNG: {e}\n")
    return 

def plot_and_save_confusion_matrix(y_true, y_pred, labels_true_map, algo_name, output_dir="plots/confusion", estimated_k=None):
    non_noise_mask = (y_pred != -1)
    y_true_eff = np.array(y_true)[non_noise_mask]
    y_pred_eff = np.array(y_pred)[non_noise_mask]

    if len(y_true_eff) == 0:
        print(f"Warning: No non-noise points found for {algo_name}. Skipping confusion matrix.")
        return

    # encode y_true to match numeric type of y_pred
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true_eff)
    label_names_encoded = le.classes_

    unique_pred_labels = np.unique(y_pred_eff)
    if len(unique_pred_labels) < 1:
        print(f"Warning: Only noise predicted for {algo_name}. Skipping.")
        return

    cluster_labels_str = [f"Cluster {i}" for i in unique_pred_labels]
    cm = confusion_matrix(y_true_encoded, y_pred_eff)

    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(6, len(label_names_encoded)*0.8), max(5, len(unique_pred_labels)*0.6)))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cluster_labels_str,
                yticklabels=label_names_encoded,
                ax=ax, cbar=True, linewidths=.5)

    title = f'Confusion Matrix: {algo_name}'
    if estimated_k is not None and not pd.isna(estimated_k):
        title += f' (Estimated k={int(estimated_k)})'
    title += '\n(True Labels vs Predicted Clusters)'

    ax.set_title(title, fontsize=12)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Cluster', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"confusion_matrix_{algo_name}.png")
    try:
        plt.savefig(filename, dpi=150)
    except Exception as e_save:
        print(f"Error saving confusion matrix for {algo_name}: {e_save}")
    finally:
        plt.close(fig) 

def save_clustering_results_to_csv(results, output_path="plots/clustering_evaluation_results.csv"):
    clean_results = []
    for res in results:
        clean_results.append({
            'algorithm': res['algorithm'],
            'n_clusters': res['n_clusters'],
            'silhouette': res['silhouette'],
            'adjusted_rand': res['adjusted_rand'],
            'calinski_harabasz': res['calinski_harabasz'],
            'davies_bouldin': res['davies_bouldin']

        })

    df = pd.DataFrame(clean_results)
    df.to_csv(output_path, index=False)
    print(f"Clustering results saved to {output_path}")
    return 


def overall_feature_importance(X, y_clusters, features, create_plot=True, output_path="plots/overall_feature_importance.png"):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_clusters, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    overall_importance = rf.feature_importances_
    accuracy = accuracy_score(y_test, rf_pred)

    if create_plot:
        plt.figure(figsize=(20, 5))
        sns.barplot(x=features, y=overall_importance, palette="viridis")
        plt.xlabel("Feature")
        plt.ylabel("Overall Importance Score")
        plt.title(f"Overall Feature Importance (Random Forest) -- {accuracy:.2f} accuracy")
        plt.tight_layout()

        try:
            plt.savefig(output_path, dpi=150)
            print(f"Feature importance plot saved to {output_path}")
        except Exception as e:
            print(f"Failed to save feature importance plot: {e}")
    return

def evaluate_clustering_algorithms(X, algorithms_to_test, cluster_range, true_pred, random_state=42):
   
    best_score = -1
    best_result = None
    results = []

    for algo in algorithms_to_test:
        name = algo['name']
        model_cls = algo['model']
        params = algo['params'].copy()
        needs_n_clusters = algo.get('needs_n_clusters', False)

        k_values = cluster_range if needs_n_clusters else [None]
        
        for k in k_values:
            try:
                if k is not None:
                    params['n_clusters'] = k

                model = model_cls(**params)
                y_pred = model.fit_predict(X)

                silhouette = silhouette_score(X, y_pred)
                db_score = davies_bouldin_score(X, y_pred)
                ch_score = calinski_harabasz_score(X, y_pred)
                ari_score = adjusted_rand_score(true_pred, y_pred)


                results.append({
                    'algorithm': name,
                    'n_clusters': k,
                    'silhouette': silhouette,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score,
                    'adjusted_rand': ari_score,
                    'num_noise_points': (y_pred == -1).sum(),
                    'labels': y_pred  
                })

                if silhouette > best_score:
                    best_score = silhouette
                   

            except Exception as e:
                logging.error(f"Exception running {name} (k={k}): {str(e)}")
                traceback.print_exc()
                continue


    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['silhouette'].idxmax()]
    print(f"Best clustering: {best_result['algorithm']} (k={best_result['n_clusters']}) with Silhouette Score: {best_result['silhouette']}, ARI score:{best_result['adjusted_rand']}, davies_bouldin:{best_result['davies_bouldin']},calinski_harabasz:{best_result['calinski_harabasz']} ")
    return best_result, results

df_all = pd.read_csv("data.csv")
#retired_graduated_bypassed = df_all.loc[df_all['status'] != 'evolved']

#scaler = PowerTransformer(method='yeo-johnson')
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MinMaxScaler()
df_numeric = df_all.select_dtypes(include=[np.number])
df_norm = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

status = df_all['status']

CLUSTER_RANGE = range(2,15)

communication = ['COM-1', 'COM-2']
popularity = ['POP-1']
stability = ['STA-1','STA-2', 'STA-3', 'STA-4', 'STA-5', 'STA-6', 'STA-7', 'STA-8', 'STA-9']
technical_activity = ['TEC-1', 'TEC-2', 'TEC-3', 'TEC-4']
quality = ['SWQ-1', 'SWQ-2.1', 'SWQ-2.2', 'SWQ-2.3', 'SWQ-2.4', 'SWQ-2.5', 'SWQ-2.6', 'SWQ-2.7']

features = quality + stability + technical_activity + communication + popularity
label_names = ['graduated', 'retired', 'bypassed', 'evolved']

best_result, all_results = evaluate_clustering_algorithms(df_norm, ALGORITHMS_TO_TEST, CLUSTER_RANGE, status)
save_clustering_results_to_csv(all_results)
plot_status_distribution(best_result['labels'], status)
overall_feature_importance(df_norm, best_result['labels'], features, create_plot=True)


plot_and_save_confusion_matrix(status, best_result['labels'], label_names, best_result['algorithm'], estimated_k=best_result['n_clusters'])

plot_umap_2d_best_cluster(
    data=df_norm,
    pred=best_result['labels'],
    status=status,
    algorithm_name=best_result['algorithm']
)



