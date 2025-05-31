import pandas as pd
import numpy as np
import re
import itertools
import time
import os
import concurrent.futures
import math

from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering,
    AgglomerativeClustering, DBSCAN, OPTICS, Birch
)
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
from tqdm.auto import tqdm 
from collections import defaultdict

DATA_FILE = 'data.csv'
LABEL_COLUMN = 'status'
MAX_FEATURES_TO_COMBINE = 3
PRIMARY_METRIC = 'ari'
TOP_N_RESULTS = 5
N_CLUSTERS = 2
NUMBER_OF_CHUNKS = 1
FINAL_TOP_N_CSV_FILENAME = "Final_Top_N_Results.csv"

RANDOM_STATE = 42

ALGORITHMS_TO_TEST = [
    {'name': 'KMeans', 'model': KMeans, 'params': {'random_state': RANDOM_STATE}, 'needs_n_clusters': True},
    {'name': 'AffinityPropagation', 'model': AffinityPropagation, 'params': {'random_state': RANDOM_STATE}, 'needs_n_clusters': False},
    {'name': 'MeanShift', 'model': MeanShift, 'params': {}, 'needs_n_clusters': False},
    {'name': 'SpectralClustering', 'model': SpectralClustering, 'params': {'assign_labels': 'kmeans', 'random_state': RANDOM_STATE, 'affinity': 'nearest_neighbors', 'n_init': 10}, 'needs_n_clusters': True},
    {'name': 'AgglomerativeClustering', 'model': AgglomerativeClustering, 'params': {}, 'needs_n_clusters': True},
    {'name': 'DBSCAN', 'model': DBSCAN, 'params': {}, 'needs_n_clusters': False},
    {'name': 'OPTICS', 'model': OPTICS, 'params': {}, 'needs_n_clusters': False},
    {'name': 'BIRCH', 'model': Birch, 'params': {}, 'needs_n_clusters': True},
    {'name': 'HDBSCAN', 'model': hdbscan.HDBSCAN, 'params': { }, 'needs_n_clusters': False}
]

worker_prefiltered_data_store = {}

def init_worker(data_for_workers):
    global worker_prefiltered_data_store
    worker_prefiltered_data_store = data_for_workers

def calculate_clustering_metrics(y_true, y_pred_original_labels):
    y_pred = np.array(y_pred_original_labels)
    non_noise_mask = (y_pred != -1)
    y_true_eff = y_true[non_noise_mask]
    y_pred_eff = y_pred[non_noise_mask]

    num_clusters_found_total = len(np.unique(y_pred))
    num_effective_clusters = 0
    ari, nmi, vm = np.nan, np.nan, np.nan

    if len(y_pred_eff) < 2:
        return ari, nmi, vm, num_clusters_found_total, num_effective_clusters

    unique_pred_eff_labels = np.unique(y_pred_eff)
    num_effective_clusters = len(unique_pred_eff_labels)

    if num_effective_clusters == 0 : 
        return ari, nmi, vm, num_clusters_found_total, num_effective_clusters
    
    try: ari = adjusted_rand_score(y_true_eff, y_pred_eff)
    except ValueError: pass
    try: nmi = normalized_mutual_info_score(y_true_eff, y_pred_eff, average_method='arithmetic')
    except ValueError: pass
    try: vm = v_measure_score(y_true_eff, y_pred_eff)
    except ValueError: pass
    
    return ari, nmi, vm, num_clusters_found_total, num_effective_clusters


def run_clustering_task_worker(task_definition_args):
    status_pair_key, features_list = task_definition_args
    global worker_prefiltered_data_store
    
    df_all_features_for_pair, y_true_pair = worker_prefiltered_data_store[status_pair_key]
    
    results_for_this_task = []

    X_subset_np = df_all_features_for_pair[features_list].values
    
    if X_subset_np.shape[0] < 2: 
        return [] 

    scaler = PowerTransformer(method='yeo-johnson')
    try:
        X_scaled_subset = scaler.fit_transform(X_subset_np)
        if np.any(np.isnan(X_scaled_subset)) or np.any(np.isinf(X_scaled_subset)):
             return [] 
    except ValueError:
        return []

    for algo_config in ALGORITHMS_TO_TEST:
        algo_name = algo_config['name']
        model_class = algo_config['model']
        current_params = algo_config['params'].copy()

        if algo_config['needs_n_clusters']:
            current_params['n_clusters'] = N_CLUSTERS
            if X_scaled_subset.shape[0] < N_CLUSTERS: # Min check for k-means like algos
                continue # Skip this algo for this feature set
        
        if algo_name == 'AgglomerativeClustering' and current_params.get('linkage', 'ward') == 'ward':
            if 'affinity' in current_params: del current_params['affinity']
            current_params.setdefault('metric', 'euclidean')
            if current_params['metric'] != 'euclidean': # Ward only works with Euclidean
                 current_params['metric'] = 'euclidean'


        try:
            model = model_class(**current_params)
            labels = model.fit_predict(X_scaled_subset)
            
            ari, nmi, vm, num_total_clusters, num_eff_clusters = calculate_clustering_metrics(
                y_true_pair, labels
            )
            results_for_this_task.append({
                'status_pair': status_pair_key, 'features': features_list,
                'algorithm': algo_name, 'params': str(current_params),
                'ari': ari, 'nmi': nmi, 'v_measure': vm,
                'num_clusters_found': num_total_clusters, 'num_effective_clusters': num_eff_clusters
            })
        except Exception: 
            results_for_this_task.append({
                'status_pair': status_pair_key, 'features': features_list,
                'algorithm': algo_name, 'params': str(current_params),
                'ari': np.nan, 'nmi': np.nan, 'v_measure': np.nan,
                'num_clusters_found': -1, 'num_effective_clusters': -1,
            })
    return results_for_this_task

def save_results_to_csv(results_list, filename, is_final_top_n=False):
    if not results_list:
        print(f"No results to save to {filename}.")
        return
    
    df_to_save = pd.DataFrame(results_list)
    if 'features' in df_to_save.columns:
        df_to_save['features'] = df_to_save['features'].apply(
            lambda x: str(list(x)) if isinstance(x, (list, tuple)) else str(x)
        )
    if 'status_pair' in df_to_save.columns:
        df_to_save['status_pair'] = df_to_save['status_pair'].astype(str)
    
    # Sorting logic for intermediate/raw vs final_top_n
    if not is_final_top_n and PRIMARY_METRIC in df_to_save.columns:
         df_to_save.sort_values(
            by=['algorithm', 'status_pair', PRIMARY_METRIC],
            ascending=[True, True, False],
            inplace=True
        )
    elif not is_final_top_n :
         df_to_save.sort_values(
            by=['algorithm', 'status_pair'],
            ascending=[True, True],
            inplace=True
        )
    # For final_top_n, sorting is handled before calling this function.

    df_to_save.to_csv(filename, index=False, float_format='%.4f')
    print(f"Saved {len(df_to_save)} results to '{filename}'")

def format_metric_value(value):
    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return str(value)

if __name__ == "__main__":
    print(f"--- Main script started ---")
    overall_script_start_time = time.time()

    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Data file '{DATA_FILE}' not found. Exiting.")
        exit()

    feature_pattern = r'.*-\d+(\.\d+)?$'
    all_feature_cols = [col for col in df.columns if re.match(feature_pattern, col) and col != LABEL_COLUMN]
    
    critical_cols = all_feature_cols + [LABEL_COLUMN]
    df.dropna(subset=critical_cols, inplace=True)
    if df.empty:
        print("No data remaining after NA drop from critical columns. Exiting.")
        exit()

    unique_statuses = sorted(df[LABEL_COLUMN].unique())
    if len(unique_statuses) < 2:
        print(f"Not enough unique statuses ({len(unique_statuses)}) in '{LABEL_COLUMN}'. Need at least 2. Exiting.")
        exit()
    status_pairs_tuples = list(itertools.combinations(unique_statuses, 2))

    if not all_feature_cols:
        print("No feature columns found matching the pattern. Exiting.")
        exit()
    feature_combinations_tuples = []
    for k in range(1, MAX_FEATURES_TO_COMBINE + 1):
        feature_combinations_tuples.extend(itertools.combinations(all_feature_cols, k))
    
    if not feature_combinations_tuples:
        print("No feature combinations generated. Check MAX_FEATURES_TO_COMBINE or feature columns. Exiting.")
        exit()

    prefiltered_data_for_worker_init = {}
    valid_status_pairs_keys = []
    
    min_samples_for_pair_processing = max(N_CLUSTERS, 2)

    for sp_key in status_pairs_tuples:
        df_pair = df[df[LABEL_COLUMN].isin(sp_key)] 
        if len(df_pair[LABEL_COLUMN].unique()) == 2 and len(df_pair) >= min_samples_for_pair_processing:
            le = LabelEncoder()
            y_true_pair_np = le.fit_transform(df_pair[LABEL_COLUMN].values)
            prefiltered_data_for_worker_init[sp_key] = (df_pair[all_feature_cols].copy(), y_true_pair_np)
            valid_status_pairs_keys.append(sp_key)
    
    if not valid_status_pairs_keys:
        print("No valid status pairs after filtering. Exiting.")
        exit()

    task_definitions = []
    for sp_key_taskdef in valid_status_pairs_keys:
        for fs_tuple in feature_combinations_tuples:
            task_definitions.append((sp_key_taskdef, list(fs_tuple)))
    
    total_tasks_generated = len(task_definitions)
    if total_tasks_generated == 0:
        print("No task definitions generated to process. Exiting.")
        exit()

    all_results_from_all_chunks = []
    chunk_size = math.ceil(total_tasks_generated / NUMBER_OF_CHUNKS)
    print(f"Processing {total_tasks_generated} tasks in {NUMBER_OF_CHUNKS} chunks of approx. size {chunk_size}.")

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 1:
        num_workers = 1
    elif cpu_count <= 3:
        num_workers = cpu_count -1
    else:
        num_workers = min(cpu_count - 2, 8)
    num_workers = max(1, num_workers)

    print(f"Using {num_workers} worker processes.")

    for i_chunk in range(NUMBER_OF_CHUNKS):
        chunk_start_time = time.time()
        start_index = i_chunk * chunk_size
        end_index = min((i_chunk + 1) * chunk_size, total_tasks_generated)
        current_chunk_tasks = task_definitions[start_index:end_index]

        if not current_chunk_tasks:
            continue

        print(f"--- Starting Chunk {i_chunk+1}/{NUMBER_OF_CHUNKS} ({len(current_chunk_tasks)} tasks) ---")
        
        chunk_results_raw = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=(prefiltered_data_for_worker_init,) 
        ) as executor:
            future_to_task_def = {
                executor.submit(run_clustering_task_worker, task_def): task_def 
                for task_def in current_chunk_tasks
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_task_def), total=len(current_chunk_tasks), desc=f"Chunk {i_chunk+1}/{NUMBER_OF_CHUNKS}", smoothing=0.1):
                try:
                    result_list_from_worker = future.result() 
                    if result_list_from_worker:
                        chunk_results_raw.extend(result_list_from_worker)
                except Exception as exc:
                    # print(f"Task {task_def_failed} generated an exception: {exc}")
                    pass
        
        all_results_from_all_chunks.extend(chunk_results_raw)
        print(f"--- Chunk {i_chunk+1} Complete. {len(all_results_from_all_chunks)} total results so far. Time: {time.time() - chunk_start_time:.2f}s ---")


    if not all_results_from_all_chunks:
        print("\nNo results collected from any chunk. Final analysis cannot proceed.")
    else:
        grouped_results_algo_pair = defaultdict(list)
        for res in all_results_from_all_chunks:
            if pd.notna(res.get(PRIMARY_METRIC)):
                sp_key = str(res['status_pair'])
                grouped_results_algo_pair[(res['algorithm'], sp_key)].append(res)

        top_n_for_csv_list = []
        if grouped_results_algo_pair:
            for (algo, pair_str_key), results_list in grouped_results_algo_pair.items():
                sorted_results = sorted(
                    results_list, 
                    key=lambda x: (x.get(PRIMARY_METRIC, -float('inf')), x.get('nmi', -float('inf')), x.get('num_effective_clusters', -1)), 
                    reverse=True
                )
                top_n_for_csv_list.extend(sorted_results[:TOP_N_RESULTS])
            
            if top_n_for_csv_list:
                 top_n_for_csv_list.sort(
                    key=lambda x: (x['algorithm'], str(x['status_pair']), 
                                   -x.get(PRIMARY_METRIC, -float('inf')),
                                   -x.get('nmi', -float('inf')), 
                                   -x.get('num_effective_clusters', -1)
                                   ) 
                )
            save_results_to_csv(top_n_for_csv_list, FINAL_TOP_N_CSV_FILENAME, is_final_top_n=True)
        else:
            print("\nNo valid results for Top-N CSV grouping.")

        pair_then_algo_grouped_results = defaultdict(lambda: defaultdict(list))
        for res in all_results_from_all_chunks:
            if pd.notna(res.get(PRIMARY_METRIC)):
                sp_key = str(res['status_pair'])
                pair_then_algo_grouped_results[sp_key][res['algorithm']].append(res)

        top_1_results_for_console_display = []
        if pair_then_algo_grouped_results:
            for pair_str, algo_map in pair_then_algo_grouped_results.items():
                for algo_name, results_list in algo_map.items():
                    sorted_results = sorted(
                        results_list, 
                        key=lambda x: (x.get(PRIMARY_METRIC, -float('inf')), x.get('nmi', -float('inf')), x.get('num_effective_clusters', -1)), 
                        reverse=True
                    )
                    if sorted_results:
                        top_1_results_for_console_display.append(sorted_results[0])
        
        if not top_1_results_for_console_display:
            print("\nNo results found to generate the per-pair algorithm summary for console.")
        else:
            final_console_data_grouped_by_pair = defaultdict(list)
            for res in top_1_results_for_console_display:
                sp_key = str(res['status_pair'])
                final_console_data_grouped_by_pair[sp_key].append(res)

            print(f"\n\n--- Top 1 Result per Algorithm for Each Pair (based on {PRIMARY_METRIC.upper()}) ---")
            
            sorted_pair_keys_for_console = sorted(final_console_data_grouped_by_pair.keys())

            for pair_key_str in sorted_pair_keys_for_console:
                print(f"\nStatus Pair: {pair_key_str}")
                algo_results_for_this_pair = sorted(final_console_data_grouped_by_pair[pair_key_str], key=lambda x: x['algorithm'])
                for res_item in algo_results_for_this_pair:
                    features_display = res_item['features']
                    features_display_str = ", ".join(map(str, features_display)) if isinstance(features_display, (list, tuple)) else str(features_display)
                    
                    print(f"  Algorithm: {res_item['algorithm']}")
                    print(f"    Features: {features_display_str}")
                    print(f"    {PRIMARY_METRIC.upper()}: {format_metric_value(res_item.get(PRIMARY_METRIC))}, "
                          f"NMI: {format_metric_value(res_item.get('nmi'))}, "
                          f"VM: {format_metric_value(res_item.get('v_measure'))}")
                    print(f"    Clusters Found (Total/Effective): {res_item.get('num_clusters_found', 'N/A')}/{res_item.get('num_effective_clusters', 'N/A')}")

    overall_script_end_time = time.time()
    print(f"\n---  execution time: {overall_script_end_time - overall_script_start_time:.2f} seconds ---")
