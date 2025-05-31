import pandas as pd
import numpy as np
import re
import itertools
import time
import os
import warnings
import concurrent.futures
import traceback

from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch
)
import hdbscan

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from tqdm.auto import tqdm
from collections import defaultdict


DATA_FILE = 'data.csv'
LABEL_COLUMN = 'status'
TOP_N_RESULTS_PER_ALGO_PAIR = 31

FEATURE_COUNT_PENALTY_STRENGTH = -0.02

N_CLUSTERS = 2
RANDOM_STATE = 42

KMEANS_N_INIT = 10
SPECTRAL_AFFINITY = 'nearest_neighbors'
SPECTRAL_ASSIGN_LABELS = 'kmeans'
SPECTRAL_N_NEIGHBORS = 10
AGGLOMERATIVE_LINKAGE = 'ward'
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
HDBSCAN_MIN_CLUSTER_SIZE = 5
OPTICS_MIN_SAMPLES = 5
BIRCH_THRESHOLD = 0.5
BIRCH_BRANCHING_FACTOR = 50
AFFINITY_PROPAGATION_DAMPING = 0.5

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

FEATURE_GROUPS = {
    "COM": ["COM-1", "COM-2"],
    "POP": ["POP-1"],
    "STA": ["STA-1", "STA-2", "STA-3", "STA-4", "STA-5", "STA-6", "STA-7", "STA-8", "STA-9"],
    "TEC": ["TEC-1", "TEC-2", "TEC-3", "TEC-4"],
    "SWQ": ["SWQ-1", "SWQ-2.1", "SWQ-2.2", "SWQ-2.3", "SWQ-2.4", "SWQ-2.5", "SWQ-2.6", "SWQ-2.7"]
}

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
        'name': 'DBSCAN',
        'model': DBSCAN,
        'params': {'eps': DBSCAN_EPS, 'min_samples': DBSCAN_MIN_SAMPLES, 'n_jobs': 1},
        'needs_n_clusters': False
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

def calculate_all_metrics(X_scaled, y_true, labels_pred_original, algo_name_debug, status_pair_debug, subset_name_debug):
    """Calculates internal and external metrics robustly."""
    sil, db, ch = np.nan, np.nan, np.nan
    ari, nmi, vm = np.nan, np.nan, np.nan
    
    labels_pred = np.array(labels_pred_original)
    
    n_labels_internal = len(np.unique(labels_pred))
    n_samples_internal = len(X_scaled)

    if n_labels_internal >= 2 and n_labels_internal < n_samples_internal:
        try: sil = silhouette_score(X_scaled, labels_pred)
        except ValueError: pass
        try: db = davies_bouldin_score(X_scaled, labels_pred)
        except ValueError: pass
        try: ch = calinski_harabasz_score(X_scaled, labels_pred)
        except ValueError: pass
    
    non_noise_mask = (labels_pred != -1)
    y_true_eff = y_true[non_noise_mask]
    labels_pred_eff = labels_pred[non_noise_mask]
    
    num_clusters_found_total = len(np.unique(labels_pred_original))
    num_effective_clusters = 0

    if len(labels_pred_eff) > 0:
        unique_pred_eff_labels = np.unique(labels_pred_eff)
        num_effective_clusters = len(unique_pred_eff_labels)

        if num_effective_clusters >= 1 and len(np.unique(y_true_eff)) > 0 :
            try: ari = adjusted_rand_score(y_true_eff, labels_pred_eff)
            except ValueError: pass
            try: nmi = normalized_mutual_info_score(y_true_eff, labels_pred_eff, average_method='arithmetic')
            except ValueError: pass
            try: vm = v_measure_score(y_true_eff, labels_pred_eff)
            except ValueError: pass
            
    return sil, db, ch, ari, nmi, vm, num_clusters_found_total, num_effective_clusters


def run_one_clustering_task(task_args):
    status_pair, subset_name, feature_subset, pair_data, y_true_pair = task_args
    feature_subset = list(feature_subset)
    num_features = len(feature_subset)
    task_results_list = []
    binary_col_to_handle_separately = 'STA-6' 

    try:
        if not all(feat in pair_data.columns for feat in feature_subset):
            print(f"Skipping {subset_name} for {status_pair}: Features missing.")
            return []
        X_subset_df = pair_data[feature_subset].copy() 
    except KeyError:
        print(f"Skipping {subset_name} for {status_pair}: KeyError selecting features.")
        return []

    general_min_samples = max(
        N_CLUSTERS + 1, # for algorithms needing n_clusters
        DBSCAN_MIN_SAMPLES,
        (SPECTRAL_N_NEIGHBORS + 1 if SPECTRAL_AFFINITY == 'nearest_neighbors' else N_CLUSTERS + 1),
        HDBSCAN_MIN_CLUSTER_SIZE,
        OPTICS_MIN_SAMPLES
    )


    if X_subset_df.shape[0] < general_min_samples:
        return []
    if X_subset_df.shape[0] < 2:
        return []

    numerical_cols_in_subset = [col for col in feature_subset if col != binary_col_to_handle_separately]
    binary_cols_in_subset = [col for col in feature_subset if col == binary_col_to_handle_separately and col in X_subset_df.columns]

    transformers = []
    if numerical_cols_in_subset:
        # Check if numerical columns in X_subset_df before adding transformer
        actual_numerical_cols = [col for col in numerical_cols_in_subset if col in X_subset_df.columns]
        if actual_numerical_cols:
            transformers.append(('num', PowerTransformer(), actual_numerical_cols))
    
    if binary_cols_in_subset:
        transformers.append(('bin', 'passthrough', binary_cols_in_subset))

    if not transformers:
        return []

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') 

    try:
        X_scaled_subset = preprocessor.fit_transform(X_subset_df)

        if not isinstance(X_scaled_subset, np.ndarray):
            X_scaled_subset = X_scaled_subset.toarray() if hasattr(X_scaled_subset, "toarray") else np.array(X_scaled_subset)

        if np.any(np.isnan(X_scaled_subset)) or np.any(np.isinf(X_scaled_subset)):
            return []

        if numerical_cols_in_subset:
             actual_numerical_cols_in_df = [col for col in numerical_cols_in_subset if col in X_subset_df.columns]
             if actual_numerical_cols_in_df:
                numeric_data_before_scaling = X_subset_df[actual_numerical_cols_in_df].values
                if numeric_data_before_scaling.shape[1] > 0 and np.any(np.std(numeric_data_before_scaling, axis=0) < 1e-9):
                    return []
        
        if X_scaled_subset.shape[1] > 0:
            variances = np.var(X_scaled_subset, axis=0)
            if np.all(variances < 1e-9):
                pass
        elif X_scaled_subset.shape[1] == 0: # No features left
             return []


    except ValueError as e:
        print(f"ValueError: {status_pair}, {subset_name}, {e}")
        return []
    except Exception as e: 
        print(f"Exception: {status_pair}, {subset_name}: {e}")
        traceback.print_exc()
        return []

    # check if X_scaled_subset is empty
    if X_scaled_subset.shape[0] == 0 or X_scaled_subset.shape[1] == 0:
        return []
    
    
    # clustering
    for algo_config in ALGORITHMS_TO_TEST:
        algo_name = algo_config['name']
        model_class = algo_config['model']
        current_params = algo_config['params'].copy() # copy to not modifi

        if algo_config['needs_n_clusters']:
            current_params['n_clusters'] = N_CLUSTERS
        
        if algo_name == 'SpectralClustering':
            if current_params.get('affinity') == 'nearest_neighbors':
                current_params.setdefault('n_neighbors', SPECTRAL_N_NEIGHBORS)
                if X_scaled_subset.shape[0] <= current_params['n_neighbors']:
                    continue
            if current_params.get('assign_labels') == 'kmeans':
                current_params.setdefault('n_init', KMEANS_N_INIT)
        
        min_algo_samples_needed = 0
        if algo_name == 'DBSCAN':
             min_algo_samples_needed = current_params.get('min_samples', DBSCAN_MIN_SAMPLES)
        elif algo_name == 'OPTICS':
             min_algo_samples_needed = current_params.get('min_samples', OPTICS_MIN_SAMPLES)
        elif algo_name == 'HDBSCAN':
            min_algo_samples_needed = current_params.get('min_cluster_size', HDBSCAN_MIN_CLUSTER_SIZE)
        
        if min_algo_samples_needed > 0 and X_scaled_subset.shape[0] < min_algo_samples_needed:
            continue

        try:
            model = model_class(**current_params)
            if algo_name == 'HDBSCAN':
                if X_scaled_subset.shape[0] < current_params.get('min_cluster_size', HDBSCAN_MIN_CLUSTER_SIZE) and X_scaled_subset.shape[0] > 0:
                    if current_params.get('min_samples') is None:
                         if X_scaled_subset.shape[0] < current_params.get('min_cluster_size', HDBSCAN_MIN_CLUSTER_SIZE):
                              continue
                    elif X_scaled_subset.shape[0] < current_params.get('min_samples', 1):
                         continue

                model.fit(X_scaled_subset)
                labels = model.labels_
            else:
                labels = model.fit_predict(X_scaled_subset)

            sil, db, ch, ari, nmi, vm, n_clusters_total, n_clusters_eff = calculate_all_metrics(
                X_scaled_subset, y_true_pair, labels, algo_name, status_pair, subset_name
            )
            task_results_list.append({
                'status_pair': status_pair, 'subset_name': subset_name,
                'features': feature_subset, 'num_features': num_features,
                'algorithm': algo_name, 'params': str(current_params),
                'n_clusters_found_total': n_clusters_total, 'n_clusters_effective': n_clusters_eff,
                'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch,
                'ari': ari, 'nmi': nmi, 'v_measure': vm
            })
        except Exception:
            print(f"Clustering exception: {algo_name}, {status_pair}, {subset_name}")
            traceback.print_exc()
            pass
    return task_results_list


def normalize_metric(series, higher_is_better=True):
    series_clean = series.dropna()
    if series_clean.empty: return pd.Series(np.nan, index=series.index)
    min_val, max_val = series_clean.min(), series_clean.max()
    if max_val == min_val:
        result = pd.Series(np.nan, index=series.index)
        result[series_clean.index] = 0.5
        return result
    if higher_is_better: normalized = (series - min_val) / (max_val - min_val)
    else: normalized = (max_val - series) / (max_val - min_val)
    return normalized.clip(0, 1).reindex(series.index)


if __name__ == "__main__":
    main_start_time = time.time()


    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully loaded data from {DATA_FILE} (Shape: {df.shape})")
    except Exception as e: 
        print(f"Error loading data: {e}")
        exit()

    all_data_columns = df.columns.tolist()
    defined_features_from_groups = list(itertools.chain.from_iterable(FEATURE_GROUPS.values()))
    available_features_in_df = [f for f in defined_features_from_groups if f in all_data_columns]
    print(f"\nDefined features from groups: {len(defined_features_from_groups)}, available in df: {len(available_features_in_df)}")
    if LABEL_COLUMN not in df.columns: 
        print(f"Error: label column: '{LABEL_COLUMN}' not found.")
        exit()

    available_feature_groups = {}
    for group_name, feats in FEATURE_GROUPS.items():
        current_group_available_feats = [f for f in feats if f in available_features_in_df]
        if current_group_available_feats: 
            available_feature_groups[group_name] = current_group_available_feats
    if not available_feature_groups: 
        print("\nError: No usable feature groups after checking availability.")
        exit()
    print("\nUsing available feature groups for subset generation:", list(available_feature_groups.keys()))
    unique_statuses = sorted(df[LABEL_COLUMN].unique())
    print(f"Found unique statuses in label column: {unique_statuses}")

    features_to_use_for_na_check = list(itertools.chain.from_iterable(available_feature_groups.values()))
    cols_to_check_na = list(set(features_to_use_for_na_check + [LABEL_COLUMN]))
    initial_rows = len(df)
    df.dropna(subset=cols_to_check_na, inplace=True)
    rows_after_na = len(df)
    if initial_rows > rows_after_na: 
        print(f"\nWarning: Dropped {initial_rows - rows_after_na} rows due to NAs.")
    if rows_after_na == 0: 
        print("\nError: No data remaining after NA drop.")
        exit()
    unique_statuses_after_na = sorted(df[LABEL_COLUMN].unique())
    if len(unique_statuses_after_na) < 2: 
        print(f"\nError: Fewer than 2 unique statuses remaining after NA drop.")
        exit()
    print(f"Statuses remaining after NA drop: {unique_statuses_after_na}")

    status_pairs = list(itertools.combinations(unique_statuses_after_na, 2))
    print(f"\nGenerated {len(status_pairs)} status pairs.")
    feature_subsets_to_test = []
    group_names_available = list(available_feature_groups.keys())
    min_groups_to_combine = 1
    max_groups_to_combine = len(group_names_available)
    print(f"\nGenerating feature subsets from combinations of {min_groups_to_combine} to {max_groups_to_combine} available groups...")
    for k in range(min_groups_to_combine, max_groups_to_combine + 1):
        for group_name_tuple in itertools.combinations(group_names_available, k):
            combined_subset_name = "+".join(sorted(group_name_tuple))
            features_for_this_combo = sorted(list(set(itertools.chain.from_iterable(available_feature_groups[g] for g in group_name_tuple))))
            if features_for_this_combo:
                is_already_added_by_name = any(name == combined_subset_name for name, _ in feature_subsets_to_test)
                if not is_already_added_by_name: 
                    feature_subsets_to_test.append((combined_subset_name, features_for_this_combo))
    all_grouped_features = sorted(list(set(itertools.chain.from_iterable(available_feature_groups.values()))))
    if all_grouped_features:
        if not feature_subsets_to_test or feature_subsets_to_test[-1][1] != all_grouped_features:
            feature_subsets_to_test.append(("ALL_GROUPS", all_grouped_features))
        elif feature_subsets_to_test[-1][0] != "ALL_GROUPS":
            last_name, _ = feature_subsets_to_test.pop()
            feature_subsets_to_test.append(("ALL_GROUPS", all_grouped_features))
    print(f"Generated {len(feature_subsets_to_test)} predefined feature subsets to test.")
    if not feature_subsets_to_test: 
        print("\nError: No feature subsets generated to test.")
        exit()

    print("\n\n--- List of Generated Feature Subsets for Testing ---")
    if feature_subsets_to_test:
        for i, (subset_name, features_list) in enumerate(feature_subsets_to_test):
            print(f"  {i+1}. Subset Name: {subset_name}")
            if len(features_list) > 10:
                 print(f"     Features ({len(features_list)}): {features_list[:5]} ... (and {len(features_list)-5} more)")
            else:
                 print(f"     Features ({len(features_list)}): {features_list}")
    else:
        print("  No feature subsets were generated.")
    print("--- End of Generated Feature Subsets List ---\n")

    print("\nPre-filtering data for each status pair...")
    prefiltered_data_for_pairs = {}
    valid_status_pairs = []
    min_samples_for_pair = max(
        N_CLUSTERS + 1, 
        DBSCAN_MIN_SAMPLES, 
        OPTICS_MIN_SAMPLES,
        HDBSCAN_MIN_CLUSTER_SIZE,
        (SPECTRAL_N_NEIGHBORS + 1 if SPECTRAL_AFFINITY == 'nearest_neighbors' else N_CLUSTERS + 1)
    )
    for sp in tqdm(status_pairs, desc="Pre-filtering pairs"):
        if not (df[LABEL_COLUMN].eq(sp[0]).any() and df[LABEL_COLUMN].eq(sp[1]).any()): 
            continue
        df_pair_current = df[df[LABEL_COLUMN].isin(sp)].copy()
        if len(df_pair_current[LABEL_COLUMN].unique()) == 2 and len(df_pair_current) >= min_samples_for_pair:
            le = LabelEncoder()
            y_true_for_pair = le.fit_transform(df_pair_current[LABEL_COLUMN])
            prefiltered_data_for_pairs[sp] = (df_pair_current[available_features_in_df], y_true_for_pair)
            valid_status_pairs.append(sp)
    print(f"Proceeding with {len(valid_status_pairs)} valid status pairs after pre-filtering.")
    if not valid_status_pairs: 
        print("\nError: No valid status pairs to process.")
        exit()

    tasks_for_parallel_run = []
    for status_p in valid_status_pairs:
        if status_p in prefiltered_data_for_pairs:
            pair_features_df, y_true_p_np = prefiltered_data_for_pairs[status_p]
            for subset_n, f_list_for_subset in feature_subsets_to_test:
                if all(feat_name in pair_features_df.columns for feat_name in f_list_for_subset):
                    tasks_for_parallel_run.append(
                        (status_p, subset_n, f_list_for_subset, pair_features_df.copy(), y_true_p_np.copy())
                    )
    total_tasks_to_run = len(tasks_for_parallel_run)
    print(f"\nPrepared {total_tasks_to_run} tasks (pair x subset combinations for all algorithms).")
    if total_tasks_to_run == 0: 
        print("\nError: No valid tasks generated for processing.")
        exit()

    all_raw_results_list = []
    num_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using {num_workers} worker processes.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total_tasks_to_run, desc="Processing subsets/algorithms") as pbar:
            future_to_task_map = {executor.submit(run_one_clustering_task, task): task for task in tasks_for_parallel_run}
            for future in concurrent.futures.as_completed(future_to_task_map):
                original_task_info = future_to_task_map[future]
                try:
                    list_of_results_from_worker = future.result()
                    if list_of_results_from_worker: all_raw_results_list.extend(list_of_results_from_worker)
                except Exception as exc:
                    sp_err, sn_err, fs_err, _, _ = original_task_info
                    print(f'\n--- ERROR: Task execution failed for {sp_err}, {sn_err}, Features: {fs_err}, Error: {exc} ---')
                    traceback.print_exc()
                finally: pbar.update(1)
    print(f"\nFinished parallel processing. Found {len(all_raw_results_list)} raw successful algorithm runs.")

    # Results: Group by (status pair, algorithm), rank by adjusted score and get top N
    if not all_raw_results_list: 
        print("\nNo results generated from clustering runs.")
        exit()
    print(f"\n--- Starting Per-Pair, Per-Algorithm Ranking (Top {TOP_N_RESULTS_PER_ALGO_PAIR} by Adjusted Internal Score) ---")
    results_df = pd.DataFrame(all_raw_results_list)
    final_top_n_results_collector = []
    grouped_by_pair_then_algo = results_df.groupby(['status_pair', 'algorithm'])
    for (current_status_pair, current_algo_name), group_df_for_algo_pair_orig in tqdm(grouped_by_pair_then_algo, desc="Ranking per pair/algorithm"):
        group_df_for_algo_pair = group_df_for_algo_pair_orig.copy()
        if group_df_for_algo_pair.empty: continue
        group_df_for_algo_pair['sil_norm'] = normalize_metric(group_df_for_algo_pair['silhouette'], higher_is_better=True)
        group_df_for_algo_pair['db_norm'] = normalize_metric(group_df_for_algo_pair['davies_bouldin'], higher_is_better=False)
        group_df_for_algo_pair['ch_norm'] = normalize_metric(group_df_for_algo_pair['calinski_harabasz'], higher_is_better=True)
        norm_cols = ['sil_norm', 'db_norm', 'ch_norm']
        group_df_for_algo_pair['combined_internal_score'] = group_df_for_algo_pair[norm_cols].mean(axis=1, skipna=True)
        group_df_for_algo_pair.dropna(subset=['combined_internal_score'], inplace=True)
        if group_df_for_algo_pair.empty: continue
        group_df_for_algo_pair['num_features'] = pd.to_numeric(group_df_for_algo_pair['num_features'], errors='coerce')
        group_df_for_algo_pair['feature_penalty_multiplier'] = group_df_for_algo_pair['num_features'].apply(
            lambda nf: max(0.0, 1.0 + FEATURE_COUNT_PENALTY_STRENGTH * (nf - 1)) if pd.notna(nf) and nf > 0 else 1.0
        )
        group_df_for_algo_pair['adjusted_score'] = group_df_for_algo_pair['combined_internal_score'] * group_df_for_algo_pair['feature_penalty_multiplier']
        ranked_group = group_df_for_algo_pair.sort_values(by='adjusted_score', ascending=False)
        top_n_for_this_group = ranked_group.head(TOP_N_RESULTS_PER_ALGO_PAIR)
        final_top_n_results_collector.append(top_n_for_this_group)

    print("\n--- Analysis Complete ---")
    main_end_time = time.time()
    print(f"Total execution time: {main_end_time - main_start_time:.2f} seconds")
    if not final_top_n_results_collector: print(f"\nNo algorithm/pair combinations yielded ranked results.")
    else:
        final_report_df = pd.concat(final_top_n_results_collector, ignore_index=True)
        if not final_report_df.empty:
            final_report_df.sort_values(by=['status_pair', 'algorithm', 'adjusted_score'], ascending=[True, True, False], inplace=True)
            print(f"\n--- Top {TOP_N_RESULTS_PER_ALGO_PAIR} Results Per (Status Pair, Algorithm) ---")
            print(f"--- (Ranked by ADJUSTED Combined Internal Score, Penalty Strength: {FEATURE_COUNT_PENALTY_STRENGTH}) ---")
            current_pair_algo_group = None
            for _, row in final_report_df.iterrows():
                new_pair_algo_group = (row['status_pair'], row['algorithm'])
                if new_pair_algo_group != current_pair_algo_group:
                    current_pair_algo_group = new_pair_algo_group
                    print(f"\n\n==================== Status Pair: {row['status_pair']}, Algorithm: {row['algorithm']} ====================")
                    rank_in_group = 1
                print(f"\n  --- Rank {rank_in_group} (Adj. Score: {row['adjusted_score']:.4f}) ---")
                print(f"  Subset Name:        {row['subset_name']}")
                print(f"  Combined Raw Score: {row['combined_internal_score']:.4f}")
                print(f"  Num Features:       {int(row['num_features']) if pd.notna(row['num_features']) else 'N/A'}") 
                print(f"  Penalty Multiplier: {row['feature_penalty_multiplier']:.3f}")
                print(f"  Features:           {str(row['features'])}")
                print(f"  ARI (Reference):    {row.get('ari', np.nan):.4f}")
                print(f"  Internal (Sil: {row['silhouette']:.4f}, DB: {row['davies_bouldin']:.4f}, CH: {row.get('calinski_harabasz', np.nan):.1f})")
                print(f"  External (NMI: {row.get('nmi', np.nan):.4f}, VM: {row.get('v_measure', np.nan):.4f})")
                print(f"  Clusters (Total/Eff): {row.get('n_clusters_found_total', 'N/A')} / {row.get('n_clusters_effective', 'N/A')}")
                print(f"  Parameters Used:    {row['params']}")
                rank_in_group += 1
        else:
            print("\nNo results after concatenation and ranking for Top N report.")


    if 'final_report_df' in locals() and not final_report_df.empty:
        # make full results df for ARI top-1
        if all_raw_results_list:
            full_results_df_for_ari = pd.DataFrame(all_raw_results_list)
            full_results_df_for_ari['ari'] = pd.to_numeric(full_results_df_for_ari['ari'], errors='coerce')
            
            if not full_results_df_for_ari['ari'].isna().all():
                idx = full_results_df_for_ari.loc[full_results_df_for_ari['ari'].notna()].groupby(['status_pair', 'algorithm'])['ari'].idxmax()
                top1_by_ari = full_results_df_for_ari.loc[idx].reset_index(drop=True)
                top1_by_ari_sorted = top1_by_ari.sort_values(by='ari', ascending=False)

                print("\n--- Top-1 per (Status Pair, Algorithm) by ARI ---")
                for _, row in top1_by_ari_sorted.iterrows():
                    print(f"Status Pair: {row['status_pair']}, Algorithm: {row['algorithm']}, Subset: {row['subset_name']}, ARI: {row['ari']:.4f}")

                output_top1_file = 'top1_by_ari_per_pair_algo.csv'
                top1_by_ari_sorted.to_csv(output_top1_file, index=False, float_format='%.4f')
                print(f"Saved Top-1 ARI results to {output_top1_file}")
            else:
                print("No valid ARI scores found to generate Top-1 ARI report.")
        else:
            print("No raw results available to extract Top-1 ARI results.")
    elif not all_raw_results_list:
        print("No raw results available to extract Top-1 ARI results.")
    else:
        print("No final_report_df available to base Top-1 ARI report on.")


    # save r3esult to csv
    if 'final_report_df' in locals() and not final_report_df.empty:
        try:
            final_report_df['features'] = final_report_df['features'].astype(str)
            final_report_df['params'] = final_report_df['params'].astype(str)
            if 'num_features' in final_report_df.columns:
                 final_report_df['num_features'] = final_report_df['num_features'].astype('Int64')
            cols_order = [
                'status_pair', 'algorithm', 'subset_name', 'adjusted_score', 
                'combined_internal_score', 'feature_penalty_multiplier', 'num_features', 
                'ari', 'nmi', 'v_measure',
                'silhouette', 'davies_bouldin', 'calinski_harabasz',
                'sil_norm', 'db_norm', 'ch_norm',
                'n_clusters_found_total', 'n_clusters_effective',
                'features', 'params'
            ]
            existing_cols_in_order = [col for col in cols_order if col in final_report_df.columns]
            df_to_save = final_report_df[existing_cols_in_order]
            # Modified output filename
            output_filename = f'all_algos_top_PowerTransformed{TOP_N_RESULTS_PER_ALGO_PAIR}_ranked_by_ADJUSTED_score_penalty{FEATURE_COUNT_PENALTY_STRENGTH}.csv'
            df_to_save.to_csv(output_filename, index=False, float_format='%.4f')
            print(f"\nConsolidated Top-{TOP_N_RESULTS_PER_ALGO_PAIR} results saved to '{output_filename}'")
        except Exception as e: 
            print(f"\nError saving final results to CSV: {e}")
            traceback.print_exc()
    else: 
        print("\nNo Top-N results to save (final_report_df is empty or not defined).")
