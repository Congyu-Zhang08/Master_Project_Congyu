import pandas as pd
import numpy as np
import joblib 
import seaborn as sns
import matplotlib.pyplot as plt
import os # Import os for directory creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import ks_2samp, spearmanr

# --- 1. Data Loading and Splitting ---
kit_file = '../KIT_all_scores.csv'
df_kit = pd.read_csv(kit_file)
# File paths
TRAIN_FILE_PATH = '../PDGFR_merged_all_scores.csv' 
NEW_TEST_FILE_PATH = '../FGFR2_merged_all_scores.csv'
MODEL_SAVE_DIR = './best_models/'

FEATURE_COLUMNS = ['interface_dG', 'interface_dSASA','binder_score','interface_dG_SASA_ratio','interface_delta_unsat_hbonds','interface_delta_unsat_hbonds_percentage','interface_hbond_percentage','interface_interface_hbonds','af3_binder_RMSD','af3_complex_RMSD','af3_plddt_binder','af3_pae_interaction_total','ptm','iptm','pDockQ','interface_packstat','interface_nres','interface_fraction','interface_hydrophobicity','clashes']

TARGET_COLUMN = 'binder_4000_nm'
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Load training data
df_train = pd.read_csv(TRAIN_FILE_PATH, usecols=ALL_COLUMNS)
df_train[TARGET_COLUMN] = df_train[TARGET_COLUMN].astype(bool)
X = df_train[FEATURE_COLUMNS]
y = df_train[TARGET_COLUMN].astype(int)

print(f"Xshape: {X.shape}")
print(f"yshape:{y.shape}")

n_features = X.shape[1]
n_tries = 100
result = []

# KS-Test to find optimal split
for random_state_i in range(n_tries):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_i,stratify=y)
    distances = list(map(lambda i : ks_2samp(X_train.iloc[:,i],X_test.iloc[:,i]).statistic,range(n_features)))
    result.append((random_state_i,max(distances)))

result.sort(key = lambda x : x[1])
print(f"KS-Test optimal split result: {result[:5]}")
random_state = result[0][0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state,stratify=y)


# --- 2. Preprocessing & Scaling ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Evaluation Function ---

def evaluate_model(model, X_data, y_true, model_name, y_rank_true=None):
    
    y_scores = None
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_data)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_data)
    
    # Handle baseline model (returns 0/1 predictions directly)
    if isinstance(model, pd.Series) or isinstance(model, np.ndarray):
        y_pred = model
    else:
        y_pred = model.predict(X_data)
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUCs only if scores are available (Skip for simple baseline prediction arrays)
    pr_auc = average_precision_score(y_true, y_scores) if y_scores is not None else np.nan
    roc_auc = roc_auc_score(y_true, y_scores) if y_scores is not None else np.nan
    
    #  Calculate Spearman Correlation if y_rank_true is provided
    spearman_rho = np.nan
    if y_scores is not None and y_rank_true is not None:
        try:
            rho, p_value = spearmanr(y_scores, y_rank_true)
            spearman_rho = rho
        except ValueError:
            # Handles cases where all scores/ranks are identical (e.g., small, uniform dataset)
            spearman_rho = 0.0
            
    
    print(f"--- {model_name} Evaluation Results ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    if y_rank_true is not None:
        print(f"Spearman Rho (Scores vs Rank): {spearman_rho:.4f}")
    
    return f1, y_pred


# --- 4. Grid Search Function (Optimizing for F1 Score) ---

def grid_search_holdout(pipeline_template, param_grid, X_train, y_train, X_test, y_test, model_name):
    
    param_combinations = list(ParameterGrid(param_grid))
    best_f1 = 0.0
    best_params = {}
    
    print(f"\n--- Starting {model_name} Hyperparameter Search (Holdout Validation) ---")
    print(f"Total Combinations: {len(param_combinations)}")
    
    for i, params in enumerate(param_combinations):
        
        # Instantiate a FRESH pipeline for this combination
        current_pipeline = pipeline_template.__class__(pipeline_template.steps)
        current_pipeline.set_params(**params)
        
        # Ensure Classifier estimators are set with class_weight='balanced'
        current_pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        
        current_pipeline.fit(X_train, y_train)
        
        # Evaluate performance (Optimizing for F1 Score)
        f1, _ = evaluate_model(current_pipeline, X_test, y_test, f"{model_name} Combination {i+1}", y_rank_true=None)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            
    # --- FIX: Retrain the best model using the optimal parameters ---
    if not best_params:
        print(f"Warning: No best parameters found for {model_name}. Returning None.")
        return None, {}

    best_pipeline = pipeline_template.__class__(pipeline_template.steps)
    best_pipeline.set_params(**best_params)

    # Re-apply class_weight for the final best model
    best_pipeline.named_steps['classifier'].set_params(class_weight='balanced')

    # Retrain on the entire training set using best parameters
    best_pipeline.fit(X_train, y_train)
    
    # --- Save the best model ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_filename = f"{MODEL_SAVE_DIR}{model_name.replace(' ', '_')}_best_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    
    print("\n---------------------------------------------------")
    print(f"Model saved to: {model_filename}")
    print(f"Best {model_name} F1 Score (Found in Search): {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    return best_pipeline, best_params


# --- 5. Define Pipelines and Search Grids ---

ALL_FEATURES_RANGE = list(range(1, n_features + 1))


# A. LR + SelectKBest Pipeline

lr_pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

param_grid_lr = {
    'feature_selection__k': ALL_FEATURES_RANGE,
    'classifier__C': [0.1, 0.5, 1, 10],
    'classifier__penalty': ['l1','l2']
}





# B. LinearSVC + SelectKBest Pipeline

svc_pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', LinearSVC(random_state=42, dual=False, max_iter=5000))
])

param_grid_svc = {
    'feature_selection__k': ALL_FEATURES_RANGE,
    'classifier__C': [0.01, 0.05, 0.1, 1]
}





# C. Random Forest + SelectKBest Pipeline

rf_pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_rf = {
    'feature_selection__k': ALL_FEATURES_RANGE,
    'classifier__n_estimators': [100, 150, 200, 250, 300], 
    'classifier__max_depth': [3, 5, 10]
}
# --- 6. Execution ---

# Perform searches
best_lr_pipe, _ = grid_search_holdout(lr_pipe, param_grid_lr, X_train_scaled, y_train, X_test_scaled, y_test, "LR+SelectKBest Pipeline")
best_svc_pipe, _ = grid_search_holdout(svc_pipe, param_grid_svc, X_train_scaled, y_train, X_test_scaled, y_test, "LinearSVC+SelectKBest Pipeline")
best_rf_pipe, _ = grid_search_holdout(rf_pipe, param_grid_rf, X_train_scaled, y_train, X_test_scaled, y_test, "RF+SelectKBest Pipeline")


# --- 7. Overall Best Model Selection & Internal Evaluation ---

final_models = {
    "LR+SelectKBest": best_lr_pipe,
    "LinearSVC+SelectKBest": best_svc_pipe,
    "RF+SelectKBest": best_rf_pipe
}

print("\n\n=============================================")
print("--- Internal Validation Set Performance ---")
print("=============================================")

# Internal Evaluation Loop
for name, model in final_models.items():
    if model is not None:
        evaluate_model(model, X_test_scaled, y_test, f"Final Internal Test {name}")


# --- 8. Baseline Model Definition and Evaluation ---

BASELINE_FEATURE = 'af3_plddt_binder'
BASELINE_THRESHOLD = 80
BASELINE_NAME = f"Baseline ({BASELINE_FEATURE} > {BASELINE_THRESHOLD})"

# a. Define the baseline prediction function
def baseline_predict(X_data_unscaled):
    # Base the prediction on the unscaled feature data
    return (X_data_unscaled[BASELINE_FEATURE] > BASELINE_THRESHOLD).astype(int)

# b. Evaluate baseline on internal test set
y_pred_baseline_internal = baseline_predict(X_test)
print("\n---------------------------------------------------")
evaluate_model(y_pred_baseline_internal, X_test_scaled, y_test, BASELINE_NAME) # Pass prediction array as model

# --- 9. Final Test on New Data & Visualization ---

BINDER_400NM_COLUMN = 'binder_400_nm'
VISUALIZATION_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN, BINDER_400NM_COLUMN]

# 9.1 Load New Data (with secondary column)
df_new = pd.read_csv(NEW_TEST_FILE_PATH, usecols=VISUALIZATION_COLUMNS)
df_new[TARGET_COLUMN] = df_new[TARGET_COLUMN].astype(bool)
df_new[BINDER_400NM_COLUMN] = df_new[BINDER_400NM_COLUMN].astype(bool)

X_new_test = df_new[FEATURE_COLUMNS]
y_new_test = df_new[TARGET_COLUMN].astype(int)
X_new_test_scaled = scaler.transform(X_new_test) # Scaled features for ML models

# --- Create Numerical Rank for Correlation and Visualization ---
# Rank 0: Non-binder (4000nm=F, 400nm=F)
# Rank 1: Binder (4000nm=T, 400nm=F)
# Rank 2: Strong Binder (4000nm=T, 400nm=T)
y_rank_new = np.select(
    [
        (~df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]), # Rank 0
        (df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]), # Rank 1
        (df_new[TARGET_COLUMN]) & (df_new[BINDER_400NM_COLUMN])  # Rank 2
    ],
    [0, 1, 2],
    default=0 # Assign 'Other/Unknown' to the lowest rank for robustness
)
y_rank_series = pd.Series(y_rank_new) # Convert to Series for passing to evaluate_model

# c. Evaluate baseline on new dataset
y_pred_baseline_new = baseline_predict(X_new_test)
print("\n=============================================")
print(f"--- FINAL TEST ON NEW DATASET: {NEW_TEST_FILE_PATH} ---")
print("=============================================")

# Calculate Spearman Rho for the baseline feature itself (unscaled scores vs ranks)
rho_baseline, p_value_baseline = spearmanr(X_new_test[BASELINE_FEATURE], y_rank_series)
print(f"--- {BASELINE_NAME} Feature Correlation ---")
print(f"Spearman Rho ({BASELINE_FEATURE} vs Rank): {rho_baseline:.4f} (p-value: {p_value_baseline:.4g})")


evaluate_model(y_pred_baseline_new, X_new_test_scaled, y_new_test, BASELINE_NAME, y_rank_true=y_rank_series)
print(f"\n{BASELINE_NAME} New Dataset Classification Report:\n", classification_report(y_new_test, y_pred_baseline_new))


# 9.2 Evaluate ML models on new dataset (for visualization)
df_plot_all = []

for model_name, model in final_models.items():
    if model is None:
        continue
        
    # Final Test evaluation
    evaluate_model(model, X_new_test_scaled, y_new_test, f"NEW DATA TEST: {model_name}", y_rank_true=y_rank_series)
    print(f"\n{model_name} New Dataset Classification Report:\n", classification_report(y_new_test, model.predict(X_new_test_scaled)))
    
    # Get Prediction Scores for Visualization
    y_scores_new = None
    if hasattr(model, "predict_proba"):
        y_scores_new = model.predict_proba(X_new_test_scaled)[:, 1]
        score_type = "Predicted Probability"
    elif hasattr(model, "decision_function"):
        y_scores_new = model.decision_function(X_new_test_scaled)
        score_type = "Decision Function Score"
    
    if y_scores_new is None:
        print(f"Warning: Skipping visualization for {model_name} as scores are unavailable.")
        continue

    # Define Visualization Categories
    df_temp = pd.DataFrame()
    df_temp['Score'] = y_scores_new
    df_temp['Model'] = model_name
    df_temp['Score Type'] = score_type
    df_temp['Category'] = np.select(
        [
            (~df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]), # Non-binder
            (df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]), # Binder 4000nm exclusive
            (df_new[TARGET_COLUMN]) & (df_new[BINDER_400NM_COLUMN])  # Strong Binder 400nm
        ],
        [
            'Non-binder',
            'Binder (4000nm)',
            'Strong Binder (400nm)'
        ],
        default='Other' # Should not happen based on logic
    )

    df_plot_all.append(df_temp)

# --- 10. Generate Separate Violin Plots ---

if not df_plot_all:
    print("\nSkipping visualization: No model scores were successfully collected.")
else:
    df_plot = pd.concat(df_plot_all)
    
    category_order = ['Non-binder', 'Binder (4000nm)', 'Strong Binder (400nm)']

    # 1. Visualization for Each ML Model (Separate Plots)
    for name in df_plot['Model'].unique():
        df_model = df_plot[df_plot['Model'] == name]
        score_type = df_model['Score Type'].iloc[0]

        plt.figure(figsize=(8, 6))

        sns.violinplot(
            x='Category', 
            y='Score', 
            data=df_model, 
            order=category_order,
            palette='viridis',
            inner='quartile'
        )

        plt.title(f'{name} Score Distribution by Binding Category (New Dataset)', fontsize=14)
        plt.xlabel('Binding Category', fontsize=12)
        plt.ylabel(score_type, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=15, ha='right')
        
        PLOT_FILENAME = f'new_dataset_scores_{name.replace(" ", "_")}_violin_plot.png'
        plt.savefig(PLOT_FILENAME, bbox_inches='tight')
        plt.close() # Close plot to save memory
        print(f"\nVisualization plot saved to {PLOT_FILENAME}")
        
    # 2. Visualization for Baseline Feature (af3_plddt_binder)
    
    # Prepare data for baseline feature visualization
    df_baseline_viz = pd.DataFrame()
    df_baseline_viz['Score'] = df_new[BASELINE_FEATURE]
    # Reuse the Category column from df_plot (since df_new indices match df_plot categories)
    df_baseline_viz['Category'] = np.select(
        [
            (~df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]),
            (df_new[TARGET_COLUMN]) & (~df_new[BINDER_400NM_COLUMN]),
            (df_new[TARGET_COLUMN]) & (df_new[BINDER_400NM_COLUMN])
        ],
        [
            'Non-binder',
            'Binder (4000nm)',
            'Strong Binder (400nm)'
        ],
        default='Other'
    )
    
    plt.figure(figsize=(8, 6))

    sns.violinplot(
        x='Category', 
        y='Score', 
        data=df_baseline_viz, 
        order=category_order,
        palette='viridis',
        inner='quartile'
    )

    plt.title(f'{BASELINE_FEATURE} Distribution by Binding Category (New Dataset)', fontsize=14)
    plt.xlabel('Binding Category', fontsize=12)
    plt.ylabel(f'{BASELINE_FEATURE} Score (Unscaled)', fontsize=12)
    plt.axhline(BASELINE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({BASELINE_THRESHOLD})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=15, ha='right')
    
    PLOT_FILENAME = f'new_dataset_baseline_{BASELINE_FEATURE}_violin_plot.png'
    plt.savefig(PLOT_FILENAME, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization plot saved to {PLOT_FILENAME}")
    
# --- 10. Output Best Model Features & Predict on KIT Dataset ---

print("\n\n=============================================")
print("--- Features Used by Best Models ---")
print("=============================================")

#print features of models
for name, model in final_models.items():
    if model:
        mask = model.named_steps['feature_selection'].get_support()
        selected_feats = [f for f, m in zip(FEATURE_COLUMNS, mask) if m]
        print(f"\n{name} Features ({len(selected_feats)}): {selected_feats}")

kit_file = '../KIT_all_scores.csv'
df_kit = pd.read_csv(kit_file)

X_kit = df_kit[FEATURE_COLUMNS]
X_kit_scaled = scaler.transform(X_kit)

for name, model in final_models.items():
    if model:
        col_name = f'pred_{name}'
        df_kit[col_name] = model.predict(X_kit_scaled)

df_kit.to_csv('KIT_all_scores_predicted.csv', index=False)
print(f"\nPredictions saved to KIT_all_scores_predicted.csv")

# --- 11. KIT Dataset Predictions & Noise Analysis ---
'''
kit_file = '../KIT_all_scores.csv'
df_kit = pd.read_csv(kit_file)

X_kit = df_kit[FEATURE_COLUMNS]
X_kit_scaled = scaler.transform(X_kit)

for name, model in final_models.items():
    if model:
        col_name = f'pred_{name}'
        df_kit[col_name] = model.predict(X_kit_scaled)

df_kit.to_csv('KIT_all_scores_predicted.csv', index=False)
print(f"\nPredictions saved to KIT_all_scores_predicted.csv")

print("\n--- Positive Rates by Noise Level ---")

for noise_val in ['50', '100']:
    subset = df_kit[df_kit['noise'].astype(str) == noise_val]
    print(f"\nNoise Level: {noise_val} (n={len(subset)})")
    
    if len(subset) > 0:
        for name in final_models.keys():
            if final_models[name]:
                pos_rate = subset[f'pred_{name}'].mean()
                print(f"  {name} Positive Rate: {pos_rate:.4%}")
    else:
        print("  No samples found for this noise level.")
'''