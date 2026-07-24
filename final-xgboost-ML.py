import h5py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import datetime
import seaborn as sns
import os 
import matplotlib.pyplot as plt


noise_fraction = 0.5
learning_rate = 0.00001
estimators=5000   
maxdepth=9
size_test = 0.2
# =========================================================
# 1. LOAD MULTIPLE FILES
# =========================================================
file_list = [
    '40906960_gen_001.i3.zst.h5',
    '40906960_gen_002.i3.zst.h5',
    '40906960_gen_003.i3.zst.h5',
    '40906960_gen_004.i3.zst.h5',
    '40906960_gen_005.i3.zst.h5',
    '40906960_gen_006.i3.zst.h5',
    '40906960_gen_007.i3.zst.h5',
    '40906960_gen_008.i3.zst.h5',
    '40906960_gen_009.i3.zst.h5',
    '40906960_gen_010.i3.zst.h5',
    '40917491_gen_001.i3.zst.h5',
    '40917491_gen_002.i3.zst.h5',
    '40917491_gen_003.i3.zst.h5',
    '40917491_gen_004.i3.zst.h5',
    '40917491_gen_005.i3.zst.h5',
    '40917491_gen_006.i3.zst.h5',
    '40917491_gen_007.i3.zst.h5',
    '40917491_gen_008.i3.zst.h5',
    '40917491_gen_009.i3.zst.h5',
    '40917491_gen_010.i3.zst.h5',
]

# We will store the condensed versions of each file in this list
all_condensed_data = []

# =========================================================
# 2. LOAD AND CONDENSE MULTIPLE FILES (SMART SUBSAMPLING)
# =========================================================
print(f"1. Loading and condensing {len(file_list)} file(s)...")

for file_name in file_list:
    print(f" -> Processing {file_name}...")
    
    try:
        with h5py.File(file_name, 'r') as f:
            # Load Signal (Label 1)
            signal_df = pd.DataFrame(f['Accepted_MCPEMap'][:])
            signal_df['label'] = 1 
            
            # Load Noise (Label 0)
            k40_df = pd.DataFrame(f['Noise_K40'][:])
            k40_df['label'] = 0 
            dark_df = pd.DataFrame(f['Noise_Dark'][:])
            dark_df['label'] = 0
            
        # Combine just for this file
        file_df = pd.concat([signal_df, k40_df, dark_df], ignore_index=True)
        
        # Keep only ML columns
        ml_columns = ['Event', 'string', 'om', 'pmt', 'time', 'npe', 'label']
        file_df = file_df[ml_columns].dropna()
        
        # --- THE MEMORY SAVING TRICK ---
        # Keep 100% of the valuable signal hits
        signal_hits = file_df[file_df['label'] == 1]
        
        # Randomly sample only 5% of the background noise to save RAM
        # (Change 0.05 to a lower number if your laptop still runs out of memory)
        noise_hits = file_df[file_df['label'] == 0].sample(frac=noise_fraction, random_state=42)
        
        condensed_file = pd.concat([signal_hits, noise_hits])
        all_condensed_data.append(condensed_file)
        
    except FileNotFoundError:
        print(f"    [WARNING] Could not find {file_name}. popraw nazwe pliku")

# =========================================================
# 3. PREPARE THE MASTER DATASET
# =========================================================
print("\n2. Combining all files into a Master Dataset...")
master_df = pd.concat(all_condensed_data, ignore_index=True)
print(f"Total rows for Machine Learning: {len(master_df)}")

# Separate Features (X) and Target (y)
# Notice we drop 'Event' here! We don't want the model memorizing event ID numbers.
X = master_df[['string', 'pmt', 'om', 'time', 'npe']]
y = master_df['label']

# Split into Training Data (80%) and taking a final Exam on Test Data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_test, random_state=42)

print(f"Training on {len(X_train)} hits, Testing on {len(X_test)} hits.")

# Calculate pos_weight for the loss function to handle class imbalance
num_noise = (y_train == 0).sum()
num_muons = (y_train == 1).sum()
weight_ratio = num_noise / num_muons
print(f"-> Calculated pos_weight ratio: {weight_ratio:.2f}")

# =========================================================
# 4. TRAIN THE XGBOOST MODEL
# =========================================================
print("\n4. Initializing XGBoost Model...")
# scale_pos_weight forces the model to pay extra attention to the rare signal hits
model = XGBClassifier(
    n_estimators=estimators,      # Number of decision trees
    learning_rate=learning_rate,     # How fast it learns
    max_depth=maxdepth,           # How complex each tree can be
    scale_pos_weight=weight_ratio,    # Balances the fact that we still have more noise than signal
    random_state=42,
    n_jobs=-1              # Uses all your Mac's CPU cores to train faster!
)

print("5.Training model... ")
model.fit(X_train, y_train)


# =========================================================
# 5. EVALUATE THE MODEL
# =========================================================
print("\n6. Evaluating the model...")
print("Making predictions on the unseen test data...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
print("\n=======================================================")
print("                 MODEL REPORT CARD")
print("=======================================================")
print(classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)']))




# =========================================================
# 6. GENERATING CONFERENCE PLOTS
# =========================================================
print("\n7. Generating and saving visualization plots...")
sns.set_theme(style="whitegrid")

# Plot 1: Receiver Operating Characteristic (ROC) Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}(area under the curve))')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('FP - False Positive Rate (Noise falsely flagged as Signal)', fontsize=12)
plt.ylabel('TP - True Positive Rate (Muons correctly detected)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join("xgboost_results", "xgboost_roc_curve.png"), dpi=300)
plt.show()

#Plot 2: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Noise', 'Predicted Muon'], 
            yticklabels=['Actual Noise', 'Actual Muon'],
            annot_kws={"size": 14})
plt.title('Model Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join("xgboost_results", "xgboost_confusion_matrix.png"), dpi=300)
plt.close()

print("-> Successfully saved 'xgboost_training_loss.png', 'xgboost_roc_curve.png', and 'xgboost_confusion_matrix.png' to your directory.")



# =========================================================
# 7. EXPERIMENT TRACKING LOG
# =========================================================
print("\n8. Saving results to experiment log...")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create a report log entry with all relevant details
log_entry = f"""
=======================================================
EXPERIMENT RUN: {timestamp}
=======================================================
Noise Fraction: {noise_fraction}
Learning Rate:  {learning_rate}
ROC AUC Score:  {roc_auc:.4f}
Test size:      {size_test}
Max Depth:      {maxdepth}
n_estimators:  {estimators}

Classification Report

{classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)'])}"""

# The 'a' means "append". It adds to the bottom instead of overwriting!
with open(os.path.join("xgboost_results", "xgboost_experiment_log.txt"), "a") as file:
    file.write(log_entry)

print("-> Journal entry added to 'xgboost_experiment_log.txt'!")