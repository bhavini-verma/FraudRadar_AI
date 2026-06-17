import time
import torch
import os
from transformers import AutoModel
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np

print('--- 1. Wav2Vec2 Indic V4 Load Test ---')
start_time = time.time()
try:
    print('Loading ai4bharat/wav2vec2-large-indic-v4-hindi...')
    model = AutoModel.from_pretrained('ai4bharat/wav2vec2-large-indic-v4-hindi')
    load_time = time.time() - start_time
    
    # Get model size in memory
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    print(f'Model loaded successfully in {load_time:.2f} seconds.')
    print(f'Model Memory Size: {size_mb:.2f} MB')
    
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'Error loading model: {e}')

print('\n--- 2. Deep Stream Standalone Accuracy ---')
# Let's write a small evaluation logic for XGB_Deep to get the explicit standalone EER.
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_index]

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    deep_path = os.path.join(base_dir, 'features', 'deep_features.csv')
    df_deep = pd.read_csv(deep_path).dropna()
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]
    
    # Reload the model
    xgb_deep = xgb.XGBClassifier()
    xgb_deep.load_model(os.path.join(base_dir, 'models', 'xgb_deep.json'))
    
    # To get test EER, we have to recreate the test split like in audit.py
    def extract_base_name(filename):
        if 'elevenlabs_advanced' in str(filename).lower():
            parts = str(filename).lower().replace('.wav', '').split('_')
            return parts[-1]
        return str(filename).lower()
        
    def group_shuffle_split(df_sub, random_state=42):
        df_sub = df_sub.copy()
        df_sub['GroupID'] = df_sub['Filename'].apply(extract_base_name)
        groups = df_sub['GroupID'].unique()
        np.random.seed(random_state)
        np.random.shuffle(groups)
        n = len(groups)
        test_idx = int(n * 0.15)
        val_idx = int(n * 0.15)
        test_groups = set(groups[:test_idx])
        return df_sub[df_sub['GroupID'].isin(test_groups)].index.values
        
    hindi_mask = df_deep['Filename'].str.contains('hindi', case=False) | df_deep['Filename'].str.contains('indicsynth', case=False) | df_deep['Filename'].str.contains('elevenlabs', case=False)
    df_hindi = df_deep[hindi_mask]
    df_eng = df_deep[~hindi_mask]
    
    eng_ts = group_shuffle_split(df_eng)
    hin_ts = group_shuffle_split(df_hindi)
    
    test_df = df_deep.loc[np.concatenate([eng_ts, hin_ts])]
    deep_cols = [c for c in test_df.columns if c.startswith('Deep_')]
    
    p_test_deep = xgb_deep.predict_proba(test_df[deep_cols])[:, 1]
    y_test = test_df['Label'].values
    
    deep_eer = compute_eer(y_test, p_test_deep)
    print(f'XGBoost A (Deep Stream) Standalone Overall EER: {deep_eer:.4f} ({deep_eer*100:.2f}%)')
    
except Exception as e:
    print(f'Error evaluating XGB_Deep: {e}')
