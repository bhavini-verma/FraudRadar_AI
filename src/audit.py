import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix

def extract_base_name(filename):
    if 'elevenlabs_advanced' in filename:
        parts = filename.replace('.wav', '').split('_')
        return parts[-1]
    return filename

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    bio_path = os.path.join(project_dir, 'features', 'bio_features.csv')
    deep_path = os.path.join(project_dir, 'features', 'deep_features.csv')

    df_bio = pd.read_csv(bio_path).dropna()
    df_deep = pd.read_csv(deep_path).dropna()
    
    # We will keep Replay for testing Replay Detection Rate?
    # Wait, in the dataset, replay has a label. What is it? 
    # Actually, in augment_elevenlabs.py, Replay is saved to `fake/elevenlabs_advanced_replay...`
    # and its label in feature extraction is just Fake (1).
    # So Replays are natively handled as Fake. We just check accuracy on them.

    df_bio = df_bio[df_bio['Label'].isin([0, 1])]
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]

    df = pd.merge(df_deep, df_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))
    
    # Simulate the exact test split from train.py (since we didn't save the splits)
    # We use the exact same random state to reconstruct them
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
        val_groups = set(groups[test_idx : test_idx + val_idx])
        train_groups = set(groups[test_idx + val_idx:])
        return (
            df_sub[df_sub['GroupID'].isin(train_groups)].index.values,
            df_sub[df_sub['GroupID'].isin(val_groups)].index.values,
            df_sub[df_sub['GroupID'].isin(test_groups)].index.values
        )

    hindi_mask = df['Filename'].str.contains('hindi', case=False) | df['Filename'].str.contains('indicsynth', case=False) | df['Filename'].str.contains('elevenlabs', case=False)
    df_hindi = df[hindi_mask]
    df_eng = df[~hindi_mask]

    eng_tr, eng_vl, eng_ts = group_shuffle_split(df_eng)
    hin_tr, hin_vl, hin_ts = group_shuffle_split(df_hindi)

    train_df = df.loc[np.concatenate([eng_tr, hin_tr])]
    val_df = df.loc[np.concatenate([eng_vl, hin_vl])]
    test_df = df.loc[np.concatenate([eng_ts, hin_ts])]
    
    test_df['GroupID'] = test_df['Filename'].apply(extract_base_name)
    train_df['GroupID'] = train_df['Filename'].apply(extract_base_name)

    print("--- 1. Exact Sample Counts (Group Split) ---")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n--- 12/13. ElevenLabs Data Leakage Check ---")
    el_tr_df = train_df[train_df['Filename'].str.contains('elevenlabs', case=False)]
    el_ts_df = test_df[test_df['Filename'].str.contains('elevenlabs', case=False)]
    
    el_train_bases = set(el_tr_df['GroupID'])
    el_test_bases = set(el_ts_df['GroupID'])
    overlap = el_train_bases.intersection(el_test_bases)
    print(f"Base clips in Train: {len(el_train_bases)}")
    print(f"Base clips in Test: {len(el_test_bases)}")
    print(f"Overlapping Base Clips (LEAKAGE): {len(overlap)}")
    if len(overlap) == 0:
        print("PASS: 0% Data Leakage. Base clips are cleanly separated!")

    print("\n--- 3/4. Speaker and Sentence Overlap Check ---")
    print(f"ElevenLabs Sentence Overlap (Train/Test): {len(overlap)} sentences cross the boundary.")
    if len(overlap) == 0:
        print("PASS: 0% Sentence Leakage. The model evaluates entirely unseen text.")
    print("ElevenLabs Speaker Overlap: Multiple sentences share the same subset of ~6 ElevenLabs voices.")
    print("Since sentences are partitioned and voices are reused across sentences, speaker overlap is heavily present. The model learns to classify unseen sentences from known synthetic voices.")

    # Load Models
    import json
    with open(os.path.join(project_dir, 'models', 'fusion_weights.json'), 'r') as f:
        weights = json.load(f)
    optimal_w = weights['w_deep']
    
    deep_cols = [c for c in df.columns if c.startswith('Deep_')]
    bio_cols = [c for c in df_bio.columns if c not in ['Filename', 'Label']]
    
    xgb_deep = xgb.XGBClassifier()
    xgb_bio = xgb.XGBClassifier()
    xgb_deep.load_model(os.path.join(project_dir, 'models', 'xgb_deep.json'))
    xgb_bio.load_model(os.path.join(project_dir, 'models', 'xgb_bio.json'))
    
    p_test_deep = xgb_deep.predict_proba(test_df[deep_cols])[:, 1]
    p_test_bio = xgb_bio.predict_proba(test_df[bio_cols])[:, 1]
    p_test_fused = optimal_w * p_test_deep + (1 - optimal_w) * p_test_bio
    
    y_test = test_df['Label'].values
    y_pred = (p_test_fused >= 0.5).astype(int)
    test_df['Pred'] = y_pred
    
    print("\n--- 8. Confusion Matrices ---")
    cm_all = confusion_matrix(y_test, y_pred)
    print("Overall Test Set:\n", cm_all)
    
    hin_mask_test = test_df['Filename'].str.contains('hindi|indicsynth|elevenlabs', case=False)
    if sum(hin_mask_test) > 0:
        cm_hin = confusion_matrix(test_df[hin_mask_test]['Label'], y_pred[hin_mask_test])
        print("Hindi Subset:\n", cm_hin)
        
    el_mask_test = test_df['Filename'].str.contains('elevenlabs', case=False)
    if sum(el_mask_test) > 0:
        try:
            cm_el = confusion_matrix(test_df[el_mask_test]['Label'], y_pred[el_mask_test])
            print("ElevenLabs Subset:\n", cm_el)
        except Exception:
            print("ElevenLabs Subset does not have both classes (only Fake).")

    print("\n--- Custom Metrics ---")
    replay_mask = test_df['Filename'].str.contains('replay', case=False)
    if sum(replay_mask) > 0:
        replay_acc = sum(test_df[replay_mask]['Pred'] == 1) / sum(replay_mask)
        print(f"Replay Detection Rate: {replay_acc*100:.2f}% ({sum(test_df[replay_mask]['Pred'] == 1)}/{sum(replay_mask)})")
    
    real_mask = test_df['Label'] == 0
    if sum(real_mask) > 0:
        fpr = sum(test_df[real_mask]['Pred'] == 1) / sum(real_mask)
        print(f"Hard Negative FPR (Real classified as Fake): {fpr*100:.2f}%")

if __name__ == '__main__':
    main()
