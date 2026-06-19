import pandas as pd
true_df = pd.read_csv('dataset/sample_claims.csv')
pred_df = pd.read_csv('code/evaluation/sample_predictions.csv')
df = true_df[['user_id', 'risk_flags']].merge(pred_df[['user_id', 'risk_flags']], on='user_id', suffixes=('_true', '_pred'))
for _, row in df.iterrows():
    if row['risk_flags_true'] != row['risk_flags_pred']:
        print(f"{row['user_id']}: TRUE='{row['risk_flags_true']}' | PRED='{row['risk_flags_pred']}'")
