def apply_label_smoothing(df, target, alpha, threshold):
    df_target = df[target].copy()
    k = len(target)
    
    for idx, row in df_target.iterrows():
        if (row > threshold).any():
            row = (1 - alpha)*row + alpha/k
            df_target.iloc[idx] = row
    return df_target