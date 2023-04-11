##generic functions

def remove_bad_reco(df):
    print(f'Removing Bad Reco - Current Events: {len(df.index.values)}')
    df = df[df.reco_energy != -9999]
    print(f'Remaining Events: {len(df.index.values)}')
    return df
