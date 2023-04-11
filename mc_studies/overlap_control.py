import os, sys
import numpy as np
import pandas as pd
import click

def check_overlap_no_index(df1, df2):
    names = ['dataset', 'run', 'event', 'subevent', 'pdg', 'inttype']

    a = [df1.dataset,
         df1.run,
         df1.event_id,
         df1.sub_event_id,
         df1.pdg,
         df1.IntType]

    b = [df2.dataset,
         df2.run,
         df2.event_id,
         df2.sub_event_id,
         df2.pdg,
         df2.IntType]

    m1 = pd.MultiIndex.from_arrays(a, names=names)
    m2 = pd.MultiIndex.from_arrays(b, names=names)

    overlap = m1.intersection(m2)
    diff1 = m1.difference(overlap)
    diff2 = m2.difference(overlap)

    #from IPython import embed
    #embed()

    #_df1 = df1
    #_df2 = df2
    df1 = df1.set_index(m1)
    df2 = df2.set_index(m2)
    #print(overlap)
    df1_o = df1.drop(diff1)
    df2_o = df2.drop(diff2)

    return df1_o, df2_o

def check_overlap(df1, df2):
    print("=== overlap_control: Technical Debt from Duplicates - dropping ===")
    df1 = df1.drop_duplicates()
    df2 = df2.drop_duplicates()

    index1 = df1.index
    index2 = df2.index
    overlap = index1.intersection(index2)
    return overlap

def drop(df, overlap):
    df = df.drop(overlap)
    return df

@click.command()
@click.argument('file1')
@click.argument('file2')
@click.argument('drop_choice', default='2')
@click.option('--out_file', '-o', default=None)
def main(file1, file2, drop_choice, out_file):
    df1 = pd.read_hdf(file1)
    df2 = pd.read_hdf(file2)
    overlap = check_overlap(df1, df2)
    if overlap.size < 100:
        print(overlap.values)
    print(f"Datafarme1: {len(df1)}")
    print(f"Dataframe2: {len(df2)}")
    print(f"Overlap   : {overlap.size}")
    if overlap.size == 0:
        print("No overlap!")
    else:
        if drop_choice == '2':
            df2 = drop(df2, overlap)
        if drop_choice == '1':
            df1 = drop(df1, overlap)
        overlap2 = check_overlap(df1, df2)
        print(overlap2.values)
        if overlap2.size != 0:
            raise ValueError("Overlap size not 0 after dropping!")
        if out_file != None:
            if drop_choice == '1':
                df1.to_hdf(out_file, key='df_filtered', mode='w') 
            if drop_choice == '2':
                df2.to_hdf(out_file, key='df_filtered', mode='w') 


if __name__ == "__main__":
    main()

##end
