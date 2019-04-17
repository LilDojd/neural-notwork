import os
import pandas as pd


def create_batch(pdb_ids, csv_df):

    pdb_raws = list(map(lambda x: os.path.basename(x).split('_')[1], pdb_ids))
    en_class = [int(csv_df.loc[str(i)][2]) if csv_df.loc[str(i)] else None for i in pdb_raws]
    dataframe = pd.DataFrame(list(zip(pdb_ids, en_class)), columns=['PDBID', 'EN_CLASS'])
    melted = pd.melt(dataframe, value_vars=['EN_CLASS'], var_name='en_class', value_name='val')
    print(melted.groupby(by=['en_class', 'val'])['val'].count())


if __name__ == '__main__':
    import glob
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("pdb_input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("energy_csv", type=str)

    args = parser.parse_args()

    print("# Arguments")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    pdb_filenames = glob.glob(os.path.join(args.pdb_input_dir, "*.pdb"))
    en_table = pd.read_csv(args.energy_csv, index_col=0, header=None)
    create_batch(pdb_filenames, en_table)



