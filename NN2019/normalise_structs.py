import os
import pandas as pd


def create_batch(pdb_ids, csv_df):

    pdb_raws = list(map(lambda x: os.path.basename(x).split('_')[1], pdb_ids))
    en_class = [int(csv_df.loc[str(i)][2]) if i in csv_df.index else None for i in pdb_raws]
    dataframe = pd.DataFrame(list(zip(pdb_ids, en_class)), columns=['PDBID', 'EN_CLASS']).dropna()
    melted = pd.melt(dataframe, value_vars=['EN_CLASS'], var_name='en_class', value_name='val')
    count = melted.groupby(by=['en_class', 'val'])['val'].count()
    print(count)
    minimal = count.min()
    new_df = pd.DataFrame(columns=dataframe.columns)
    for val in dataframe['EN_CLASS'].unique():
        new_df = pd.concat([new_df, dataframe[dataframe['EN_CLASS'] == val].sample(minimal)])
    return new_df['PDBID'].tolist()


if __name__ == '__main__':

    import glob
    import argparse
    from shutil import copyfile

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
    pdbs = create_batch(pdb_filenames, en_table)
    for pdb in pdbs:
        print("...")
        copyfile(pdb, f"{args.output_dir}{os.path.basename(pdb)}")
    print("DONE")




