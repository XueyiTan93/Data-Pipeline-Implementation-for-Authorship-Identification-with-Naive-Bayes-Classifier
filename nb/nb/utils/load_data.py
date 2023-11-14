from pathlib import Path
import numpy as np
import pandas as pd

def extract_text(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data (
    ronald_reagan and barack_obama) and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=['author'])
    df_test = pd.DataFrame(columns=['author'])
    author_to_id_map = {'kennedy': 0, 'johnson': 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f'./{dir_name}/*.txt'):
            with open(f,encoding='utf-8') as fp:
                text = fp.read()
                if dir_name in ('kennedy', 'johnson'):
                    temp_df = pd.DataFrame({'author': dir_name, 'text': [text]})
                    df = pd.concat([df, temp_df], ignore_index=True)
                else:
                    temp_df = pd.DataFrame({'author': str(f).split('_')[-1][
                                                      :-4],
                                                      'text': [text]})
                    df = pd.concat([df, temp_df], ignore_index=True)
        return df
    for p in path.iterdir():
        if p.name in ('kennedy', 'johnson'):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == 'unlabeled':
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train['author'] = df_train['author'].apply(lambda x:
                                                  author_to_id_map.get(x))
    # do the same for the test data
    df_test['author'] = df_test['author'].apply(lambda x:
                                                  author_to_id_map.get(x))
    return df_train, df_test

