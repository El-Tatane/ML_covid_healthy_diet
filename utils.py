import re
import matplotlib.pyplot as plt
import yaml

from sklearn.model_selection import train_test_split



def camel_to_snake(name):
    name = name.replace("-", '_').replace(" ", "")
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    name = name.replace("&", '_&_').replace("__", "_")
    return name


def check_data_df(df):
    for col in df.columns:
        print(col,
              '# data: '+str(len(df[col])),
              '# unique data: '+str(len(df[col].unique())), df[col].unique().tolist()[:10],
              '% NAN: '+str(float(df[col].isna().sum())/float(len(df[col]))))


def plot_correlation_matrix(df):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


def get_correlated_columns(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr


def read_config_file(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def split_df_to_train_test(df, output_col, test_size=0.2):
    df_train, df_test = train_test_split(df, test_size=test_size)
    X_train = df_train.loc[:, df_train.columns != output_col].values
    y_train = df_train.loc[:, [output_col]].values
    X_test = df_test.loc[:, df_test.columns != output_col].values
    y_test = df_test.loc[:, [output_col]].values
    return X_train, y_train, X_test, y_test
