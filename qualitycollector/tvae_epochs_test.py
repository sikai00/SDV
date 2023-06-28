from rdt.transformers import AnonymizedFaker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

from src.QualityCollector import QualityCollector
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

df_training = pd.read_csv(r'./data/raw_data_70_new.csv')
df_testing = pd.read_csv(r'./data/raw_data_30_new.csv')

bad_rows_training = [13316, 5057, 10644, 14932, 86, 89, 4425]
bad_rows_empty = [1566, 2231, 7250, 8687, 8807, 12048, 12250, 12581, 12590, 13333, 14693, 17978, 18313, 18752, 22320]
df_training = df_training.drop(labels=bad_rows_training)
df_training = df_training.drop(labels=bad_rows_empty)

bad_rows_testing = [6094, 1725, 4887, 5846, 1538]
df_testing = df_testing.drop(labels=bad_rows_testing)

df_training['feature_10'] = pd.to_numeric(df_training['feature_10'])
df_training['feature_18'] = pd.to_numeric(df_training['feature_18'])
df_training['feature_61'] = pd.to_numeric(df_training['feature_61'])

df_testing['feature_10'] = pd.to_numeric(df_testing['feature_10'])
df_testing['feature_18'] = pd.to_numeric(df_testing['feature_18'])
df_testing['feature_61'] = pd.to_numeric(df_testing['feature_61'])


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(
    data=df_training
)


def manual_update_metadata(metadata):
    metadata.update_column(
        column_name='dt_opened',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )

    metadata.update_column(
        column_name='customer_no',
        sdtype='id'
    )

    metadata.set_primary_key(
        column_name='customer_no'
    )

    metadata.update_column(
        column_name='entry_time',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )

    metadata.update_column(
        column_name='feature_2',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )
    metadata.update_column(
        column_name='feature_10',
        sdtype='numerical'
    )
    metadata.update_column(
        column_name='feature_18',
        sdtype='numerical'
    )
    metadata.update_column(
        column_name='feature_20',
        sdtype='lexify',
        pii=True
    )
    metadata.update_column(
        column_name='feature_21',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )
    metadata.update_column(
        column_name='feature_22',
        sdtype='numerify',
        pii=True
    )
    metadata.update_column(
        column_name='feature_47',
        sdtype='lexify',
        pii=True
    )
    metadata.update_column(
        column_name='feature_53',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )
    metadata.update_column(
        column_name='feature_54',
        sdtype='datetime',
        datetime_format='%d-%b-%y'
    )

    metadata.update_column(
        column_name='feature_61',
        sdtype='numerical'
    )
    metadata.update_column(
        column_name='feature_77',
        sdtype='numerify',
        pii=True
    )


manual_update_metadata(metadata)
transformers = {
    'feature_20': AnonymizedFaker(function_name='lexify',
                                  function_kwargs={'text': '?????XXXXX', 'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}),
    'feature_22': AnonymizedFaker(function_name='numerify', function_kwargs={'text': '%####XXXXX'}),
    'feature_47': AnonymizedFaker(function_name='lexify',
                                  function_kwargs={'text': '?????XXXXX', 'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}),
    'feature_77': AnonymizedFaker(function_name='numerify', function_kwargs={'text': '%####XXXXX'})
}

synthesizer = TVAESynthesizer

# Parameter to be varied
parameter = "epochs"

# 5000 samples per model fitted
sample_size = 10000

# Series of the variable, e.g., 1,2,3 epochs
series = [1, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000]

# Options, empty for now, serve as a way to input synthesizer options
options = {"batch_size": 1500}

validation_ratio = 1/7

label_col_name = 'Bad_label'

qc = QualityCollector(synthesizer, metadata, df_training, df_testing, validation_ratio, label_col_name,
                      parameter, series, sample_size, "tvae_epochs_test.csv",
                      transformers=transformers)
qc.get_quality_reports()
qc.get_best_model_quality_report()

