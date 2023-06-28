import errno
from copy import deepcopy
from pathlib import Path
from time import time
from csv import DictReader

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sklearn.model_selection import train_test_split

from src.CsvUtil import CsvUtil
from src.LogsManager import LogsManager


# Tool to find out the relationship between a particular parameter in the specified synthesizer, and the quality report.
# This works by running through the entire process of generating data multiple times, using the different values for
# the parameter as specified by series.
class QualityCollector:
    pkl_directory = "fitted_synthesizers"

    logger = LogsManager.get_logger(__name__)

    def __init__(self, synthesizer_func, metadata, df_training, df_testing, valid_size_ratio, label_col_name,
                 parameter, series, sample_size, path, options=None, transformers=None, constraints=None):
        self.synthesizer_func = synthesizer_func  # Synthesizer constructor. Options are to be provided in `options`.
        self.metadata = metadata  # Metadata. Needed for the evaluation of quality
        self.parameter = parameter  # Name of parameter to be varied
        self.series = series  # Series of parameters points
        self.sample_size = sample_size  # Number of samples to synthesize and be evaluated against
        self.path = path  # Path of the output CSV file.

        # Default attributes
        if constraints is None:
            constraints = []
        if transformers is None:
            transformers = {}
        if options is None:
            options = {}
        self.options = options  # Dictionary. Keyword arguments for synthesizer, except the varied parameter.
        self.transformers = transformers  # Dictionary of Transformer mappings.
        self.constraints = constraints  # List of constraints mappings.

        self.df_testing = df_testing
        # Splitting into training and validation set
        self.df_training, self.df_valid = self._split_train_set(df_training, label_col_name, valid_size_ratio)
        self.results = []  # List of results, where key is aligned with the index of series.

    # Entry point of QualityCollector. Creates a CSV file with the quality reports at the end.
    def get_quality_reports(self):
        self._fit_synthesizer_series()

    # Get the directory of the model with the highest overall score.
    def _get_best_model_name(self):
        scores = [i["overall"] for i in self.results]
        highest_overall = max(scores)
        index = scores.index(highest_overall)
        param = self.results[index][self.parameter]
        return self._create_synthesizer_name(self.synthesizer_func, self.parameter, param, index)

    # Get the param that produced the model with the highest overall score.
    def _get_best_param(self):
        scores = [i["overall"] for i in self.results]
        highest_overall = max(scores)
        index = scores.index(highest_overall)
        return self.results[index][self.parameter]

    # Using the model with the best result, run the test dataset on it. This function is only intended to be called
    # after get_quality_reports has been completed. Otherwise, it will fail. Saves to self.path prefixed with "best_".
    def get_best_model_quality_report(self):
        best_model_results = {self.parameter: self._get_best_param()}
        model_directory = f"./{self.pkl_directory}/{self._get_best_model_name()}"
        loaded_synthesizer = self.synthesizer_func.load(filepath=model_directory)
        synthetic_data = loaded_synthesizer.sample(num_rows=self.sample_size)
        self._evaluate_quality(synthetic_data, best_model_results)
        CsvUtil.save_quality_report_to_csv([best_model_results], self.parameter, f"best_{self.path}")

    # Converts CSV to self.results and saves it. Needed if `get_best_model_quality_report` is used in a separate run.
    def csv_to_results(self, csv_path):
        results = []
        with open(csv_path) as f:
            reader = DictReader(f)
            for result in reader:
                results.append(result)
        self.results = results
        return results


    # Split the training set into training set and validation set according to the ratio given, using stratified
    # sampling.
    @staticmethod
    def _split_train_set(df_training, label_col_name, valid_size_ratio):
        train, test = train_test_split(df_training, test_size=valid_size_ratio, stratify=df_training[label_col_name])
        return train, test

    def _fit_synthesizer(self, param, results):
        tmp_options = deepcopy(self.options)  # Prevent mutating original options
        tmp_options[self.parameter] = param
        tmp_synthesizer = self.synthesizer_func(self.metadata, **tmp_options)
        tmp_synthesizer.auto_assign_transformers(self.df_training)
        tmp_synthesizer.add_constraints(self.constraints)
        tmp_transformer = deepcopy(self.transformers)  # SDV will mutate the transformer mapping
        tmp_synthesizer.update_transformers(tmp_transformer)
        self.logger.info(f"Fitting for parameter: '{self.parameter}', value: '{param}'...")
        ts = time()
        tmp_synthesizer.fit(self.df_training)
        te = time()
        fit_time = round(te - ts, 4)
        results["fit_time"] = fit_time
        self.logger.info(f"Time taken to fit: {fit_time}")
        return tmp_synthesizer

    def _save_fitted_synthesizer(self, synthesizer, parameter, param, index):
        Path(self.pkl_directory).mkdir(parents=True, exist_ok=True)
        pkl_name = self._create_synthesizer_name(self.synthesizer_func, parameter, param, index)
        pkl_full = f"./{self.pkl_directory}/{pkl_name}"
        if Path(pkl_full).exists():
            self.logger.warning("A synthesizer of the same name already exists. "
                  "Current iteration of synthesizer will not be saved.")
        else:
            synthesizer.save(filepath=pkl_full)

    @staticmethod
    def _create_synthesizer_name(synthesizer, parameter, param, index):
        return f"{index}-{synthesizer.__name__}_{parameter}_{param}.pkl"

    def _fit_synthesizer_series(self):
        for index, param in enumerate(self.series):
            results = {self.parameter: param}
            self.results.append(results)

            curr_synthesizer = self._fit_synthesizer(param, results)
            # Once model is fitted, temporarily save the .pkl file. This is done
            # since fitting takes a long time, and if the evaluation somehow fails
            # or takes too long, we can restart the progress.
            try:
                self.logger.info(f"Saving fitted synthesizer...")
                self._save_fitted_synthesizer(curr_synthesizer, self.parameter, param, index)
                self.logger.info(f"Saved fitted synthesizer.")
            except PermissionError:
                self.logger.warning("Unable to save current iteration of synthesizer. Sampling will still proceed.")
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    self.logger.error("Not enough space on the hard drive to save the current synthesizer.")
                else:
                    self.logger.error(f"An OSError occurred: {e}")

            # After we fit the synthesizer, we can draw samples.
            self.logger.info(f"Sampling for {self.sample_size} rows...")
            curr_synthetic_data = curr_synthesizer.sample(num_rows=self.sample_size)
            self.logger.info(f"Sampled {self.sample_size} rows.")

            # We can finally get our quality report.
            self.logger.info(f"Evaluating current synthesizer at {param} using validation set...")

            self._evaluate_quality(curr_synthetic_data, results)

            # Save the current available quality reports to CSV.
            CsvUtil.save_quality_report_to_csv(self.results, self.parameter, self.path)

    def _evaluate_quality(self, curr_synthetic_data, results):
        # Filling NaN cells with fake data to prevent exception:
        # See in progress issue: https://github.com/sdv-dev/SDMetrics/issues/273
        self.logger.info(f"_fillna running...")
        self.df_valid = self._fillna(self.df_valid)
        curr_synthetic_data = self._fillna(curr_synthetic_data)

        ts = time()
        curr_quality_report = evaluate_quality(self.df_valid, curr_synthetic_data, self.metadata)
        te = time()
        evaluate_time = round(te - ts, 4)

        results["evaluate_time"] = evaluate_time
        results["overall"] = curr_quality_report.get_score()
        results["shape"] = curr_quality_report.get_properties().loc[0, 'Score']
        results["trend"] = curr_quality_report.get_properties().loc[1, 'Score']

        self.logger.info(
            f"Overall: {results['overall']}, Shape: {results['shape']}, Trend: {results['trend']}")
        self.logger.info(f"Time taken to evaluate: {evaluate_time}")

    # Saves the quality report in CSV format.
    def _save_quality_report_to_csv(self, results, path):
        # Take each item in the list of quality reports, and create a record.
        records = []
        self.logger.info(f"Saving CSV to {path}...")
        for result in results:
            record = {}
            for col_name, value in result.items():
                record[col_name] = value
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved CSV to {path}.")

    def _fillna(self, df):
        for col in df:
            dt = self.metadata.to_dict()['columns'][col]['sdtype']

            if dt == 'numerical':
                nan_value = 0
            elif dt == 'categorical':
                nan_value = ''
            elif dt == 'datetime':
                nan_value = '01-Jan-15'
            else:
                nan_value = 0

            df[col] = df[col].fillna(value=nan_value)

        return df
