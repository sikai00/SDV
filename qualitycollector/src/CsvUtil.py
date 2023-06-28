import warnings
from csv import DictReader
import pandas as pd

from src.LogsManager import LogsManager


# Utility class for CSV-related functionalities.
class CsvUtil:
    logger = LogsManager.get_logger(__name__)

    EXPECTED_COL_NAMES = ["fit_time", "evaluate_time", "overall", "shape", "trend"]

    READ_INVALID_ROW_WARNING = "Reading row {} failed as it contains invalid/missing values. Skipping..."
    SAVE_INVALID_ROW_WARNING = "Saving row {} failed as it contains invalid/missing values. Skipping..."

    # Converts the CSV saved from a previous QualityCollector run to self.results and return it.
    # Rows with invalid or missing data are skipped and not used.
    # Needed if `get_best_model_quality_report` is used in a separate run.
    @classmethod
    def csv_to_results(cls, csv_path, parameter):
        results = []
        with open(csv_path) as f:
            dict_reader = DictReader(f)
            for index, result in enumerate(dict_reader):
                if not cls._validate_row(result, parameter):
                    warnings.warn(cls.READ_INVALID_ROW_WARNING.format(index))
                else:
                    results.append(result)
        return results

    # Check if the rows in the input CSV is valid, including the varied parameter.
    @classmethod
    def _validate_row(cls, row, parameter):
        if parameter not in row or row[parameter] is None or row[parameter] == "":
            return False
        for col_name in cls.EXPECTED_COL_NAMES:
            if col_name not in row or row[col_name] is None or row[col_name] == "":
                return False
        return True

    # Saves the quality report in CSV format.
    # Validates that the quality report produce has the correct columns and is not empty.
    @classmethod
    def save_quality_report_to_csv(cls, results, parameter, path):
        # Take each item in the list of quality reports, and create a record.
        records = []
        cls.logger.info(f"Saving CSV to {path}...")
        for index, result in enumerate(results):
            record = {}
            for col_name, value in result.items():
                record[col_name] = value
            if not cls._validate_row(record, parameter):
                warnings.warn(cls.READ_INVALID_ROW_WARNING.format(index), RuntimeWarning)
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df.to_csv(path, index=False)
        cls.logger.info(f"Saved CSV to {path}.")
