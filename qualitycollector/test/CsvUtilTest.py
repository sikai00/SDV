import logging
import sys
import unittest
import warnings

from src.CsvUtil import CsvUtil


class CsvUtilTest(unittest.TestCase):
    VALID_CSV_PATH = "resources/CsvUtilTest/valid_csv_to_results.csv"
    MISSING_COL_CSV_PATH = "resources/CsvUtilTest/missing_col_csv_to_results.csv"
    MISSING_CELL_CSV_PATH = "resources/CsvUtilTest/missing_cell_csv_to_results.csv"

    def test_csv_to_results_when_valid_csv(self):
        results = CsvUtil.csv_to_results(self.VALID_CSV_PATH, "epochs")
        expected = [{
            "epochs": "1", "fit_time": "139.2727", "evaluate_time": "1382.5486",
            "overall": "0.6977931418408917", "shape": "0.7566633982175963", "trend": "0.6389228854641871"
        }, {
            "epochs": "50", "fit_time": "3023.0843", "evaluate_time": "724.6156",
            "overall": "0.8621655979870735", "shape": "0.8864614878107149", "trend": "0.837869708163432"
        }, {
            "epochs": "100", "fit_time": "6094.2817", "evaluate_time": "810.4943",
            "overall": "0.8428829854122655", "shape": "0.87167871751511", "trend": "0.814087253309421"
        }]
        self.assertEquals(results, expected)

    def test_csv_to_results_when_missing_col_csv(self):
        self.assertWarns(UserWarning, CsvUtil.csv_to_results, self.MISSING_COL_CSV_PATH, "epochs")

    def test_csv_to_results_when_missing_cell_csv(self):
        # Missing cells will result in a value of an empty string.
        self.assertWarns(UserWarning, CsvUtil.csv_to_results, self.MISSING_CELL_CSV_PATH, "epochs")
