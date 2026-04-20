import unittest

import pandas as pd

from scripts.load_data import detect_water_header_row


class LoadDataTests(unittest.TestCase):
    def test_detect_water_header_row_prefers_second_row_when_first_is_generic(self):
        raw = pd.DataFrame(
            [
                ["Time", "Digesters Sludge Out Flow", "Unnamed: 2", "Unnamed: 3"],
                ["Time", "West Sludge Out (GPM)", "Eest Sludge Out (GPM)", "GBT Sludge Feed Pump 1 (GPM)"],
                ["10/01/25 12:00:00 AM", None, None, None],
            ]
        )

        header_row = detect_water_header_row(raw)

        self.assertEqual(header_row, 1)


if __name__ == "__main__":
    unittest.main()
