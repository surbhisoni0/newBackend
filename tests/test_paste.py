import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.data_service import parse_paste
from app.services.errors import ValidationFailure


class TestPaste(unittest.TestCase):
    def test_plain_numbers_and_full_rows(self):
        rows = parse_paste("5\n20260101100051234 0 Small red violet\n7 Big green")
        self.assertEqual([r["number"] for r in rows], [5, 0, 7])
        self.assertEqual(rows[1]["period_id"], "20260101100051234")

    def test_newest_first_reverses(self):
        self.assertEqual([r["number"] for r in parse_paste("1\n2\n3", newest_first=True)], [3, 2, 1])

    def test_rejects_scientific_notation_and_mismatch(self):
        for bad in ("2.03E+16 5 Big green violet", "3 Big green", "4 Small green", "7 Big red", "hello"):
            with self.assertRaises(ValidationFailure, msg=bad):
                parse_paste(bad)

    def test_empty(self):
        with self.assertRaises(ValidationFailure):
            parse_paste("  \n ")


if __name__ == "__main__":
    unittest.main()
