from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare.gnt_parser import iter_gnt_file


class GNTParserTest(unittest.TestCase):
    def test_single_record(self) -> None:
        bitmap = np.array([[0, 127], [200, 255]], dtype=np.uint8)
        tag = "中".encode("gb2312")
        size = 10 + bitmap.size
        record = struct.pack("<I2sHH", size, tag, 2, 2) + bitmap.tobytes()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.gnt"
            path.write_bytes(record)
            samples = list(iter_gnt_file(path))

        self.assertEqual(len(samples), 1)
        char, actual = samples[0]
        self.assertEqual(char, "中")
        np.testing.assert_array_equal(actual, bitmap)


if __name__ == "__main__":
    unittest.main()
