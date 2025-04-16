import unittest
from datetime import datetime

from utils.SnowFlake import Snowflake


class SnowFlakeTestCase(unittest.TestCase):

    def test_snow_flake(self):
        var = Snowflake().get_id()
        print(var)
        print(datetime.timestamp(datetime.now()))
        self.skipTest("test")


if __name__ == '__main__':
    unittest.main()
