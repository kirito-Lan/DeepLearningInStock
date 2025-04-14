import unittest
import akshare as ak


class MyTestCase(unittest.TestCase):

    # 上证50
    def test_get_stock_sz50(self):
        option_cffex_sz50_list_sina_df = ak.option_cffex_sz50_list_sina()
        print(option_cffex_sz50_list_sina_df)

    # 中证1000
    def test_get_stock_zz1000(self):
        option_cffex_zz1000_list_sina_df = ak.option_cffex_zz1000_list_sina()
        print(option_cffex_zz1000_list_sina_df)
        self.skipTest("skip")
    # 沪深300
    def test_get_stock_hs300(self):
        option_cffex_hs300_list_sina_df = ak.option_cffex_hs300_list_sina()
        print(option_cffex_hs300_list_sina_df)
        self.skipTest("skip")

    # 中证500
    def test_get_stock_zz500(self):
        option_finance_board_df = ak.option_finance_board(symbol="南方中证500ETF期权", end_month="2306")
        print(option_finance_board_df)
        self.skipTest("skip")

    # 华夏科创50  期权
    def test_get_stock_hxkc(self):
        option_finance_board_df = ak.option_finance_board(symbol="华夏科创50ETF期权", end_month="2306")
        print(option_finance_board_df)
        self.skipTest("skip")
    def test_get_stock_yfdkc(self):
        option_finance_board_df = ak.option_finance_board(symbol="易方达科创50ETF期权", end_month="2306")
        print(option_finance_board_df)
        self.skipTest("skip")

    # 科创板
    def test_get_stock_kcb_report(self):
        stock_kc_a_spot_em_df = ak.stock_kc_a_spot_em()
        print(stock_kc_a_spot_em_df)
        self.skipTest("skip")

    # 创业板
    def test_get_stock_cyb_report(self):
        stock_cy_a_spot_em_df = ak.stock_cy_a_spot_em()
        print(stock_cy_a_spot_em_df)
        self.skipTest("skip")


if __name__ == '__main__':
    unittest.main()
