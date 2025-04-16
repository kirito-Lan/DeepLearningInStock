import unittest
import akshare as ak

from config.LoguruConfig import log
from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from constant.ExponentEnum import ExponentEnum
from test.Client.MainClient import  client

"""
akshare获取数据
"""

class MyTestCase(unittest.TestCase):
    def test_something(self):
        stock_zh_a_hist_df = ak.stock_zh_a_hist(
            symbol="000001",
            period="monthly",
            start_date="20170301",
            end_date='20231022',
            adjust=""
        )
        print(stock_zh_a_hist_df)
        self.skipTest("skip")  # 临时跳过该测试

    # 中国官方cpi
    def test_china_macro_cpi(self):
        macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()
        print(macro_china_cpi_monthly_df.to_string())
        self.skipTest("skip get cpi")

    # 中国官方pmi(另外还有财新的数据)
    def test_china_macro_pmi(self):
        macro_china_pmi_yearly_df = ak.macro_china_pmi_yearly()
        print(macro_china_pmi_yearly_df.to_string())
        self.skipTest("skip get pmi")

    # 中国官方ppi
    def test_china_macro_ppi(self):
        macro_china_ppi_yearly_df = ak.macro_china_ppi_yearly()
        print(macro_china_ppi_yearly_df.to_string())
        self.skipTest("skip get ppi")


    def test_base_response(self):
        print(BaseResponse[int].success(200))
        print(BaseResponse.fail(ErrorCode.PARAMS_ERROR))
        self.skipTest("test_base_response")

    def test_get_exponent(self):
        # 获取指数数据 源自东方财富  指标的数据15:00收盘
        index_zh_a_hist_df = ak.index_zh_a_hist(symbol=ExponentEnum.HS300.get_code(), period="daily", start_date="19700101",
                                                end_date="22220101")
        print(index_zh_a_hist_df)

    # 健康检查
    def test_health_check(self):
        response = client.get("/healthCheck")
        log.info(response.json())
        assert response.json() == {
            "code": 200,
            "msg": 'success',
            "data": "ok"
        }

if __name__ == '__main__':
    unittest.main()
