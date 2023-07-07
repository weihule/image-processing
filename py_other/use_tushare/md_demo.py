import thostmduserapi as mdapi


class CFtdcMdSpi(mdapi.CThostFtdcMdSpi):

    def __init__(self, tapi):
        mdapi.CThostFtdcMdSpi.__init__(self)
        self.tapi = tapi

    def OnRtnDepthMarketData(self, pDepthMarketData: mdapi.CThostFtdcDepthMarketDataField):
        print(pDepthMarketData.TradingDay,
              pDepthMarketData.reserve1,
              pDepthMarketData.ExchangeID,
              pDepthMarketData.reserve2,
              pDepthMarketData.LastPrice,
              pDepthMarketData.InstrumentID,
              pDepthMarketData.AveragePrice)


def main():
    mduserapi = mdapi.CThostFtdcMdApi.CreateFtdcMdApi()
    mduserspi = CFtdcMdSpi(mduserapi)
    mduserapi.RegisterSpi(mduserspi)
    mduserapi.Init()
    instruments = ["600000", "000001", "00700", "AAPL"]
    mduserapi.SubscribeMarketData([i.encode('utf8') for i in instruments], len(instruments))
    input()


if __name__ == '__main__':
    main()

