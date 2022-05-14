from pykrx import stock
from pykrx import bond
import time
import random

def get_market_date_by_year(year):
    # 그 해 개장일 날짜를 구하기 위해서
    # 가장 상장일 및 창립이 오래된 회사 5곳인
    # 경방(000050) 대한통운(000120) 한진중공업홀딩스(003480) 동화약품(000020) 두산(000150)의 주가를 이용한다.
    set_marketdate=set()
    startdate=str(year)+"0101"
    enddate=str(year)+"1231"    
    for code in ["000050", "000120", "003480", "000020", "000150"]: # 경방(000050) 대한통운(000120) 한진중공업홀딩스(003480) 동화약품(000020) 두산(000150)
        df = stock.get_market_ohlcv(startdate, enddate, code)
        for datetime in df.index:
            set_marketdate.add(datetime.strftime("%Y%m%d"))
    lst_marketdate = sorted(set_marketdate)
    return lst_marketdate

def my_get_market_ohlcv(date, market):
    while True:
        try:
            df = stock.get_market_ohlcv(date, market=market)
            if len(df) < 1:
                time.sleep(180) # sleep 3 minutes
                continue
            break
        except:
                time.sleep(180) # sleep 3 minutes
                continue
    return df

def save_all_market_by_year(year):
    OF1=open("stock_%s_KOSPI.csv"%(year), "w")
    OF2=open("stock_%s_KOSDAQ.csv"%(year), "w")
    # KOSDAQ 시장은 1996년 7월 1일 시작 됨
    # KONEX 시장은 2013년 7월 1일 시작 됨
    #OF3=open("stock_%s_KONEX.csv"%(year), "w")
    header = ["date", "code", "open", "high", "low", "close", "volume", "tradingvalue", "change"]
    OF1.write(",".join(header)+"\n")
    OF2.write(",".join(header)+"\n")
    #OF3.write(",".join(header)+"\n")
    lst_marketdate = get_market_date_by_year(year)
    for date in lst_marketdate:
        # 1996년 7월 1일 (KOSDAQ 시작일)보다 이전 데이터는 검색 안되도록 함(api 오류 발생)
        if date < "19960701":
            continue
        # 2022년 3월 31일 (데이터 다운로드 종료기준일) 이후 데이터는 검색 안되도록 함
        if date > "20220331":
            continue
        df1 = my_get_market_ohlcv(date, market="KOSPI")
        for code, data in df1.iterrows():
            row = [date, code]+list(map(int, data[:-1]))+["%.2f"%data[-1]]
            OF1.write(",".join(list(map(str,row)))+"\n")
        print("%s KOSPI complete"%(date))
        time.sleep(random.uniform(0.8, 1.2))
        df2 = my_get_market_ohlcv(date, market="KOSDAQ")
        for code, data in df2.iterrows():
            row = [date, code]+list(map(int, data[:-1]))+["%.2f"%data[-1]]
            OF2.write(",".join(list(map(str,row)))+"\n")
        print("%s KOSDAQ complete"%(date))
        time.sleep(random.uniform(0.8, 1.2))
        #df3 = my_get_market_ohlcv(date, market="KONEX")
        #for row in df.values.tolist():
        #    OF3.write(",".join([date]+list(map(str,row)))+"\n")
        #print("%s KONEX complete"%(date))
        #time.sleep(random.uniform(0.8, 1.2))
    OF1.close()
    OF2.close()
    #OF3.close()

for year in range(1996, 2023):
    save_all_market_by_year(str(year))
