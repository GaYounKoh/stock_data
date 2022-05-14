from pykrx import stock
from pykrx import bond
from datetime import datetime
import time
import random
import sys

def get_market_date(startdate="19960701",enddate="20220331"):
    # 그 해 개장일 날짜를 구하기 위해서
    # 가장 상장일 및 창립이 오래된 회사 5곳인
    # 경방(000050) 대한통운(000120) 한진중공업홀딩스(003480) 동화약품(000020) 두산(000150)의 주가를 이용한다.
    set_marketdate=set()
    for code in ["000050", "000120", "003480", "000020", "000150"]: # 경방(000050) 대한통운(000120) 한진중공업홀딩스(003480) 동화약품(000020) 두산(000150)
        df = stock.get_market_ohlcv(startdate, enddate, code)
        for datetime in df.index:
            set_marketdate.add(datetime.strftime("%Y%m%d"))
    lst_marketdate = sorted(set_marketdate)
    return lst_marketdate

def adjusted_price(data):
    adjusted_data=[]
    lst_event=[]
    if len(data) <= 1:
        return data, lst_event
    for i in range(len(data)):
        close, change = data[i][5], data[i][8]
        if close == "0":
            if float(change) != 0.0 and change != "-100.00":
                print("ERROR: unexpected change", data[i])
                sys.exit()
            if i != 0:
                data[i][5] = data[i-1][5]
                data[i][8] = "0.00"
    cumulative_ratio = 1.0
    for i in range(len(data)-1,0,-1):
        pre_date, pre_code, pre_o, pre_h, pre_l, pre_c, pre_volume, pre_trading_value, pre_change = data[i-1]
        date, code, o, h, l, c, volume, trading_value, change = data[i]
        if cumulative_ratio == 1.0:
            adjusted_data.append(data[i])
        else:
            new_o, new_h, new_l, new_c = map(lambda x:str(int(round(float(x)*cumulative_ratio,0))), [o, h, l, c])
            new_volume = float(volume)/cumulative_ratio
            if new_volume > 0 and new_volume < 1:
                new_volume = "1"
            else:
                new_volume = str(int(round(new_volume,0)))
            new_change = change
            if new_c == "0":
                new_c = adjusted_data[-1][5]
                new_change = "0.00"
            adjusted_data.append([date, code, new_o, new_h, new_l, new_c, new_volume, trading_value, change])
        pre_c, c, change = float(pre_c), float(c), float(change)
        if pre_c != 0.0:
            if abs(round(c / pre_c, 4) - (1.0+change/100)) <= 0.0005:
                pass
            else:
                ratio = (c/(1.0+change/100))/pre_c
                lst_event.append([date,ratio])
                cumulative_ratio *= ratio
    date, code, o, h, l, c, volume, trading_value, change = data[0]
    if cumulative_ratio == 1.0:
        adjusted_data.append(data[0])
    else:
        new_o, new_h, new_l, new_c = map(lambda x:str(int(round(float(x)*cumulative_ratio,0))), [o, h, l, c])
        new_volume = str(int(round(float(volume)/cumulative_ratio,0)))
        new_change = change
        if new_c == "0":
            new_c = adjusted_data[-1][5]
            new_change = "0.00"
        adjusted_data.append([date, code, new_o, new_h, new_l, new_c, new_volume, trading_value, change])

    adjusted_data.reverse()
    return adjusted_data, lst_event

lst_marketdate = get_market_date()

def fill_date_fill_open_low_high(data):
    if len(data) == 0:
        return data
    new_data=[]
    i, j =0, 0
    date, code, o, h, l, c, vol, tv, change = data[0]
    if o == "0":
        o, h, l = c, c, c
    new_data.append([date, code, o, h, l, c, vol, tv, change])
    if len(data) == 1:
        return new_data 
    while True:
        pre_date = data[i][0]
        date = data[i+1][0]
        if (datetime.strptime(date, "%Y%m%d") - datetime.strptime(pre_date, "%Y%m%d")).days <= 180:
            while True:
                marketdate = lst_marketdate[j]
                if marketdate <= pre_date:
                    j+=1
                    continue
                if marketdate == date:
                    break
                if pre_date < marketdate and marketdate < date:
                    print("market date data added.", data[i][1], pre_date, date, marketdate)
                    code, c  = data[i][1], data[i][5]
                    o, h, l, vol, tv, change = c, c, c, "0", "0", "0.0"
                    new_data.append([marketdate, code, o, h, l, c, vol, tv, change])
                    j+=1
        date, code, o, h, l, c, vol, tv, change = data[i+1]
        if o == "0":
            o, h, l = c, c, c
        new_data.append([date, code, o, h, l, c, vol, tv, change])
        i+=1
        if i == len(data)-1:
            break
    return new_data

IF = open("stock.KOSPI.KOSDAQ.19960701to20220331.csv","r")
OF = open("stock.KOSPI.KOSDAQ.19960701to20220331.adjusted.datefilled.csv","w")
header = IF.readline() # read header
OF.write(header)
data = []
pre_code = None
for line in IF:
    s=line.strip().split(',')
    code = s[1]
    if pre_code != None and code != pre_code:
        #print(code, "adjust price start")
        adjusted_data, lst_event = adjusted_price(data)
        #print(code, "fill data start")
        adjusted_filled_data = fill_date_fill_open_low_high(adjusted_data)
        #print(len(data), len(adjusted_data),len(adjusted_filled_data))
        for row in adjusted_filled_data:
            OF.write(",".join(row)+"\n")
        data = []
    data.append(s)
    pre_code = code

if pre_code != None and code != pre_code:
    adjusted_data, lst_event = adjusted_price(data)
    adjusted_filled_data = fill_date_fill_open_low_high(adjusted_data)
    for row in adjusted_filled_data:
        OF.write(",".join(row)+"\n")
    data = []
