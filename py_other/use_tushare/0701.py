import tushare as ts
import time
from tqdm import tqdm
import queue
import threading
from openpyxl import Workbook, load_workbook


pro = ts.pro_api(token='818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')


def get_datas():
    datas = pro.stock_basic()
    datas = datas.to_dict()
    ts_codes = datas['ts_code'].values()     # 股票代码
    names = datas['name'].values()           # 公司名称
    areas = datas['area'].values()           # 地区
    industries = datas['industry'].values()  # 行业
    markets = datas['market'].values()       # 板块
    # print(len(ts_codes), len(names), len(areas), len(industries), len(markets))

    # 存储示例: [[股票代码, 公司名称, 地区, 行业, 板块], ...]
    infos = []
    for t, n, a, i, m in zip(ts_codes, names, areas, industries, markets):
        temp = []
        temp.extend([t, n, a, i, m])
        infos.append(temp)

    return infos, list(datas['ts_code'].values())


# 存放处理之后的各个公司信息 [[[交易日], [开盘价], [最高价], [最低价], [涨跌幅]], ...]
all_datas = []


class ThreadQueue(threading.Thread):
    def __init__(self, name, q):
        super(ThreadQueue, self).__init__()
        self.name = name
        self.q = q

    def run(self):
        print('Starting ' + self.name)
        while True:
            try:
                self.process(self.name, self.q)

            except:
                break

    @staticmethod
    def process(thread_name, q: queue.Queue):
        per_info = q.get(timeout=2.5)
        try:
            temp = {}
            ts_code = per_info[0]
            temp[ts_code] = []
            temp[ts_code].append(per_info[1:])
            df = pro.daily(ts_code=ts_code)
            df = df.to_dict()
            trade_dates = list(df['trade_date'].values())[:5][::-1]
            opens = list(df['open'].values())[:5][::-1]
            highs = list(df['high'].values())[:5][::-1]
            lows = list(df['low'].values())[:5][::-1]
            pct_chgs = list(df['pct_chg'].values())[:5][::-1]
            temp[ts_code].append(trade_dates)
            temp[ts_code].append(opens)
            temp[ts_code].append(highs)
            temp[ts_code].append(lows)
            temp[ts_code].append(pct_chgs)
            all_datas.append(temp)
            if ts_code == '000001.SZ':
                print(ts_code, temp)
        except Exception as e:
            print(thread_name, 'Error: ', e)


def dict_slice(dicts, length):
    """
    截取字典中value的长度
    Args:
        dicts:
        length:

    Returns:

    """
    new_dict = {}
    keys = dicts.keys()
    for k in keys:
        v = dicts[k]
        new_v = {}
        for i in range(length):
            new_v[i] = v[i]
        new_dict[k] = new_v

    return new_dict


def main_queue(infos):
    # 填充队列
    work_queue = queue.Queue()
    for i in infos:
        work_queue.put(i)

    threads = []
    # 创建5个线程
    for i in range(5):
        thread = ThreadQueue(name='Thread-'+str(i+1),
                             q=work_queue)
        # 开启新线程
        thread.start()

        # 添加新线程到线程列表
        threads.append(thread)

    # 等到所有线程完成
    for t in threads:
        t.join()


def main(infos):
    # 遍历每一个公司
    for info in tqdm(infos):
        temp = []
        ts_code = info[0]
        df = pro.daily(ts_code=ts_code)
        df = df.to_dict()
        df = dict_slice(df, length=5)
        trade_dates = list(df['trade_date'].values())[:5][::-1]
        opens = list(df['open'].values())[:5][::-1]
        highs = list(df['high'].values())[:5][::-1]
        lows = list(df['low'].values())[:5][::-1]
        pct_chgs = list(df['pct_chg'].values())[:5][::-1]
        temp.append(trade_dates)
        temp.append(opens)
        temp.append(highs)
        temp.append(lows)
        temp.append(pct_chgs)
        all_datas.append(temp)

    print(len(all_datas))


def sort_dict(infos1, keys):
    """
    按照原始的ts_code顺序，给最终数据重新排序
    Returns:

    """
    temp_dict = {}
    for index, info in enumerate(infos1):
        ts_code = list(info.keys())[0]
        temp_dict[ts_code] = index
    # print(temp_dict)
    new_infos1 = []
    for k in keys:
        v = infos1[temp_dict[k]]
        new_infos1.append(v)

    return new_infos1


def process_file(infos1, keys):
    infos1 = sort_dict(infos1=infos1, keys=keys)
    print('开始写入...')
    wb = Workbook()
    ws = wb.active

    ws.append(['股票代码', '公司名称', '地点', '行业', '板块',
               '第一天日期', '开盘价', '最高价', '最低价', '跌涨幅'])

    width_20 = ['A', 'B', 'D', 'E', 'F']
    for i in width_20:
        ws.column_dimensions[i].width = 20.0

    print(f'len(infos1) = {len(infos1)}')
    for infos in infos1:
        for k, info in infos.items():
            ts_code = k
            company_name, loc, industry, market = info[0][0], info[0][1], info[0][2], info[0][3]
            try:
                date1 = info[1][0]
                date1_open = info[2][0]
                date1_high = info[3][0]
                date1_low = info[4][0]
                date1_pct_chg = info[5][0]
                # print(info)
                # print(info)
                # print(len(info[4]))
                if len(info[5]) == 5:
                    date2_pct_chg = info[5][1]
                    date3_pct_chg = info[5][2]
                    date4_pct_chg = info[5][3]
                    date5_pct_chg = info[5][4]
                elif len(info[5]) == 4:
                    date2_pct_chg = info[5][1]
                    date3_pct_chg = info[5][2]
                    date4_pct_chg = info[5][3]
                    date5_pct_chg = None
                elif len(info[5]) == 3:
                    date2_pct_chg = info[5][1]
                    date3_pct_chg = info[5][2]
                    date4_pct_chg = None
                    date5_pct_chg = None
                else:
                    date2_pct_chg, date3_pct_chg, date4_pct_chg, date5_pct_chg = None, None, None, None
                ws.append([ts_code, company_name, loc, industry, market,
                           date1, date1_open, date1_high, date1_low, date1_pct_chg,
                           date2_pct_chg, date3_pct_chg, date4_pct_chg, date5_pct_chg])
            except Exception as e:
                print(e)

    wb.save('test.xlsx')


def run():
    start_t = time.time()
    infos, origin_keys = get_datas()
    # infos, origin_keys = infos[:10], origin_keys[:10]
    # main(infos=infos)
    main_queue(infos=infos)
    process_file(infos1=all_datas, keys=origin_keys)
    cost_t = time.time() - start_t
    print(f'耗时 {cost_t:.2f} s')

    print(len(infos), infos[0])
    print(len(all_datas), all_datas[0])


def test():
    ts_code = '000010.SZ'
    df = pro.daily(ts_code=ts_code)
    # print(df)
    df = df.to_dict()
    new_df = dict_slice(df, length=5)
    print(new_df)


if __name__ == "__main__":
    # main()
    run()
    # process_file()
    # test()

