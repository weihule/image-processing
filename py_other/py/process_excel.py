from openpyxl import Workbook, load_workbook


def main():
    wb = Workbook()
    ws = wb.active

    ws1 = wb.create_sheet('sheet1', 1)
    ws2 = wb.create_sheet('sheet2', 2)

    # 获取名为 ‘sheet2’ 的表
    get_ws2 = wb['sheet2']
    print(get_ws2.title)

    wb.move_sheet(ws2, -1)

    cp_sheet = wb.copy_worksheet(ws1)

    print(wb.sheetnames)

    wb.save(r'D:\Desktop\delete_excel\first.xlsx')


if __name__ == "__main__":
    main()


