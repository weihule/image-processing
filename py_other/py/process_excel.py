from openpyxl import Workbook, load_workbook


def main():
    wb = Workbook()
    ws = wb.active

    for i in range(1, 11):
        for j in range(1, 6):
            ws.cell(i, j, i+j)

    print(ws['a:c'])
    wb.save(r'first.xlsx')


if __name__ == "__main__":
    main()


