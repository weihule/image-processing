import sqlite3
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit
import tushare as ts
import requests


db_url = "https://www.pythonanywhere.com/user/fanxing/files/home/fanxing/mysite/db/test-user.db"


def main():
    app = QApplication([])
    window = QMainWindow()
    window.resize(500, 400)
    window.move(300, 310)
    window.setWindowTitle('薪资统计')

    textEdit = QPlainTextEdit(window)
    textEdit.setPlaceholderText('请输入薪资表')
    textEdit.move(10, 25)
    textEdit.resize(300, 350)

    button = QPushButton('统计', window)
    button.move(380, 80)

    window.show()

    app.exec()


class SqLiteDB:
    def __init__(self, addr="./test-user.db"):
        if not Path(addr).exists():
            raise FileNotFoundError(f"{addr} not exists !")
        self.conn = sqlite3.connect(addr)
        self.cursor = self.conn.cursor()
        print(self.conn, type(self.conn))

    def query(self):
        sql = "select * from ACCOUNT"
        res = self.cursor.execute(sql)
        print(res.fetchall())


def test():
    # response = requests.get(db_url)
    # page_text = response.text
    # print(page_text)
    addr = "https://www.pythonanywhere.com/user/fanxing/files/home/fanxing/mysite/db/test-user.db"
    db = SqLiteDB(addr)


if __name__ == "__main__":
    # main()
    test()
    # test02()

