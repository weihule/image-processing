import os
import pymysql


class MySQL:
    def __init__(self, host, user, password, db, port=3306, charset="utf8"):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.port = port
        self.charset = charset

    def connect(self):
        """
        连接数据库
        """
        conn = pymysql.connect(host=self.host,
                               user=self.user,
                               password=self.password,
                               port=self.port,
                               db=self.db,
                               charset=self.charset)

        return conn

    def find(self):
        db = self.connect()
        cursor = db.cursor()    # 生成游标对象
        sql = """select * from `emp`"""
        # noinspection PyBroadException
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            print(result[0])
        except Exception:
            db.rollback()
            print("查询失败")


def main():
    conn = pymysql.connect(host="127.0.0.1",
                           user="root",
                           password="123456",
                           port=3306,
                           db="itcase",
                           charset="utf8")
    cursor = conn.cursor()  # 生成游标对象
    for _ in range(1):
        sql = """select * from `emp`"""
        # noinspection PyBroadException
        try:
            cursor.execute(sql)
            result = cursor.fetchone()  # 返回数据库查询的第一条信息，用元组显示
            # result = cursor.fetchall()
            conn.commit()  # commit命令把事务做的修改保存到数据库
            print(result)
        except Exception:
            conn.rollback()  # 发生错误时回滚
            print("查询失败")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
