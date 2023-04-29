import os
import pymysql


def main():
    local_host = "127.0.0.1"
    aliyun = "47.115.228.106"
    conn = pymysql.connect(host=aliyun,
                           user="root",
                           password="123456",
                           port=3306,
                           db="itcase",
                           charset="utf8")

    cursor = conn.cursor()  # 生成游标对象
    for _ in range(1):
        sql = """select * from user_account"""
        try:
            cursor.execute(sql)
            # result = cursor.fetchone()  # 返回数据库查询的第一条信息，用元组显示
            result = cursor.fetchall()
            conn.commit()  # commit命令把事务做的修改保存到数据库
            print(result)
        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)
            print("查询失败")
    cursor.close()
    conn.close()


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
        try:
            conn = pymysql.connect(host=self.host,
                                   user=self.user,
                                   password=self.password,
                                   port=self.port,
                                   db=self.db,
                                   charset=self.charset)
            print("数据库连接成功")
            return conn
        except Exception as e:
            print("数据库连接失败")
            print(e)

    def find(self):
        db = self.connect()
        print(type(db))
        cursor = db.cursor()    # 生成游标对象
        sql = """select * from employee"""
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            print(result[0])
        except Exception as e:
            db.rollback()
            print("查询失败")
            print(e)

    def register(self, username, password):
        conn = self.connect()
        sql2 = """select * from user_account where username={}""".format(repr(username))

        sql = """insert into user_account (username, password) values ({}, {})""".\
            format(repr(username), repr(password))

        if conn.cursor().execute(sql2):
            print("用户名已存在")
            return

        cursor = conn.cursor()    # 生成游标对象
        cursor.execute(sql)
        conn.commit()
        print("注册成功")

        # 关闭连接
        cursor.close()
        conn.close()

    def login(self, username, password):
        conn = self.connect()
        sql = """select * from user_account where username={} and password={}""".\
            format(repr(username), repr(password))
        cursor = conn.cursor()    # 生成游标对象
        cursor.execute(sql)
        result = cursor.fetchone()

        if result:
            print(result)
            print("登陆成功")
        else:
            print("用户名或密码不正确")

        # 关闭连接
        cursor.close()
        conn.close()


def run(order: int, username: str, password: str):
    localhost = "127.0.0.1"
    aliyun = "47.115.228.106"
    conn = MySQL(host=localhost,
                 user="root",
                 password="123456",
                 db="itcase")
    if order == 1:
        conn.register(username, password)
    elif order == 0:
        conn.login(username, password)
    else:
        print("命令错误")
        return


if __name__ == "__main__":
    run(1, "zhangsan", "123456")
    # main()
