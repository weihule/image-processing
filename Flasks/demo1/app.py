from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)


# url: http[80]/https[443]://www.baidu.com:443/path
@app.route('/')
def hello_world():  # put application's code here
    user = User("张三", "zuolinyou@163.com")
    person = {
        "name": "李四",
        "email": "lisi@gmail.com"
    }
    return render_template("index.html", user=user, person=person)


@app.route('/profile')
def profile():
    return render_template("index.html")


# 带参数的url,将参数固定到了path中
@app.route('/blog/<blog_id>')
def blog_detail(blog_id):
    return render_template("blog_detail.html", blog_id=blog_id, username="知了")


# /book/list 返回第一页信息
# /book/list?page=2 返回第二页信息
@app.route('/book/list')
def books():
    page = request.args.get(key="page", default=1, type=int)
    return f"您获取的是第 {page} 的列表"


class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email


def date_format(value, time_format="%Y-%m-%d %H:%m"):
    return value.strftime(time_format)


app.add_template_filter(date_format, "dformat")


@app.route('/filter')
def filter_demo():
    user = User("wangzhi", "zuolinyou@163.com")
    tnow = datetime.now()
    return render_template("filter.html", user=user, time=tnow)


@app.route('/control')
def control_demo():
    age = 67
    book_infos = [{"name": "三国演义", "author": "罗贯中"},
                  {"name": "银河帝国", "author": "阿西莫夫"}]
    return render_template("control.html", age=age, book_infos=book_infos)


@app.route('/child1')
def child1():
    return render_template("child1.html")


@app.route('/child2')
def child2():
    return render_template("child2.html")


@app.route('/static')
def static_demo():
    return render_template("static.html")


if __name__ == '__main__':
    # 开启debug模式
    app.run(debug=True,
            host="0.0.0.0",
            port=5001)
