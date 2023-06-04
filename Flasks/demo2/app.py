from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime

app = Flask(__name__)

HOSTNAME = "111.230.192.230"
PORT = 3306
USERNAME = "root"
PASSWORD = "weiyounuLI01#"
DATABASE = "itcast"

app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/" \
                                        f"{DATABASE}?charset=utf8mb4"

db = SQLAlchemy(app)

with app.app_context():
    with db.engine.connect() as conn:
        res = conn.execute(text("""select * from tb_user"""))
        print(res.fetchall())


# url: http[80]/https[443]://www.baidu.com:443/path
@app.route('/')
def hello_world():  # put application's code here
    person = {
        "name": "李四",
        "email": "lisi@gmail.com"
    }
    return render_template("index.html", person=person)


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=5001)

