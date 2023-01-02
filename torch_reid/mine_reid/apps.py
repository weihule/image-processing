import os
from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import messagebox, ttk
import json
from PIL import Image, ImageTk


class LoginPage:
    def __init__(self, master: tk.Tk):
        # super().__init__(master=master)
        self.master = master
        self.master.geometry("690x350")
        self.master.title("登录页")

        self.login_page = tk.Frame(self.master, width=690, height=350)  # 创建Frame
        self.login_page.pack()

        self.v1 = tk.StringVar()
        self.v2 = tk.StringVar()
        self.master = master

        self.create_login()

    def create_login(self):
        """
        登录页面
        """

        # 创建 用户名 框
        tk.Label(self.login_page, text="用户名:").place(x=200, y=120)
        entry_usr_name = tk.Entry(self.login_page, textvariable=self.v1)
        entry_usr_name.place(x=260, y=120)

        # 创建 密码 框
        tk.Label(self.login_page, text="密码:").place(x=200, y=160)
        entry_usr_pwd = tk.Entry(self.login_page, textvariable=self.v2, show="*")
        entry_usr_pwd.place(x=260, y=160)

        # 登录, 注册, 退出按钮
        bt_login = ttk.Button(self.login_page, text="登录", command=self.usr_login)
        bt_login.place(x=200, y=200)
        bt_logup = ttk.Button(self.login_page, text="注册", command=self.usr_sign_up)
        bt_logup.place(x=300, y=200)
        bt_logquit = ttk.Button(self.login_page, text="退出", command=self.login_page.quit)
        bt_logquit.place(x=400, y=200)

    def usr_login(self):
        """
        用户登录
        """
        usr_name = self.v1.get()
        usr_pwd = self.v2.get()

        db = DataBase()
        flag, message = db.check_login(usr_name, usr_pwd)

        if flag:
            # messagebox.showinfo("登录成功", message=message)
            self.login_page.destroy()   # 销毁当前登录页面
            MainPage(self.master)
        else:
            messagebox.showerror("登录失败", message=message)

    def usr_sign_up(self):
        """
        用户注册
        """
        page_sign_up = tk.Frame(self.master, width=300, height=300)
        page_sign_up.pack()

        sign_up_name = tk.StringVar()
        sign_up_pwd = tk.StringVar()
        sign_up_pwd_confirm = tk.StringVar()

        tk.Label(page_sign_up, text="请输入用户名：").grid(row=2, column=1)
        tk.Entry(page_sign_up, textvariable=sign_up_name).grid(row=2, column=2, columnspan=4)

        tk.Label(page_sign_up, text="请输入密码：").grid(row=3, column=1)
        tk.Entry(page_sign_up, textvariable=sign_up_pwd, show='*').grid(row=3, column=2, columnspan=4)

        tk.Label(page_sign_up, text="请再次输入密码：").grid(row=4, column=1)
        tk.Entry(page_sign_up, textvariable=sign_up_pwd_confirm, show='*').grid(row=4, column=2, columnspan=4)

        bt_confirm = tk.Button(page_sign_up,
                               text="确认注册",
                               command=self.usr_sign_up_process)
        bt_confirm.grid(row=6, column=2)

    def usr_sign_up_process(self):
        pass


class DataBase:
    def __init__(self):
        with open('users.json', 'r', encoding='utf-8') as fr:
            self.users = json.load(fr)
    
    def check_login(self, user_name, user_pwd):
        if user_name in self.users:
            if user_pwd == self.users[user_name]:
                return True, "登陆成功!"
            else:
                return False, "密码错误!"
        else:
            return False, "用户名不存在!"


class MainPage:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Re-ID system")
        self.master.geometry("1000x600")        # W x H

        self.login_page = tk.Frame(self.master, width=1000, height=600)  # 创建Frame
        self.login_page.pack()

        # self.weights = tk.StringVar()
        # self.src_pic = tk.StringVar()
        self.weights = tk.StringVar()
        self.src_pic = tk.StringVar()

        self.create_page()

    def create_page(self):
        ttk.Button(self.login_page, text="选择模型", command=self.open_file).place(x=50, y=50)
        ttk.Button(self.login_page, text="选择图片", command=self.open_image).place(x=200, y=50)
        ttk.Button(self.login_page, text="开始重识别").place(x=350, y=50)

        ttk.Label(self.login_page, text="模型名称: ").place(x=50, y=100)
        ttk.Entry(self.login_page, textvariable=self.weights, width=40).place(x=120, y=100)

        # 图片排列控件
        ttk.Label(self.login_page, text="src", font=("宋体", 25)).place(x=50, y=150)
        ttk.Label(self.login_page, text="1", font=("宋体", 25)).place(x=200, y=150)
        ttk.Label(self.login_page, text="2", font=("宋体", 25)).place(x=350, y=150)
        ttk.Label(self.login_page, text="3", font=("宋体", 25)).place(x=500, y=150)
        ttk.Label(self.login_page, text="4", font=("宋体", 25)).place(x=650, y=150)
        ttk.Label(self.login_page, text="5", font=("宋体", 25)).place(x=800, y=150)

    def open_file(self):
        file = askopenfilename(title="请选择权重文件",
                               initialdir=r"D:",
                               filetypes=[("pth文件", ".pth"),
                                          ("onnx文件", ".onnx")])
        self.weights.set(file.split("/")[-1])

    def open_image(self):
        # paned = tk.PanedWindow(self.login_page)
        # paned.place(x=40, y=180)
        # paned.pack(fill=tk.X, side=tk.LEFT)
        file = askopenfilename(title="请选择图片文件",
                               initialdir=r"D:",
                               filetypes=[("图片文件", ".jpg"),
                                          ("图片文件", ".png"),
                                          ("图片文件", ".JPEG"),
                                          ("图片文件", ".PNG")])
        img = Image.open(file)
        print(img.size)
        self.login_page.photo = ImageTk.PhotoImage(img.resize((128, 256)))
        tk.Label(self.login_page, image=self.login_page.photo).place(x=10, y=200)


def run():
    root = tk.Tk()
    # LoginPage(root)
    MainPage(root)

    root.mainloop()


if __name__ == "__main__":
    run()

