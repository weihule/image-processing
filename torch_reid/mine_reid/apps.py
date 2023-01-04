import os
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory
import tkinter as tk
from tkinter import messagebox, ttk
import json
from PIL import Image, ImageTk
import onnxruntime
from analysis import SimpleInfer


class LoginPage(object):
    def __init__(self, master: tk.Tk):
        # super().__init__(master=master)
        self.master = master
        self.master.geometry("690x350")
        self.master.title("欢迎登录")

        self.login_page = tk.Frame(self.master, width=690, height=350)  # 创建Frame
        self.login_page.pack()

        self.v1 = tk.StringVar()
        self.v2 = tk.StringVar()

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
            self.login_page.destroy()  # 销毁当前登录页面
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


class DataBase(object):
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


class MainPage(object):
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Re-ID system")
        self.master.geometry("1000x600")  # W x H

        self.login_page = tk.Frame(self.master, width=1000, height=600)  # 创建Frame
        self.login_page.pack()

        self.weights = tk.StringVar()       # 权重文件
        self.src_pic = tk.StringVar()       # probe图片
        self.model_name = tk.StringVar()    # 模型名称
        self.gallery_dir = tk.StringVar()   # 搜索文件夹
        self.cv2_img = ''

        self.create_page()

    def create_page(self):
        ttk.Label(self.login_page, text="输入模型名称: ").place(x=50, y=50)
        ttk.Entry(self.login_page, textvariable=self.model_name, width=40).place(x=135, y=50)
        ttk.Button(self.login_page, text="点击确定", command=self.confirm_model).place(x=440, y=48)

        ttk.Label(self.login_page, text="权重文件路径: ").place(x=50, y=80)
        ttk.Entry(self.login_page, textvariable=self.weights, width=40).place(x=135, y=80)
        ttk.Button(self.login_page, text="选择路径", command=self.open_file).place(x=440, y=78)

        ttk.Label(self.login_page, text="图片库: ").place(x=50, y=110)
        ttk.Entry(self.login_page, textvariable=self.gallery_dir, width=40).place(x=135, y=110)
        ttk.Button(self.login_page, text="选择文件夹", command=self.open_dir).place(x=440, y=108)

        ttk.Button(self.login_page, text="选择图片", command=self.open_image).place(x=50, y=145)
        ttk.Button(self.login_page, text="开始重识别", command=self.start_infer).place(x=150, y=145)

        # 图片排列控件
        ttk.Label(self.login_page, text="src", font=("宋体", 25)).place(x=50, y=180)
        ttk.Label(self.login_page, text="1", font=("宋体", 25)).place(x=200, y=180)
        ttk.Label(self.login_page, text="2", font=("宋体", 25)).place(x=350, y=180)
        ttk.Label(self.login_page, text="3", font=("宋体", 25)).place(x=500, y=180)
        ttk.Label(self.login_page, text="4", font=("宋体", 25)).place(x=650, y=180)
        ttk.Label(self.login_page, text="5", font=("宋体", 25)).place(x=800, y=180)

    def open_file(self):
        file = askopenfilename(title="请选择权重文件",
                               initialdir=r"D:",
                               filetypes=[("pth文件", ".pth")])
        # 将file赋值给self.weights
        self.weights.set(file)

    def open_dir(self):
        file = askdirectory(title="请选择图片库文件夹",
                            initialdir=r"D:")
        # 将file赋值给self.gallery_dir
        self.gallery_dir.set(file)

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
        self.src_pic.set(file)
        img = Image.open(file)
        self.cv2_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.login_page.photo = ImageTk.PhotoImage(img.resize((128, 256)))
        tk.Label(self.login_page, image=self.login_page.photo).place(x=10, y=220)

    def confirm_model(self):
        # 获取entry输入的内容
        model_name = self.model_name.get()

    def start_infer(self):
        file = self.weights.get()
        onnx_infer = OnnxInfer(file)
        outs = onnx_infer.forward(self.cv2_img)
        print(outs[0].shape)


class OnnxInfer(object):
    def __init__(self, onnx_file, resized_w=128, resized_h=256, batch_size=1):
        self.onnx_file = onnx_file
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.batch_size = batch_size

        self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
    
    def forward(self, image_numpy, gallery_dir, gallery_data_name):
        # batched_img = np.random.random(size=(1, 3, 256, 128)).astype(np.float32)
        img = self.resize_img(image_numpy, resized_w=self.resized_w, resized_h=self.resized_h)
        batched_img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)   # [h, w, c] -> [1, c, h, w]
        input_feed = self.get_input_feed(self.input_name, batched_img)
        output_value = self.onnx_session.run(self.output_name, input_feed=input_feed)

        return output_value

    def post_process(self, output_values):
        pass

    @staticmethod
    def resize_img(img, resized_w, resized_h):
        h, w, _ = img.shape
        img = cv2.resize(img, (resized_w, resized_h))       # (width, height)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # BGR -> RGB

        return img.astype(np.float32)

    @staticmethod
    def normalize_img(batch_img, mean, std):
        mean = np.array(mean, dtype=np.float32).reshape((1, 1, 1, -1))
        std = np.array(std, dtype=np.float32).reshape((1, 1, 1, -1))

        normalize_img = (batch_img.astype(np.float32) / 255. - mean) / std

        return normalize_img

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def get_output_name(self):
        """
        output_name = onnx_session.get_outputs()[0].name
        """
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self):
        """
        input_name = onnx_session.get_inputs()[0].name
        """
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name


def run():
    root = tk.Tk()
    # LoginPage(root)
    MainPage(root)

    root.mainloop()


if __name__ == "__main__":
    run()
