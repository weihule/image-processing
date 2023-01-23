import os
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory
import tkinter as tk
from tkinter import messagebox, ttk
import json
from PIL import Image, ImageTk
# import matplotlib.pyplot as plt
import onnxruntime
import glob


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

        self.model_name = tk.StringVar()  # 模型名称
        self.gallery_name = tk.StringVar()  # 搜索库名称
        self.weights = tk.StringVar()  # 权重文件
        self.src_pic = tk.StringVar()  # probe图片
        self.gallery_dir = tk.StringVar()  # 搜索文件夹
        self.save_dir = tk.StringVar()  # 保存路径

        self.create_page()

    def create_page(self):
        ttk.Label(self.login_page, text="选择模型: ").place(x=50, y=10)
        # ttk.Entry(self.login_page, textvariable=self.model_name, width=40).place(x=135, y=20)
        # ttk.Button(self.login_page, text="点击确定", command=self.confirm_model).place(x=440, y=18)
        com = ttk.Combobox(self.login_page, textvariable=self.model_name,
                           width=25,
                           values=("osnet_ibn_x1_0_origin",
                                   "osnet_x1_0",
                                   "osnet_x0_75")
                           )
        com.place(x=135, y=10)
        com.current(0)  # 设定下拉菜单的默认值为第0个

        ttk.Label(self.login_page, text="搜索库名称: ").place(x=400, y=10)
        gallery_name_com = ttk.Combobox(self.login_page,
                                        textvariable=self.gallery_name,
                                        width=25,
                                        values=("market1501",
                                                "duke",
                                                "msmt17",
                                                "mine")
                                        )
        gallery_name_com.place(x=485, y=10)
        gallery_name_com.current(0)  # 设定下拉菜单的默认值为第0个

        ttk.Label(self.login_page, text="权重文件路径: ").place(x=50, y=40)
        ttk.Entry(self.login_page, textvariable=self.weights, width=50).place(x=135, y=40)
        ttk.Button(self.login_page, text="点击选择", command=self.open_file).place(x=500, y=38)

        ttk.Label(self.login_page, text="搜索库路径: ").place(x=50, y=80)
        ttk.Entry(self.login_page, textvariable=self.gallery_dir, width=50).place(x=135, y=80)
        ttk.Button(self.login_page, text="点击选择", command=self.open_dir).place(x=500, y=78)

        ttk.Label(self.login_page, text="保存路径: ").place(x=50, y=120)
        ttk.Entry(self.login_page, textvariable=self.save_dir, width=50).place(x=135, y=120)
        ttk.Button(self.login_page, text="点击选择", command=self.save_dir_path).place(x=500, y=118)

        ttk.Button(self.login_page, text="选择图片", command=self.open_image).place(x=50, y=150)
        ttk.Button(self.login_page, text="开始重识别", command=self.start_infer).place(x=150, y=150)
        ttk.Button(self.login_page, text="退出系统", command=self.master.destroy).place(x=250, y=150)

        # 图片排列控件
        ttk.Label(self.login_page, text="query", font=("Times New Roman", 20)).place(x=50, y=180)
        # ttk.Label(self.login_page, text="1", font=("宋体", 25)).place(x=250, y=180)
        # ttk.Label(self.login_page, text="2", font=("宋体", 25)).place(x=400, y=180)
        ttk.Label(self.login_page, text="gallery", font=("Times New Roman", 20)).place(x=550, y=180)
        # ttk.Label(self.login_page, text="4", font=("宋体", 25)).place(x=700, y=180)
        # ttk.Label(self.login_page, text="5", font=("宋体", 25)).place(x=850, y=180)
        

    def open_file(self):
        file = askopenfilename(title="请选择权重文件",
                            #    initialdir="D:\\Desktop\\tempfile\\weights",
                               filetypes=[("onnx文件", ".onnx")])
        # 将file赋值给self.weights
        self.weights.set(file)

    def open_dir(self):
        file = askdirectory(title="请选择图片库文件夹",
                            # initialdir="D:\\workspace\\data\\dl\\reid"
                            )
        # 将file赋值给self.gallery_dir
        self.gallery_dir.set(file)

    def save_dir_path(self):
        file = askdirectory(title="保存路径",
                            # initialdir="D:\\Desktop\\tempfile\\some_infer"
                            )
        self.save_dir.set(file)

    def open_image(self):
        # paned = tk.PanedWindow(self.login_page)
        # paned.place(x=40, y=180)
        # paned.pack(fill=tk.X, side=tk.LEFT)
        file = askopenfilename(title="请选择图片文件",
                            #    initialdir="D:\\workspace\\data\dl\\reid",
                               filetypes=[("图片文件", ".jpg"),
                                          ("图片文件", ".png"),
                                          ("图片文件", ".JPEG"),
                                          ("图片文件", ".PNG")])
        self.src_pic.set(file)
        # img = Image.open(file)
        # self.login_page.photo = ImageTk.PhotoImage(img.resize((128, 256)))
        # tk.Label(self.login_page, image=self.login_page.photo).place(x=10, y=220)

        paned = tk.PanedWindow(self.login_page)
        paned.place(x=10, y=220)
        img = Image.open(file)
        paned.photo = ImageTk.PhotoImage(img.resize((128, 256)))  # 改变图片显示大小
        ttk.Label(self.login_page, image=paned.photo).place(x=10, y=220)

    def start_infer(self):
        file = self.weights.get()
        onnx_infer = OnnxInfer(onnx_file=file, save_dir=self.save_dir.get())
        onnx_infer.infer(probe_path=self.src_pic.get(),
                         gallery_dir=self.gallery_dir.get(),
                         gallery_data_name=self.gallery_name.get())
        # 将重识别之后的行人图像展示在页面上
        save_path = os.path.join(self.save_dir.get(), self.src_pic.get().split("/")[-1][:-3] + "txt")
        with open(save_path, 'r', encoding="utf-8") as fr:
            lines = fr.readlines()
        # print(lines)
        for idx, line in enumerate(lines):
            line = line.strip()
            paned = tk.PanedWindow(self.login_page)
            paned.place(x=10, y=220)
            img = Image.open(line)
            paned.photo = ImageTk.PhotoImage(img.resize((128, 256)))  # 改变图片显示大小
            ttk.Label(self.login_page, image=paned.photo).place(x=(idx+1)*170, y=220)


class OnnxInfer(object):
    def __init__(self, onnx_file, save_dir, resized_w=128, resized_h=256, batch_size=6):
        self.onnx_file = onnx_file
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.batch_size = batch_size

        self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        self.input_name = [self.onnx_session.get_inputs()[0].name]
        self.output_name = [self.onnx_session.get_outputs()[0].name]

        self.save_dir = save_dir

    def infer(self, probe_path, gallery_dir, gallery_data_name):
        prepares = self.prepare(probe_path, gallery_dir, gallery_data_name)
        self.post_process(prepares=prepares)

    def prepare(self, probe_path, gallery_dir, gallery_data_name):
        q_img, q_pid, q_camid = self._process_probe(probe_path, gallery_data_name)
        input_feed = self.get_input_feed(self.input_name, q_img)
        q_fs = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        q_pid = np.array([q_pid])
        q_camid = np.array([q_camid])

        datasets = self._process_dir(gallery_dir, gallery_data_name)
        datasets_batches = [datasets[i: i + self.batch_size] for i in range(0, len(datasets), self.batch_size)]
        g_fs, g_pids, g_camids = [], [], []
        for batch_info in datasets_batches:
            batch_img = []
            # 开始整合一个batch中的数据
            for per_info in batch_info:
                g_img_path, g_pid, g_camid = per_info
                img = self.resize_img(g_img_path)
                batch_img.append(img)
                g_pids.append(g_pid)
                g_camids.append(g_camid)
            batch_img = np.stack(batch_img, axis=0)  # [B, 256, 128, 3]
            batch_img = self.normalize_img(batch_img)

            input_feed = self.get_input_feed(self.input_name, batch_img)
            g_features = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
            g_fs.append(g_features)
        g_fs = np.concatenate(g_fs, axis=0)  # [num_gallery, feature_maps]
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        q_fs = 1. * q_fs / (np.linalg.norm(q_fs, ord=2, axis=-1, keepdims=True) + 1e-12)
        g_fs = 1. * g_fs / (np.linalg.norm(g_fs, ord=2, axis=-1, keepdims=True) + 1e-12)

        m, n = q_fs.shape[0], g_fs.shape[0]
        # dis_mat shape is [m, n]
        dis_mat = np.power(q_fs, 2).sum(axis=1, keepdims=True).repeat(n, axis=1) + \
                  np.power(g_fs, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
        dis_mat = dis_mat - 2 * q_fs @ g_fs.T

        return (probe_path, datasets, dis_mat, q_pid, g_pids, q_camid, g_camids)

    def post_process(self, prepares):
        # save_dir = "D:\\Desktop\\tempfile\\some_infer\\reid_infer"
        probe_path, dataset, dist_mat, q_pids, g_pids, q_camids, g_camids = prepares
        max_rank = 5
        num_q, num_g = dist_mat.shape
        if num_g < max_rank:
            max_rank = num_g
            print(f"Note: number of gallery samples is quite small, got {num_q}")
        # indices: [num_q, num_g] 输出按行排列的索引 (升序，从小到大)
        indices = np.argsort(dist_mat, axis=1)
        # g_pids[indices] shape is [m, n]
        # g_pids 原来是 [n, ], g_pids[indices]操作之后, 扩展到了 [m, n]
        # 也就是每一行中的n个元素都按照 indices 中每一行的顺序进行了重排列
        g_pids_exp_dims, g_camids_exp_dims = g_pids[indices], g_camids[indices]
        g_pids_indices_sorted = np.ones_like(g_pids_exp_dims, dtype=np.int32) * (-1)
        q_pids_exp_dims = np.expand_dims(q_pids, axis=1)  # [m, 1]

        # matches中为 1 的表示query中和gallery中的行人id相同, 也就是表示同一个人
        # matches中的结果就是和后续预测出的结果进行对比的正确label
        matches = (g_pids_exp_dims == q_pids_exp_dims).astype(np.int32)  # shape is [m, n]

        for q_idx in range(num_q):
            # q_pid, q_camid 分别都是一个数字
            q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            # TODO: 这里要用 & ,因为前后都是np.ndarray类型, 如果前后都是list, 则可以使用 and
            removed = (g_pids_exp_dims[q_idx] == q_pid) & (g_camids_exp_dims[q_idx] == q_camid)  # [n, ]

            # keep中为True的表示gallery中符合查找条件的行人图片，
            # 这些为True的部分还需要借助matches才能完成正确的查找
            # 且keep中从左到右就是当前查找图片和每一个gallery中图片的距离距离依次递增的顺序
            keep = np.where(removed == 0, True, False)  # [n, ]

            # orig_cmc中为1的位置表示查找的图片匹配正确了
            orig_cmc = matches[q_idx][keep]

            # orig_cmc中为1即表示匹配正确，为0表示匹配错误, 但是orig_cmc经过keep这个mask之后，数量就发生了变化
            g_pids_indices_sorted[q_idx][:len(orig_cmc)] = indices[q_idx][keep]

        for idx in range(num_q):
            # q_pid = q_pids[idx]
            # fig = plt.figure(figsize=(16, 4))
            # ax = plt.subplot(1, 11, 1)
            # ax.axis('off')
            # img = plt.imread(probe_path)
            # plt.title("query")
            # plt.imshow(img)

            # 没有在gallery中找到匹配的行人
            if (g_pids_indices_sorted[idx] == -1).all():
                print('no matched in gallery')
                continue

            # for i, g_pid_idx in enumerate(g_pids_indices_sorted[idx][:10]):
            #     ax = plt.subplot(1, 11, i + 2)
            #     ax.axis('off')
            #     g_path = dataset[int(g_pid_idx)][0]
            #     g_pid = dataset[int(g_pid_idx)][1]
            #     img = plt.imread(g_path)
            #     plt.title(str(g_pid_idx))
            #     plt.imshow(img)

            #     if g_pid == q_pid:
            #         ax.set_title('%d' % (i + 1), color='green')
            #     else:
            #         ax.set_title('%d' % (i + 1), color='red')
            # save_path = os.path.join(self.save_dir, probe_path.split("/")[-1])
            # fig.savefig(save_path)
            # plt.close()
            save_path = os.path.join(self.save_dir, probe_path.split("/")[-1][:-3] + "txt")
            with open(save_path, "w", encoding="utf-8") as fa:
                for g_pid_idx in g_pids_indices_sorted[idx][:5]:
                    g_path = dataset[int(g_pid_idx)][0]
                    fa.write(g_path + "\n")

    def _process_probe(self, probe_path, gallery_data_name):
        """处理probe图片信息"""
        img = self.resize_img(probe_path, resized_w=128, resized_h=256)
        img = np.expand_dims(img, axis=0)  # [1, H, W, C]
        img = self.normalize_img(img)  # [1, C, H, W]
        probe_name = probe_path.split("/")[-1]
        if gallery_data_name == "market1501":
            pid, camid = int(probe_name.split('_')[0]), int(probe_name.split('_')[1][1]) - 1
        elif gallery_data_name == "duke":
            pid, camid = -1, -1
        elif gallery_data_name == "msmt17":
            pid, camid = int(probe_name.split('_')[0]), int(probe_name.split('_')[1][1:]) - 1
        else:
            pid, camid = -1, -1

        return img, pid, camid

    @staticmethod
    def _process_dir(gallery_dir, gallery_data_name):
        """处理gallery文件夹信息"""
        datasets = list()
        img_paths = glob.glob(os.path.join(gallery_dir, '*.jpg'))

        if gallery_data_name == "market1501":
            for img_path in img_paths:
                img_name = img_path.split(os.sep)[-1]
                infos = img_name.split('_')
                pid, camid = int(infos[0]), int(infos[1][1])
                # ignore junk image
                if pid == -1:
                    continue
                assert 0 <= pid <= 1501
                assert 1 <= camid <= 6
                camid -= 1  # camera id start from 0
                datasets.append([img_path, pid, camid])
        elif gallery_data_name == "duke":
            pass
        elif gallery_data_name == "msmt17":
            for img_path in img_paths:
                img_name = img_path.split(os.sep)[-1]
                infos = img_name.split('_')
                pid, camid = int(infos[0]), int(infos[1][1:])
                # ignore junk image
                if pid == -1:
                    continue
                assert 0 <= pid <= 4101
                assert 1 <= camid <= 15
                camid -= 1  # camera id start from 0
                datasets.append([img_path, pid, camid])
        else:
            pass
        return datasets

    @staticmethod
    def resize_img(img_path, resized_w=128, resized_h=256):
        # 这种写法支持opencv读取中文路径
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        img = cv2.resize(img, (resized_w, resized_h))  # (width, height)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        return img.astype(np.float32)

    @staticmethod
    def normalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        标准化
        Args:
            img: [B, H, W, C]
            mean: Tuple or List
            std: Tuple or List
        Returns:
        """
        mean = np.array(mean, dtype=np.float32).reshape((1, 1, 1, -1))
        std = np.array(std, dtype=np.float32).reshape((1, 1, 1, -1))

        img = (img / 255. - mean) / std

        img = img.transpose((0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]

        return img

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed


def run():
    root = tk.Tk()
    LoginPage(root)
    # MainPage(root)

    root.mainloop()

# import matplotlib.pyplot as plt
# import random

# def plt_test():
#     xs = [i for i in range(10)]
#     y1s = [0.1, 0.2, 0.3, 0.25, 0.31, 0.33, 0.4, 0.5, 0.55, 0.86]
#     y2s = [0.05, 0.15, 0.35, 0.32, 0.43, 0.45, 0.57, 0.61, 0.65, 0.79]
    
#     plt.plot(xs, y1s, label="osnet_1_0")
#     plt.plot(xs, y2s, label="osnet_0_75")
#     plt.legend(loc="lower right")
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.show()


if __name__ == "__main__":
    run()
    # plt_test()
    # onnx_infer = OnnxInfer(onnx_file="D:\\Desktop\\tempfile\\weights\\osnet_ibn_x1_0_onnx.onnx")
    # p1 = "D:\\workspace\\data\\dl\\reid\\demo\\market1501\\query\\0002_c3s1_000076_01.jpg"
    # p2 = "D:\\workspace\\data\\dl\\reid\\demo\\market1501\\gallery"
    # name = "market1501"
    # onnx_infer.infer(probe_path=p1,
    #                  gallery_dir=p2,
    #                  gallery_data_name=name)
