#!/usr/bin/env python3

from tkinter import *
from tkinter.ttk import Combobox
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageTk
from interp import RegularGridInterpolator

LOG_LINE_NUM = 0


class GUI():
    def __init__(self, main):
        self.main = main
        self.city_dict = self.get_city_dict()
        self.NOX = [0.5, 0.65, 0.8, 0.9, 1]
        self.VOC = [0.5, 0.65, 0.8, 0.9, 1]
        self.mat = None

    # 设置窗口
    def set_init_window(self):
        self.main.title("快速评估模型v0.1")
        w, h = 600, 700
        sw = self.main.winfo_screenwidth()
        sh = self.main.winfo_screenheight()
        x = (sw-w) // 2
        y = (sh-h) // 2
        self.main.geometry(f"{w}x{h}+{x}+{y}")
        self.main.resizable(False, False)

        # 选择城市标签
        self.sel_city_label = Label(self.main, text="选择城市", width=20, height=2)
        self.sel_city_label.grid(row=0, column=0, sticky=NSEW)

        # 选择城市
        self.sel_city = Combobox(self.main, text="选择城市")
        self.sel_city["value"] = list(self.city_dict.keys())
        self.sel_city.current(0)
        self.sel_city.bind("<<ComboboxSelected>>", self.reset_mat)
        self.sel_city.grid(row=0, column=1, columnspan=1, sticky=NSEW)

        self.sel_city_label = Label(self.main, text="NOx比例", height=2)
        self.sel_city_label.grid(row=1, column=0, sticky=NSEW)

        self.NOX_scale = Scale(self.main, orient=HORIZONTAL, length=200,
                               from_=0.5, to=1.0, resolution=0.01, variable=DoubleVar())
        self.NOX_scale.bind("<ButtonRelease-1>", self.NOX_change)
        self.NOX_scale.grid(row=1, column=1, sticky=NSEW)
        self.NOX_scale.set(1)

        self.sel_city_label = Label(self.main, text="VOC比例", height=3)
        self.sel_city_label.grid(row=2, column=0, sticky=NSEW)

        self.VOC_scale = Scale(self.main, orient=HORIZONTAL, length=200,
                               from_=0.5, to=1.0, resolution=0.01, variable=DoubleVar())
        self.VOC_scale.bind("<ButtonRelease-1>", self.VOC_change)
        self.VOC_scale.grid(row=2, column=1, sticky=NSEW)
        self.VOC_scale.set(1)

        # 按钮
        self.compute_button = Button(
            self.main, text="计算", bg="lightblue", command=self.compute)
        self.compute_button.grid(row=3, column=0, sticky=NS)

        self.img_label = Label(self.main, text="平均MDA8", height=2)
        self.img_label.grid(row=4, column=0, sticky=NSEW)

        self.result_data_Text = Text(self.main, width=30, height=1)  # 处理结果展示
        self.result_data_Text.grid(row=4, column=1, sticky=EW)

        # 按钮
        self.plot_button = Button(
            self.main, text="绘图", bg="lightblue", command=self.plot)
        self.plot_button.grid(row=5, column=0, sticky=NS)

        self.ekma_label = Label(self.main, text="EKMA", height=2)
        self.ekma_label.grid(row=6, column=0, rowspan=3, sticky=NSEW)

        self.log_label = Label(self.main, text="日志", height=2)
        self.log_label.grid(row=8, column=0, rowspan=2, sticky=NSEW)

        self.log_data_Text = Text(self.main, width=40, height=10)  # 日志框
        self.log_data_Text.grid(row=8, column=1, rowspan=2, sticky=NSEW)

    # 功能函数

    def NOX_change(self, e):
        self.write_log_to_Text(f"NOX比例为=>{self.NOX_scale.get()}")

    def VOC_change(self, e):
        self.write_log_to_Text(f"VOC比例为=>{self.VOC_scale.get()}")

    def reset_mat(self, e):
        self.write_log_to_Text(f"选择的城市为=>{self.sel_city.get()}")
        self.mat = None

    def get_city_dict(self):
        # self.write_log_to_Text("正在加载城市数据")
        with open("data/city.csv",encoding="utf8") as f:
            lines = f.readlines()
        city_dict = {}
        for line in lines:
            cells = line.split(",")
            city_dict[cells[0]] = {"LON": float(
                cells[1]), "LAT": float(cells[2])}
        return city_dict

    def get_ij(self):
        lon = np.loadtxt("data/lon.txt")
        lat = np.loadtxt("data/lat.txt")

        city = self.city_dict[self.sel_city.get()]
        self.write_log_to_Text(f"城市经纬度为=>LON:{city['LON']},LAT:{city['LAT']}")
        r2 = (lat - city['LAT'])**2 + (lon - city['LON'])**2
        return np.unravel_index(r2.argmin(), r2.shape)

    def load_mat(self):

        if not self.mat is None:
            return
        i, j = self.get_ij()
        self.write_log_to_Text(f"格点为i={i},j={j}")

        EKMA_MAT = np.zeros((len(self.NOX), len(self.VOC)))

        for index, element in np.ndenumerate(EKMA_MAT):
            case = f"nox_{str(self.NOX[index[0]])}_voc_{str(self.VOC[index[1]])}"
            case_mat = np.loadtxt(f"data/{case}.csv")
            EKMA_MAT[index] = case_mat[i, j]

        self.mat = EKMA_MAT

    def compute(self):
        self.load_mat()

        self.write_log_to_Text(f"NOx比例为=>{self.NOX_scale.get()}")
        self.write_log_to_Text(f"VOX比例为=>{self.VOC_scale.get()}")

        interpolator = RegularGridInterpolator(
            np.array((self.NOX, self.VOC)), self.mat)

        res = interpolator(
            np.array([self.NOX_scale.get(), self.VOC_scale.get()]))

        self.write_log_to_Text(f"计算结果为=>{res:.4f} ug/m3")
        self.result_data_Text.delete(1.0, END)
        self.result_data_Text.insert(1.0, f"{res:.4f} ug/m3")

    # 获取当前时间

    def get_current_time(self):
        current_time = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time

    def plot(self):
        self.load_mat()
        nox = self.NOX_scale.get()
        voc = self.VOC_scale.get()

        self.write_log_to_Text(f"正在绘制")

        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)

        X, Y = np.meshgrid(self.NOX, self.VOC,)
        ct = ax.contour(X, Y, self.mat)
        # plt.colorbar(ct)
        ax.set_xlabel("NOx")
        ax.set_ylabel("VOC")

        ax.scatter(nox,voc,marker="o", edgecolors="red", c=None)
        ax.clabel(ct, fontsize=10, colors='k', fmt="%.0f")
        plt.savefig(".ekma.png", dpi=200, bbox_inches='tight')
        self.write_log_to_Text(f"绘图完成,加载中")

        photo = Image.open(".ekma.png")  # file：t图片路径
        v = photo.resize((400, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(v)
        imgLabel = Label(self.main, image=photo, width=400,
                         height=300)  # 把图片整合到标签类中
        imgLabel.img = photo
        imgLabel.grid(row=6, column=1, rowspan=2)  # 自动对齐

    # 日志动态打印
    def write_log_to_Text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
        if LOG_LINE_NUM <= 9:
            self.log_data_Text.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0, 2.0)
            self.log_data_Text.insert(END, logmsg_in)


def gui_start():
    init_window = Tk()  # 实例化出一个父窗口
    ZMJ_PORTAL = GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()

gui_start()
