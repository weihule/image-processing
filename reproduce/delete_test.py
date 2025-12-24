import numpy as np
import matplotlib.pyplot as plt
import threading
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def test():
    # 先把原函数复制过来（补全依赖）
    def resample_segments(segments, n=1000):
        for i, s in enumerate(segments):
            if len(s) == n:
                continue
            # 闭合线段（首尾衔接）
            s = np.concatenate((s, s[0:1, :]), axis=0)
            # 生成插值的目标横坐标
            x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
            xp = np.arange(len(s))
            # 升采样时保留原始点
            x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
            # 对x/y分别线性插值，拼接成(n,2)的坐标
            segments[i] = (
                np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32)
                .reshape(2, -1).T
            )
        return segments

    # -------------------------- 案例1：升采样（少点→多点） --------------------------
    # 原始线段：正方形，仅4个点（极简）
    square_4p = np.array([[0,0], [0,1], [1,1], [1,0]])
    # 重采样到20个点
    square_20p = resample_segments([square_4p], n=20)[0]

    # 可视化对比
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(square_4p[:,0], square_4p[:,1], c='red', s=50, label='原始4个点')
    plt.plot(square_4p[:,0], square_4p[:,1], 'r--', alpha=0.5)
    plt.title('原始：4个点的正方形')
    plt.axis('equal')
    plt.legend()

    plt.subplot(122)
    plt.scatter(square_20p[:,0], square_20p[:,1], c='blue', s=20, label='重采样20个点')
    plt.plot(square_20p[:,0], square_20p[:,1], 'b-', alpha=0.8)
    plt.title('重采样后：20个点的正方形')
    plt.axis('equal')
    plt.legend()
    plt.suptitle('案例1：升采样（补点，形状不变）', fontsize=12)
    plt.tight_layout()
    plt.show()

    # -------------------------- 案例2：降采样（多点→少点） --------------------------
    # 原始线段：正弦曲线，100个点（密集）
    x_org = np.linspace(0, 2*np.pi, 100)
    sin_100p = np.column_stack([x_org, np.sin(x_org)])
    # 重采样到20个点
    sin_20p = resample_segments([sin_100p], n=20)[0]

    # 可视化对比
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(sin_100p[:,0], sin_100p[:,1], c='red', s=10, label='原始100个点')
    plt.plot(sin_100p[:,0], sin_100p[:,1], 'r-', alpha=0.8)
    plt.title('原始：100个点的正弦曲线')
    plt.legend()

    plt.subplot(122)
    plt.scatter(sin_20p[:,0], sin_20p[:,1], c='blue', s=50, label='重采样20个点')
    plt.plot(sin_20p[:,0], sin_20p[:,1], 'b-', alpha=0.8)
    plt.title('重采样后：20个点的正弦曲线')
    plt.legend()
    plt.suptitle('案例2：降采样（精简点，形状基本不变）', fontsize=12)
    plt.tight_layout()
    plt.show()

    # -------------------------- 案例3：实际场景（轮廓统一点数） --------------------------
    # 模拟两个不规则轮廓：一个8个点，一个30个点
    contour1_8p = np.array([[1,1], [2,3], [4,4], [5,2], [4,0], [2,-1], [0,1], [1,1]])
    contour2_30p = np.column_stack([
        np.linspace(0, 6, 30),
        np.sin(np.linspace(0, 3*np.pi, 30)) + 1
    ])
    # 统一重采样到50个点
    contour1_50p = resample_segments([contour1_8p], n=50)[0]
    contour2_50p = resample_segments([contour2_30p], n=50)[0]

    # 可视化对比
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(contour1_8p[:,0], contour1_8p[:,1], 'r-', label='轮廓1（8个点）')
    plt.plot(contour2_30p[:,0], contour2_30p[:,1], 'g-', label='轮廓2（30个点）')
    plt.title('原始：点数不一致的两个轮廓')
    plt.axis('equal')
    plt.legend()

    plt.subplot(122)
    plt.plot(contour1_50p[:,0], contour1_50p[:,1], 'r-', label='轮廓1（50个点）')
    plt.plot(contour2_50p[:,0], contour2_50p[:,1], 'g-', label='轮廓2（50个点）')
    plt.title('重采样后：统一50个点的两个轮廓')
    plt.axis('equal')
    plt.legend()
    plt.suptitle('案例3：实际场景（统一所有轮廓的点数）', fontsize=12)
    plt.tight_layout()
    plt.show()


def test2():
    arr1 = np.array([[1, 2, 3], [4, 1, 0.5], [2, 0.5, 3]])
    print(arr1, arr1.shape)
    ret = np.argmin(arr1, axis=None)
    print(ret)
    ret_2d = np.unravel_index(ret, shape=arr1.shape)
    print(ret_2d)


def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"[The Log Function] {func.__name__} starts executing!")
        # Call the original function and obtain the return value
        result = func(*args, **kwargs)
        print(f"[The Log Function] {func.__name__} execution completed!")
        return result
    return wrapper


def log_decorator_with_prefix(prefix="默认前缀"):
    def outer(func):
        def wrapper(*args, **kwargs):
            print(f"[{prefix}] 函数 {func.__name__} 开始执行!")
            result = func(*args, **kwargs)
            print(f"[{prefix}] 函数 {func.__name__} 执行完成!")
            return result
        return wrapper
    return outer

# use decorator
@log_decorator_with_prefix()
def add(a, b):
    print(f"---")
    return a+b

@log_decorator_with_prefix(prefix="测试")
def add2(a, b):
    print(f"---")
    return a+b


def task(name, delay):
    print(f"线程 {name} 启动， 延迟  {delay} 秒")
    time.sleep(delay)
    print(f"线程 {name} 结束")


class MyThread(threading.Thread):
    def __init__(self, name, delay):
        super().__init__()
        self.name = name
        self.delay = delay
        
    def run(self):
        print(f"线程{self.name}启动，延迟{self.delay}秒")
        time.sleep(self.delay)
        print(f"线程{self.name}结束")

def test3():
    # # 创建线程
    # t1 = threading.Thread(target=task, args=("A", 2))
    # t2 = threading.Thread(target=task, args=("B", 5))

    # # 启动线程
    # t1.start()
    # t2.start()

    # # 等待线程结束
    # t1.join()
    # t2.join()

    # print("所有线程完成")

    t1 = MyThread("A", 2)
    t2 = MyThread("B", 5)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("所有线程执行完成")

def producer(queue):
    for i in range(3):
        queue.put(i)
        print(f"生产数据：{i}")
        
def consumer(queue):
    while not queue.empty():
        data = queue.get()
        print(f"消费数据：{data}")
        
def test4():
    queue = multiprocessing.Queue()    # 创建进程队列
    p1 = multiprocessing.Process(target=producer, args=(queue,))
    p2 = multiprocessing.Process(target=consumer, args=(queue,))
    p1.start()
    p1.join()    # 确保生产者生产完成
    p2.start()
    p2.join()


# 共享资源
count = 0
lock = threading.Lock()

def increment():
    global count
    for _ in range(10000000):
        with lock:
            count += 1
        
def test5():
    # 创建两个线程
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"最终count值: {count}") 


def increment2(count, lock):
    for _ in range(1000000):
        with lock:
            count.value += 1


def test6():
    # 进程间共享的数值（需用multiprocessing的Value）
    count = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()  # 创建进程锁
    p1 = multiprocessing.Process(target=increment2, args=(count, lock))
    p2 = multiprocessing.Process(target=increment2, args=(count, lock))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    print(f"最终count值: {count.value}")


import threading
import time

def threaded(func):
    def wrapper(*args, **kwargs):
        if kwargs.pop("threaded", True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            # thread.join()  # 如果打开这行，就是同步执行
            return thread
        else:
            return func(*args, **kwargs)
    return wrapper

@threaded
def weihltest():
    time.sleep(4)
    print("子线程执行完成")


def square(x):
    return x*x


def test8():
    nums = [1, 2, 3, 4, 5, 6]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(square, nums)
        for num, res in zip(nums, results):
            print(f"num={num} res={res}")


def test9():
    nums = [1, 2, 3, 4, 5, 6]
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_nums = {executor.submit(square, num): num for num in nums}
        for future in as_completed(future_nums):
            num = future_nums[future]
            res = future.result()
            print(f"num={num} res={res}")


def test7():
    # 调用1：异步（不加join）
    print("开始调用test")
    weihltest()
    print("主线程继续执行")  
    time.sleep(3)

    # 调用2：同步（加join）
    # print("开始调用test")
    # test()
    # print("主线程继续执行")  # 会等待2秒，先打印“子线程执行完成”，再打印这行


if __name__ == "__main__":
    # test2()
    # print(add(2, 3))
    # print(add2(4, 5))
    # test4()
    # test5()
    # test6()
    # test7()
    test8()
    test9()
    







