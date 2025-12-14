import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    test2()


