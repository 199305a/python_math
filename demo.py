import numpy as np

nb = 0
N = 1000000
for i in range(N):
    s = np.random.randint(0, 50, 3)
    fial = False
    for j in range(3):
        if s[j] < 4:
            fial = True
    if not fial:
        nb += 1

print("nb = %0.4f " % (nb / N,))


n = np.random.randint(0, 10, 100000)
h = np.bincount(n)
h

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from fldr import fldr_preprocess, fldr_sample

# 打开图像
image = Image.open(
    "/Users/cui/Downloads/flutter_samples-master/animations/assets/eat_cape_town_sm.jpg"
)

# 将图像转换为灰度图
gray_image = image.convert("L")

# 获取图像的像素值
pixels = list(gray_image.getdata())
pixels = np.array(pixels)
# 计算直方图
hist, bins = np.histogram(pixels.flatten(), bins=256, range=[0, 256])
# 转换为概率分布
prob_dist = hist / hist.sum()

# 将概率分布转换为整数权重
weights = (prob_dist * 10000).astype(int)  # 乘以一个合适的整数并转换为整数类型

# FLDR 算法预处理
preprocessed = fldr_preprocess(weights.tolist())

# 进行采样
num_samples = 1000
samples = [fldr_sample(preprocessed) for _ in range(num_samples)]

# 绘制原始直方图和采样结果的直方图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(256), prob_dist, color="gray")
plt.title("Original Image Histogram (Probability Distribution)")
plt.xlabel("Pixel Value")
plt.ylabel("Probability")

plt.subplot(1, 2, 2)
plt.hist(samples, bins=256, range=[0, 256], color="blue", density=True)
plt.title("Sampled Histogram using FLDR")
plt.xlabel("Pixel Value")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 样本数量
N = 10000000
# 分箱数量
num_bins = 100

# 均匀分布
t_uniform = np.random.random(N)
u_hist, u_bins = np.histogram(t_uniform, bins=num_bins)
u_prob = u_hist / u_hist.sum()

# 正态分布
t_normal = np.random.normal(0, 1, N)
n_hist, n_bins = np.histogram(t_normal, bins=num_bins)
n_prob = n_hist / n_hist.sum()

# 伽马分布
t_gamma = np.random.gamma(5.0, 1, N)
g_hist, g_bins = np.histogram(t_gamma, bins=num_bins)
g_prob = g_hist / g_hist.sum()

# 贝塔分布
t_beta = np.random.beta(5, 2, N)
beta_hist, beta_bins = np.histogram(t_beta, bins=num_bins)
beta_prob = beta_hist / beta_hist.sum()


# 创建画布
plt.figure(figsize=(10, 12))

# 绘制均匀分布直方图
plt.subplot(4, 1, 1)
plt.bar(u_bins[:-1], u_prob, width=(u_bins[1] - u_bins[0]), color="blue")
plt.title("Uniform Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")

# 绘制正态分布直方图
plt.subplot(4, 1, 2)
plt.bar(n_bins[:-1], n_prob, width=(n_bins[1] - n_bins[0]), color="blue")
plt.title("Normal Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")

# 绘制伽马分布直方图
plt.subplot(4, 1, 3)
plt.bar(g_bins[:-1], g_prob, width=(g_bins[1] - g_bins[0]), color="blue")
plt.title("Gamma Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")

# 绘制贝塔分布直方图
plt.subplot(4, 1, 4)
plt.bar(beta_bins[:-1], beta_prob, width=(beta_bins[1] - beta_bins[0]), color="blue")
plt.title("Beta Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

t_beta.mean()

m = []
for n in np.linspace(0, 8, 30):
    m.append(np.random.normal(1, 1, size=int(10**n)).mean())
plt.boxplot(m)
plt.title("折线图")
plt.xlabel("索引")
plt.ylabel("数值")
plt.show()


n = 100000

a = np.random.random(n)

(1 / n) * ((a - a.mean()) ** 2).sum()

(1 / (n - 1)) * ((a - a.mean()) ** 2).sum()

a = np.arange(10, dtype="float64")

a[3] = np.nan

np.isnan(a[3])


import numpy as np
import matplotlib.pyplot as plt

N = 1000
np.random.seed(73939133)
x = np.zeros((N, 4))
x[:, 0] = 5 * np.random.random(N)
x[:, 1] = np.random.normal(10, 1, size=N)
x[:, 2] = 3 * np.random.beta(5, 2, size=N)
x[:, 3] = 0.3 * np.random.lognormal(size=N)

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 0] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 1] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 2] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 3] = np.nan

# 绘制箱线图
import numpy as np
import matplotlib.pyplot as plt

N = 1000
np.random.seed(73939133)
x = np.zeros((N, 4))
x[:, 0] = 5 * np.random.random(N)
x[:, 1] = np.random.normal(10, 1, size=N)
x[:, 2] = 3 * np.random.beta(5, 2, N)
x[:, 3] = 0.3 * np.random.lognormal(size=N)

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 0] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 1] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 2] = np.nan

# i = np.random.randint(0, N, size=int(0.05 * N))
# x[i, 3] = np.nan

# 绘制箱线图
plt.boxplot(x)
plt.title("箱线图")
plt.xlabel("列索引")  # 说明是第几列的数据
plt.ylabel("数值")
plt.ylim(0, 15)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

N = 1000
np.random.seed(73939133)
x = np.zeros((N, 4))
x[:, 0] = 5 * np.random.random(N)
x[:, 1] = np.random.normal(10, 1, size=N)
x[:, 2] = 3 * np.random.beta(5, 2, size=N)
x[:, 3] = 0.3 * np.random.lognormal(size=N)

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 0] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 1] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 2] = np.nan

i = np.random.randint(0, N, size=int(0.05 * N))
x[i, 3] = np.nan

# mask = ~np.isnan(x)
# filtered_data = [d[m] for d, m in zip(x.T, mask.T)]
font = {"family": "MicroSoft YaHei", "weight": "bold", "size": 20}
plt.rc("font", **font)

good_index = np.where(np.isnan(x[:, 0]) == False)

# 绘制箱线图
plt.boxplot(filtered_data)
plt.title("箱线图")
plt.xlabel("列索引")  # 说明是第几列的数据
plt.ylabel("数值")
plt.show()
