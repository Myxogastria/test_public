# testtest

# リモートコントロールのテスト
1+1

1+3

1+5

2+6
2+6


import matplotlib.pyplot as plt
from sklearn import datasets

# irisデータセット読み込み
iris = datasets.load_iris()
data = iris.data
species = iris.target

# とりあえずプロット
plt.scatter(data[:,0],data[:,1])
plt.show()

# 種ごとに色を変える
x = iris['data']
y = iris['target']
x = data
y = species
x = iris.data
y = iris.target
colname = iris['feature_names']

import itertools

plt.close('all')
plt.figure(1)
subplotstart = 321
colnumbers = range(0,4)
colpairs = itertools.combinations(colnumbers,2)

# i = next(colpairs)
# print(i)

# for i in colpairs:
#     print(i)

for i in colpairs:
    plt.subplot(subplotstart)
    plt.scatter(x[:50,i[0]],x[:50,i[1]],label='setosa') 
    plt.scatter(x[50:100,i[0]],x[50:100,i[1]],label='versicolor')
    plt.scatter(x[100:,i[0]],x[100:,i[1]],label='virginica')
    plt.xlabel(colname[i[0]])
    plt.ylabel(colname[i[1]])
    plt.legend()

    subplotstart += 1

plt.show(block=False)


import numpy as np
a = np.random.randint(0, 150)
data[a]

# petal_lengthとpetal_widthでvirginicaなのかどうかを判定
# petal_lengthとpetal_widthのバイアス付きの線形和を使ってうまく判定しよう

import tensorflow as tf

# 重み
w = tf.Variable(initial_value=np.ones(3))

# データとバイアス(1)
datax = np.concatenate([data[:, 2:4], np.ones([data.shape[0], 1])], 1)

# 繰り返し
for i in range(1000):
    # 目的関数の勾配を計算するためにGradientTapeを使う
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        tape.watch(w)

        # wを使って目的関数を計算

        # バイアス付き線形和
        f = tf.tensordot(w, datax, [0, 1])

        # sigmoid
        g = tf.sigmoid(1*f)

        # 答え：virginicaだったら1、そうでなければ0
        answer = (species == 2) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse = tf.reduce_sum((g - answer) ** 2)

    # 勾配を計算
    grad = tape.gradient(sse, w)

    # wを勾配を使って変更
    w.assign_sub(0.005*grad)

print(w.numpy())


print(g.numpy())

print(f.numpy())
print(sse.numpy())

# test of gradient
testw = tf.Variable(initial_value=1.0) # 1.0にしとかないとだめ
with tf.GradientTape() as testtape:
    testtape.watch(testw)

    testf = 2*testw

print(testtape.gradient(testf, testw)) # 2.0

# https://www.tensorflow.org/api_docs/python/tf/GradientTape#watch
# By default GradientTape will automatically watch any trainable variables 
# that are accessed inside the context.
# tf.Variableを使う限りはtape.watchは必要ではない。
x = tf.Variable(1.0)
y = tf.Variable(2.0)
with tf.GradientTape() as tape:
    #tape.watch(x)
    f = x*y/(x+y)
print(tape.gradient(f, [x,y]))

x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    # type(x)
    f = x[0]*x[1]/(x[0]+x[1])
print(tape.gradient(f, x))


init = np.ones(10)
init = np.array([float(i) for i in range(10)])  # リスト内包表記
init = np.array([float(i)*(i > 8) for i in range(10)])  # リスト内包表記
# init = np.array([0.00000001,0,0,0,0,0,0,0,0,0],dtype=float)

init = np.array([float(i > 1) for i in range(3)])  # リスト内包表記
x = tf.Variable(init)
with tf.GradientTape() as tape:
    f = tf.math.reduce_std(x)**2

# 全部同じ要素からなるベクトルだと結果がnanになる
# 分散なら0
print(tape.gradient(f,x))

w1 = tf.Variable(initial_value=np.ones(3))

for i in range(1000):
    # 目的関数の勾配を計算するためにGradientTapeを使う
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        tape.watch(w1)

        # wを使って目的関数を計算

        # バイアス付き線形和
        f1 = tf.tensordot(w1, datax, [0, 1])

        # sigmoid
        g1 = tf.sigmoid(1*f1)

        # 答え：setosaだったら1、そうでなければ0
        answer1 = (species == 0) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse1 = tf.reduce_sum((g1 - answer1) ** 2)

    # 勾配を計算
    grad1 = tape.gradient(sse1, w1)

    # wを勾配を使って変更
    w1.assign_sub(0.005*grad1)

print(w1.numpy())


print(g1.numpy())

print(f1.numpy())
print(sse1.numpy())


"""
import pandas as pd

df = pd.DataFrame()
df["a"] = data[:,1]
df["b"] = data[:,2]
"""

w2 = tf.Variable(initial_value=np.ones(3))

np.ones(data.shape[0]).shape

g.numpy().reshape((-1, 1)).shape
datay = np.concatenate((g.numpy().reshape((-1, 1)), g1.numpy().reshape((-1, 1)), np.ones((data.shape[0], 1))), axis=1)
datay = np.stack((g.numpy(), g1.numpy(), np.ones(data.shape[0]))).T

for i in range(1000):
    # 目的関数の勾配を計算するためにGradientTapeを使う
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        tape.watch(w2)

        # wを使って目的関数を計算

        # バイアス付き線形和
        f2 = tf.tensordot(w2, datay, [0, 1])

        # sigmoid
        g2 = tf.sigmoid(1*f2)

        # 答え：versicolorだったら1、そうでなければ0
        answer2 = (species == 1) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse2 = tf.reduce_sum((g2 - answer2) ** 2)

    # 勾配を計算
    grad = tape.gradient(sse2, w2)

    # wを勾配を使って変更
    w2.assign_sub(0.005*grad)

print(w2.numpy())


print(g2.numpy())

print(f2.numpy())
print(sse2.numpy())


# save previous results for later comparison
w0_separate_method = w1.numpy()
w2_separate_method = w.numpy()
w1_separate_method = w2.numpy()
sse_separate_method = sse2.numpy()

# update all weights at one iteration
# setosa=0; versicolor=1; virginica=2
# d0 = w0[0]*datax[,0] + w0[1]*datax[,1] + w0[2]
# d2 = w2[0]*datax[,0] + w2[1]*datax[,1] + w2[2]
# d1 = w1[0]*d2 + w1[1]*d0 + w1[2]
w0 = tf.Variable(initial_value=np.ones(3))
w1 = tf.Variable(initial_value=np.ones(3))
w2 = tf.Variable(initial_value=np.ones(3))

N = 1000

sse_x = np.ones(N)

for i in range(N):
    # 目的関数の勾配を計算するためにGradientTapeを使う
    
    # weight of setosa
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        #tape.watch(w2)

        # wを使って目的関数を計算
        
        # バイアス付き線形和
        f0 = tf.tensordot(w0, datax, [0, 1])

        # sigmoid
        g0 = tf.sigmoid(1*f0)

        # 答え：setosaだったら1、そうでなければ0
        answer0 = (species == 0) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse0 = tf.reduce_sum((g0 - answer0) ** 2)
    # 勾配を計算
    grad0 = tape.gradient(sse0, w0)

    # wを勾配を使って変更
    w0.assign_sub(0.005 * grad0)
    
    # weight of virginica
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        #tape.watch(w2)

        # wを使って目的関数を計算
        
        # バイアス付き線形和
        f2 = tf.tensordot(w2, datax, [0, 1])

        # sigmoid
        g2 = tf.sigmoid(1*f2)

        # 答え：virginicaだったら1、そうでなければ0
        answer2 = (species == 2) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse2 = tf.reduce_sum((g2 - answer2) ** 2)
    # 勾配を計算
    grad2 = tape.gradient(sse2, w2)

    # wを勾配を使って変更
    w2.assign_sub(0.005 * grad2)
    
    # function for versicolor
    with tf.GradientTape() as tape:
        # 動かす変数を指定
        #tape.watch(w2)

        # wを使って目的関数を計算
        
        # create datay with previously estimated g0 and g2
        datay = np.concatenate((g2.numpy().reshape((-1, 1)), g0.numpy().reshape((-1, 1)), np.ones((data.shape[0], 1))), axis=1)
        
        # バイアス付き線形和
        f1 = tf.tensordot(w1, datay, [0, 1])

        # sigmoid
        g1 = tf.sigmoid(1*f1)

        # 答え：versicolorだったら1、そうでなければ0
        answer1 = (species == 1) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse1 = tf.reduce_sum((g1 - answer1) ** 2)
    # 勾配を計算
    grad1 = tape.gradient(sse1, w1)

    sse_x[i] = sse1.numpy()
    # wを勾配を使って変更
    w1.assign_sub(0.005 * grad1)

plt.plot(sse_x)
plt.show(block=False)

# compare both methods
w0_combined_method = w0.numpy()
w2_combined_method = w2.numpy()
w1_combined_method = w1.numpy()

sse_combined_method = sse1.numpy()

print('w0_combined_method=' + str(w0_combined_method))
print('w2_combined_method=' + str(w2_combined_method))
print('w1_combined_method=' + str(w1_combined_method))

print('w0_separate_method=' + str(w0_separate_method))
print('w2_separate_method=' + str(w2_separate_method))
print('w1_separate_method=' + str(w1_separate_method))

print('sse_combined_method=' + str(sse_combined_method))
print('sse_separate_method=' + str(sse_separate_method))

w0_combined = w0
w1_combined = w1
w2_combined = w2

# update all weights at one iteration
# d0 = w0[0]*datax[,0] + w0[1]*datax[,1] + w0[2]
# d2 = w2[0]*datax[,0] + w2[1]*datax[,1] + w2[2]
# d1 = w1[0]*sigmoid(d2) + w1[1]*sigmoid(d0) + w1[2]
# d1 = w1[0]*datax[,0] + w1[1]*datax[,1] + w1[2]
w0 = tf.Variable(initial_value=-np.ones(3))
w1 = tf.Variable(initial_value=np.ones(3))
w2 = tf.Variable(initial_value=np.ones(3))

# w0 = tf.Variable(initial_value=w0_combined.numpy())
# w1 = tf.Variable(initial_value=w1_combined.numpy())
# w2 = tf.Variable(initial_value=w2_combined.numpy())

N = 1000

sse_x = np.ones(N)

for i in range(N):
    # 目的関数の勾配を計算するためにGradientTapeを使う
    
    # weight of setosa
    with tf.GradientTape() as tape:

        # wを使って目的関数を計算
        
        # バイアス付き線形和
        f0 = tf.tensordot(w0, datax, [0, 1])

        # sigmoid
        g0 = tf.sigmoid(1*f0)

        # バイアス付き線形和
        f2 = tf.tensordot(w2, datax, [0, 1])

        # sigmoid
        g2 = tf.sigmoid(1*f2)

        # create datay with previously estimated g0 and g2
        # datay = np.concatenate((g2.numpy().reshape((-1, 1)), g0.numpy().reshape((-1, 1)), np.ones((data.shape[0], 1))), axis=1)
        
        # バイアス付き線形和
        # f1 = tf.tensordot(w1, datay, [0, 1])
        f1 = w1[0]*g2 + w1[1]*g0 + w1[2]
        
        # sigmoid
        g1 = tf.sigmoid(1*f1)

        # 答え：versicolorだったら1、そうでなければ0
        answer1 = (species == 1) * 1

        # sigmoidの出力と答えの誤差を計算
        # squared sum of error （目的関数）
        sse1 = tf.reduce_sum((g1 - answer1) ** 2)
    # 勾配を計算
    grad = tape.gradient(sse1, [w0, w1, w2])
    # grad1 = tape.gradient(sse1, w1)
    # grad2 = tape.gradient(sse1, w2)

    # wを勾配を使って変更
    w0.assign_sub(0.05 * grad[0])
    w1.assign_sub(0.05 * grad[1])
    w2.assign_sub(0.05 * grad[2])

    sse_x[i] = sse1.numpy()

plt.plot(sse_x)
plt.show(block=False)


# compare both methods
w0_combined_method = w0.numpy()
w2_combined_method = w2.numpy()
w1_combined_method = w1.numpy()

sse_combined_method = sse1.numpy()

print('w0_combined_method=' + str(w0_combined_method))
print('w2_combined_method=' + str(w2_combined_method))
print('w1_combined_method=' + str(w1_combined_method))

print('w0_separate_method=' + str(w0_separate_method))
print('w2_separate_method=' + str(w2_separate_method))
print('w1_separate_method=' + str(w1_separate_method))

print('sse_combined_method=' + str(sse_combined_method))
print('sse_separate_method=' + str(sse_separate_method))


import time

# 学習途中の結果を可視化する
plt.figure()

feature_use = (2, 3)

plt.clf()
for i, i_species in enumerate(iris.target_names):
    plt.scatter(iris.data[iris.target == i, feature_use[0]], iris.data[iris.target == i, feature_use[1]], label=i_species)
plt.xlabel(iris.feature_names[feature_use[0]])
plt.ylabel(iris.feature_names[feature_use[1]])
plt.legend()
plt.show(block=False)

# 入力データ
x = np.concatenate([iris.data[:, feature_use], np.ones([iris.data.shape[0], 1])], 1)

# ウェイト
w_00 = tf.Variable(initial_value=np.array([1.0, 1, -2]))
w_01 = tf.Variable(initial_value=np.array([1.0, 1, -7]))
w_1 = tf.Variable(initial_value=np.ones(3))

# 境界線はf=0として引く．つまり，各ウェイトに対して
# w[0]*x + w[1]*y + w[2] = 0

def abline(slope, intercept, fmt='-', c=None):
    if isinstance(slope, tf.Tensor):
        slope = slope.numpy()
    if isinstance(intercept, tf.Tensor):
        intercept = intercept.numpy()

    axes = plt.gca()
    x_lim0 = axes.get_xlim()
    y_lim0 = axes.get_ylim()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, fmt, c=c)
    plt.xlim(x_lim0)
    plt.ylim(y_lim0)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
abline(-w_00[0]/w_00[1], -w_00[2], c=default_colors[0])
abline(-w_01[0]/w_01[1], -w_01[2], c=default_colors[2])
plt.show(block=False)

# プロット用関数
def plot_with_lines(w_00, w_01):
    for i, i_species in enumerate(iris.target_names):
        plt.scatter(iris.data[iris.target == i, feature_use[0]], iris.data[iris.target == i, feature_use[1]], label=i_species)
    plt.xlabel(iris.feature_names[feature_use[0]])
    plt.ylabel(iris.feature_names[feature_use[1]])
    plt.legend()
    # plt.show(block=False)

    # 境界線はf=0として引く．つまり，各ウェイトに対して
    # w[0]*x + w[1]*y + w[2] = 0
    abline(-w_00[0]/w_00[1], -w_00[2]/w_00[1], c=default_colors[0])
    abline(-w_01[0]/w_01[1], -w_01[2]/w_01[1], c=default_colors[2])


# 学習

# 入力データ
x = np.concatenate([iris.data[:, feature_use], np.ones([iris.data.shape[0], 1])], 1)


# それぞれsetosa, versicolorの境界線として学習させる
fig = plt.figure()

# ウェイト
# w_00 = tf.Variable(initial_value=-np.array([1.0, 1, -2]))
# w_01 = tf.Variable(initial_value=np.array([1.0, 1, -7]))
w_00 = tf.Variable(initial_value=-np.ones(3)) # sigmoidの方向的にマイナスにしておくとうまくいく
w_01 = tf.Variable(initial_value=np.ones(3))
w_1 = tf.Variable(initial_value=np.ones(3))

hist_sse_00 = []
hist_sse_01 = []
hist_sse_1 = []

eta = 0.005
x_lim0 = [-1, 8]
y_lim0 = [-1, 3]

for i in range(1000):
#     pass
# for i in range(1):
    with tf.GradientTape() as tape_00, tf.GradientTape() as tape_01, tf.GradientTape() as tape_1:
        f_00 = tf.tensordot(w_00, x, [0, 1])
        g_00 = tf.sigmoid(f_00)
        target_00 = (iris.target == 0)
        sse_00 = tf.reduce_sum((g_00 - target_00) ** 2)

        f_01 = tf.tensordot(w_01, x, [0, 1])
        g_01 = tf.sigmoid(f_01)
        target_01 = (iris.target == 2)
        sse_01 = tf.reduce_sum((g_01 - target_01) ** 2)

        f_1 = w_1[0]*g_00 + w_1[1]*g_01 + w_1[2]
        g_1 = tf.sigmoid(f_1)
        target_1 = (iris.target == 1)
        sse_1 = tf.reduce_sum((g_1 - target_1) ** 2)

    hist_sse_00.append(sse_00.numpy())
    hist_sse_01.append(sse_01.numpy())
    hist_sse_1.append(sse_1.numpy())

    grad_00 = tape_00.gradient(sse_00, w_00)
    w_00.assign_sub(eta*grad_00)
    grad_01 = tape_01.gradient(sse_01, w_01)
    w_01.assign_sub(eta*grad_01)
    grad_1 = tape_1.gradient(sse_1, w_1)
    w_1.assign_sub(eta*grad_1)

    plt.clf()

    plt.subplot(121)
    plt.plot(hist_sse_00, label='sse_00', c=default_colors[0])
    plt.plot(hist_sse_01, label='sse_01', c=default_colors[2])
    plt.plot(hist_sse_1, label='sse_1', c=default_colors[1])
    plt.legend()

    plt.subplot(122)
    plt.xlim(x_lim0)
    plt.ylim(y_lim0)

    plot_x = np.squeeze(np.mgrid[x_lim0[0]:x_lim0[1]+0.1:0.1, y_lim0[0]:y_lim0[1]+0.05:0.05, 1:2])
    plot_f_00 = tf.tensordot(w_00, plot_x, [0, 0])
    plot_g_00 = tf.sigmoid(plot_f_00)
    plot_f_01 = tf.tensordot(w_01, plot_x, [0, 0])
    plot_g_01 = tf.sigmoid(plot_f_01)
    plot_f_1 = w_1[0]*plot_g_00 + w_1[1]*plot_g_01 + w_1[2]
    plot_g_1 = tf.sigmoid(plot_f_1)

    plt.pcolormesh(plot_x[0], plot_x[1], plot_g_1.numpy(), alpha=0.5, shading='gouraud')

    plot_with_lines(w_00, w_01)

    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.1)
    # plt.show(block=False)



# すべてverginica判定を目標に学習させる
fig = plt.figure()
plt.show(block=False)

# ウェイト
w_00 = tf.Variable(initial_value=-np.array([1.0, 1, -2]))
w_01 = tf.Variable(initial_value=np.array([1.0, 1, -7]))
# w_00 = tf.Variable(initial_value=-np.ones(3)) # sigmoidの方向的にマイナスにしておくとうまくいく
# w_01 = tf.Variable(initial_value=np.ones(3))
w_1 = tf.Variable(initial_value=-np.array([1.0, 1, 0]))

hist_sse_1 = []

eta = 0.005
x_lim0 = [-1, 8]
y_lim0 = [-1, 3]

for i in range(1000):
#     pass
# for i in range(1):
    with tf.GradientTape() as tape:
        f_00 = tf.tensordot(w_00, x, [0, 1])
        g_00 = tf.sigmoid(f_00)

        f_01 = tf.tensordot(w_01, x, [0, 1])
        g_01 = tf.sigmoid(f_01)

        f_1 = w_1[0]*g_00 + w_1[1]*g_01 + w_1[2]
        g_1 = tf.sigmoid(f_1)
        target_1 = (iris.target == 1)
        sse_1 = tf.reduce_sum((g_1 - target_1) ** 2)

    hist_sse_1.append(sse_1.numpy())

    grad = tape.gradient(sse_1, [w_00, w_01, w_1])

    w_00.assign_sub(eta*grad[0])
    w_01.assign_sub(eta*grad[1])
    w_1.assign_sub(eta*grad[2])

    plt.clf()

    plt.subplot(121)
    plt.plot(hist_sse_1, label='sse_1', c=default_colors[1])
    plt.legend()

    plt.subplot(122)
    plt.xlim(x_lim0)
    plt.ylim(y_lim0)

    plot_x = np.squeeze(np.mgrid[x_lim0[0]:x_lim0[1]+0.1:0.1, y_lim0[0]:y_lim0[1]+0.05:0.05, 1:2])
    plot_f_00 = tf.tensordot(w_00, plot_x, [0, 0])
    plot_g_00 = tf.sigmoid(plot_f_00)
    plot_f_01 = tf.tensordot(w_01, plot_x, [0, 0])
    plot_g_01 = tf.sigmoid(plot_f_01)
    plot_f_1 = w_1[0]*plot_g_00 + w_1[1]*plot_g_01 + w_1[2]
    plot_g_1 = tf.sigmoid(plot_f_1)

    plt.pcolormesh(plot_x[0], plot_x[1], plot_g_1.numpy(), alpha=0.5, shading='gouraud')

    plot_with_lines(w_00, w_01)

    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.1)
    # plt.show(block=False)


