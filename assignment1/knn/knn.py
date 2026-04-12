# numpy是内存杀手！
import numpy as np

# 这个是cs231n用来导入数据集的
from data_utils import load_CIFAR10

# 臭加进度条的
from tqdm import tqdm

# 计数的
from collections import Counter

# tr就是train, te就是test
Xtr, Ytr, Xte, Yte = load_CIFAR10("data/cifar-10-batches-py")
# Xtr读出来是一个50000*32*32*3的矩阵
# Xte读出来也是只不过它是10000，我们这个操作是把他拍扁，变成一个50000*3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


# 这cs231n 谁教的这样子写变量名
class knn:
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]

        # 开始计算L2距离，利用numpy的矩阵广播机制，内存杀手
        # 求和会降维，要保证右对齐，所以就要改一下
        test_sum = np.sum(np.square(X), axis=1)[:, None]
        train_sum = np.sum(np.square(self.Xtr), axis=1)

        inner_product = np.dot(X, self.Xtr.T)
        dists = np.sqrt(test_sum + train_sum - 2 * inner_product)

        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in tqdm(range(num_test), desc="正在分类图片"):
            # 找出距离最近的几个邻居
            nearest_k_indices = np.argsort(dists[i])[: self.k]
            nearest_k_labels = self.ytr[nearest_k_indices]
            # 统计出现次数
            # 把数组拍成1维的，然后统计标签出现次数，找到第一个元组的第一个元素就是了
            Ypred[i] = Counter(nearest_k_labels.flatten()).most_common(1)[0][0]

        return Ypred


def main():
    k = int(input())
    n = knn(k)
    
    n.train(Xtr_rows[:, :], Ytr[:])
    print("训练完成，开始测试")
    Ytr_predict = n.predict(Xtr_rows[:1, :])
    acc = np.mean(Ytr_predict == Ytr[:1])
    print("准确率: %f" % acc)


if __name__ == "__main__":
    main()
