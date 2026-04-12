import numpy as np
from data_utils import load_CIFAR10
from tqdm import tqdm

Xtr, Ytr, Xte, Yte = load_CIFAR10("data/cifar-10-batches-py")
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


class nn:
    def __init__(self,k):
        self.k=k
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in tqdm(range(num_test), desc="正在分类图片"):
            distance = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[min_index]

            
        return Ypred


if __name__ == "__main__":
    n = nn()
    n.train(Xtr_rows, Ytr)
    print("开始预测...")
    Yte_predict = n.predict(Xte_rows[:100, :])
    accuracy = np.mean(Yte_predict == Yte[:100])
    print("accuracy: %f" % (np.mean(Yte_predict == Yte[:100])))
