# Pytorch 速成

## 第一部分：张量

张量是 Pytorch 最基本的构成模块，每一个张量都是一个多维矩阵

张量可以这样子创建

```Python
list_of_lists = [
  [1, 2, 3],
  [4, 5, 6],
]
data = torch.tensor(list_of_lists)
```

也可以直接声明了写出来

```Python
data = torch.tensor([
                     [0, 1],
                     [2, 3],
                     [4, 5]
                    ])
```

每一个张量都有对应的数据类型，可以在创建张量的时候去显式指定数据类型

```Python
data = torch.tensor([
                     [0, 1],
                     [2, 3],
                     [4, 5]
                    ], dtype=torch.float32)
```

有一些工具函数

- `torch.zeros(n,m)` 创建一个 n 维的，每一维有 m 个 0 的张量
- `torch.ones(n,m)` 同上，不过这次是创建一个全是 1 的张量
- `torch.arange(n,m)` 创建一个 n 维度的，每一维的数据是 1 到 m 的张量。
- 一些和运算有关的函数
- `torch.matmul` 这个方法用于实现两个张量的点积，就是线代的矩阵乘法。
- 矩阵的形状可以用 `torch.shape` 方法访问。
- 重塑张量可以使用 `torch.view` 方法进行，填上你想要塑成啥样的就行了
- 可以与 `numpy` 的数组进行互相转换。
- `torch.sum` 用于对某一个维度上的数据进行求和操作的，这个用法和 `numpy` 的基本一致。需要指定一下 `dim` 参数，从 0 开始。
- `torch.mean` 用于对某一个维度上的数据进行求平均操作

- 索引访问和 `numpy` 基本没差别

### 反向传播

调用 `backward()` 方法进行自动求导。

示例如下

```Python
x = torch.tensor([2.], requires_grad=True)
# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)
```

这里定义了 $x$ 的指为 2，并且初始化梯度为 0。

```Python
y = x * x * 3 # 3x^2
y.backward()
pp.pprint(x.grad) # d(y)/d(x) = d(3x^2)/d(x) = 6x = 12
```

这里对 $y$ 求一次导，然后会自动加到 x 的梯度上。

如果想要单独获得这个式子的梯度，需要每次把 $x$ 的梯度都声明为 None

## 第二部分：神经网络

导入头文件

```Python
import torch.nn as nn
```

### 线性层

使用 `nn.Linear(H_in,H_out)` 来创建一个线性层，改成接收一个 （N，...，H_in）的矩阵，最后输出一个最后一维是 H_out 的基本相同矩阵。

该线性层执行时会对权重和 b 进行随机初始化。可以设置 `bias=False` 来规避掉这一点。

另外的一些方法

- `nn.Conv2d` 二维卷积层，额外可接收参数有 `kernel_size` 卷积核的大小，`stride` 步长，`padding` 填充
- `nn.ConvTranspose2d`反卷积层，主要用于上采样，参数与卷积层基本一致。
- `nn.BatchNorm1d` 批归一化，主要参数 `num_features` 输入的特征维度数量，`eps`为了数值稳定加在分母上的微小值。
- `nn.LayerNorm` 层归一化，`normalized_shape`指定想要归一化的维度形状。`eps`为了数值稳定，`elementwise_affine`，布尔值，如果为 1 则会提供两个可训练参数让模型自己决定归一化后的分布。
- `nn.BatchNorm2d`对二维特征图进行批归一化。主要参数 `num_features` 输入的通道数。
- `nn.MaxPool2d` 二维最大池化。主要参数 `kernel_size` 池化窗口大小，`stride` 步长，默认和前一个参数相同，`padding` 填充。

### 激活函数层

- `nn.ReLU`
- `nn.Sigmoid`
- `nn.LeakyReLU`

输入的形状与输出的形状保持一致。

### 将各层组合在一起

使用 `nn.Sequential` 

使用示例

```Python
block = nn.Sequential(
    nn.Linear(4, 2),
    nn.Sigmoid()
)

input = torch.ones(2,3,4)
output = block(input)
output
```

### 自定义模块

使用 `nn.Module`**类**来实现自定义

所有继承自 `nn.Module`的类都需要去实现一个 `forward(x)` 函数

示例

```Python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # Call to the __init__ function of the super class
    super(MultilayerPerceptron, self).__init__()

    # Bookkeeping: Saving the initialization parameters
    self.input_size = input_size
    self.hidden_size = hidden_size

    # Defining of our model
    # There isn't anything specific about the naming of `self.model`. It could
    # be something arbitrary.
    self.model = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.input_size),
        nn.Sigmoid()
    )

  def forward(self, x):
    output = self.model(x)
    return output
```

### 优化

导入头文件

```Python
import torch.optim as optim
```

这个库提供了常用的例如 `optim.SGD` 和 `optim.Adam` 等等。

这些方法需要我们传入模型参数，还有一个学习率参数 `lr`

示例代码

```Python
# Instantiate the model
model = MultilayerPerceptron(5, 3)

# Define the optimizer
adam = optim.Adam(model.parameters(), lr=1e-1)

# Define loss using a predefined loss function
loss_function = nn.MSELoss()

# Calculate how our model is doing now
y_pred = model(x)
loss_function(y_pred, y).item()

# Set the number of epoch, which determines the number of training iterations
n_epoch = 10

for epoch in range(n_epoch):
  # Set the gradients to 0
  adam.zero_grad()

  # Get the model predictions
  y_pred = model(x)

  # Get the loss
  loss = loss_function(y_pred, y)

  # Print stats
  print(f"Epoch {epoch}: traing loss: {loss}")

  # Compute the gradients
  loss.backward()

  # Take a step to optimize the weights
  adam.step()
```

