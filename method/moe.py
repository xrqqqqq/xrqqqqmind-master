import torch

# torch.div
# 逐元素相除
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4, 5, 6])
result = torch.div(a, b)
print(result)  # 输出: tensor([0.2500, 0.4000, 0.5000])

# torch.mean 求平均值
# dim = 0,消除0维度
torch.mean(a, dim=0)  # 输出: tensor(2. )

# torch.scatter_add 按照索引把值加入目标张量指定位置
out = torch.zeros(5)
index = torch.tensor([0, 1, 0, 3])
src = torch.tensor([1.0, 2.0, 3.0, 4.0])

out.scatter_add_(0, index, src)
# 0是维度，index是索引，src是要加的值
print(out)  # 输出: tensor([4., 2., 0., 4., 0.])

x = torch.tensor([1, 2, 3])
# 每个元素复制repeats次，按原顺序排好
y = torch.repeat_interleave(x, repeats=2)
print(y)  # 输出: tensor([1, 1, 2, 2, 3, 3])

x = torch.tensor([30, 10, 20])
# torch.argsort返回排序后的索引
idx = torch.argsort(x)
print(idx)  # 输出: tensor([1, 2, 0])
print(x[idx])  # 输出: tensor([10, 20, 30])

x = torch.tensor([1, 2, 3])
# 计算正整数出现的次数
count = torch.bincount(x)
print(count)  # 输出: tensor([0, 1, 1, 1])
