import torch

# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10,20,30,40,50])

# condition = x > 3
# #where函数根据condition的值选择x或y中的元素，如果condition为True，则选择x中的元素，否则选择y中的元素。
# result = torch.where(condition,x,y)

# print(result)  # tensor([10, 20, 30,  4,  5])

# #等差数列

# t = torch.arange(0,10,2)
# print(t)  # tensor([0, 2, 4, 6, 8])

# t2 = torch.arange(5,0,-1)
# print(t2)  # tensor([5, 4, 3, 2, 1])

# outer积是指两个向量的乘积，结果是一个矩阵，其中每个元素是第一个向量的一个元素与第二个向量的一个元素的乘积。

# v1 = torch.tensor([1,2])
# v2 = torch.tensor([3,4])
# print(torch.outer(v1,v2))

# tensor([[3, 4],
#         [6, 8]])

# cat函数用于将多个张量沿指定维度连接起来。
# t1 = torch.tensor([[1,2],[3,4]])
# t2 = torch.tensor([[5,6],[7,8]])

# c1 = torch.cat((t1,t2),dim=0)  #沿着第0维连接
# c2 = torch.cat((t1,t2),dim=1)  #沿着第1维连接

# print(c1)
# print(c1.shape)

# print(c2)
# print(c2.shape)
# tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
# torch.Size([4, 2])
# tensor([[1, 2, 5, 6], [3, 4, 7, 8]])
# torch.Size([2, 4])


# unsqueeze函数用于在指定位置插入一个维度为1的维度。
# t1 = torch.tensor([1,2,3])
# t2 = t1.unsqueeze(0)  #在第0维插入一个维度
# print(t1.shape)
# print(t2.shape)
# print(t2)
# torch.Size([3])
# torch.Size([1, 3])
# tensor([[1, 2, 3]])
print(8//2)