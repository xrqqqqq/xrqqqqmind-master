import torch
import torch.nn as nn
from rich.traceback import install

install()

# dropout_layer = nn.Dropout(p=0.3)

# t1 = torch.Tensor([1,2,3])
# #Tensor与tensor方法的区别在于，Tensor是torch库中的一个类，而tensor方法是torch库中的一个函数。Tensor类用于创建和操作张量对象，而tensor方法用于创建张量对象并返回一个新的张量。
# t2 = dropout_layer(t1)
# #总期望不变，剩下成比例扩大
# # tensor([1.4286, 2.8571, 4.2857])
# print(t2)

# layer = nn.Linear(in_features = 3,out_features = 5,bias = True)
# t1 = torch.Tensor([1,2,3])
# # print(t1.shape)  # torch.Size([3])
# t2 = torch.Tensor([[1,2,3]])#(1,3)
# output2 = layer(t2) 
# print(output2)  # shape(1,5)


# t = torch.tensor([[1,2],[3,4]])
# t_view1 = t.view(1,4)
# print(t_view1)  # tensor([[1, 2, 3, 4]]),(1,4)


#交换维度
# t1 = torch.Tensor([[1,2],[3,4]])
# t1 = t1.transpose(0,1)#0维度换1维度
# print(t1)
# tensor([[1., 3.],
#         [2., 4.]])

# #mask
# x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

# print(torch.triu(x))
# #triu函数返回输入张量的上三角部分，保留对角线上的元素，并将下三角部分的元素设置为零。 

# # tensor([[1, 2, 3],
# #         [0, 5, 6],
# #         [0, 0, 9]])
# print(torch.triu(x,diagonal = 1))
# # diagonal参数指定了对角线的偏移量。默认值为0，表示主对角线。正值表示上方的对角线，负值表示下方的对角线。
# tensor([[0, 2, 3],
#         [0, 0, 6],
#         [0, 0, 0]])

# #reshape

# x = torch.arange(1,7) #tensor([1, 2, 3, 4, 5, 6])

# y = torch.reshape(x,(2,3)) 
# print(y)
# tensor([[1, 2, 3],
#         [4, 5, 6]])