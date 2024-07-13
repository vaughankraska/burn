import torch

torch.set_printoptions(threshold=5000, sci_mode=False)

### LHS

for i in range(4):
    i = 2 * i

    a = torch.arange(256 * i, 256 * (i + 1), dtype=float).reshape(16, 16)
    b = torch.arange(256 * (i + 1), 256 * (i + 2), dtype=float).reshape(16, 16)
    lhs_tmp = torch.cat((a, b), dim=1)
    rhs_tmp = torch.cat((a, b), dim=0)
    
    if i == 0:
        lhs = lhs_tmp
        rhs = rhs_tmp
    else:
        if i < 2: 
            lhs = torch.cat((lhs, lhs_tmp), dim=0)
        if i < 2:
            rhs = torch.cat((rhs, rhs_tmp), dim=1)


# torch.set_printoptions(threshold=5000)
print(lhs)
print(rhs)
res = lhs.matmul(rhs)
# print(res.shape)

for i in range(1):
    for j in range(1):
        sixteenth = res[i*16:(i+1)*16,j*16:(j+1)*16]
        sixteenth.reshape(256)
        if i == 0 and j == 0:
            result = sixteenth
        else:
            result = torch.cat((result, sixteenth), dim=0)

print(result.reshape(256))

# import torch

# torch.set_printoptions(threshold=5000, sci_mode=False)

# ### LHS

# for i in range(4):
#     i = 2 * i

#     a = torch.arange(256 * i, 256 * (i + 1), dtype=float).reshape(16, 16)
#     b = torch.arange(256 * (i + 1), 256 * (i + 2), dtype=float).reshape(16, 16)
#     lhs_tmp = torch.cat((a, b), dim=1)
#     rhs_tmp = torch.cat((a, b), dim=0)
    
#     if i == 0:
#         lhs = lhs_tmp
#         rhs = rhs_tmp
#     else:
#         lhs = torch.cat((lhs, lhs_tmp), dim=0)
#         rhs = torch.cat((rhs, rhs_tmp), dim=1)


# # torch.set_printoptions(threshold=5000)
# # print(lhs)
# # print(lhs.shape)
# res = lhs.matmul(rhs)
# for i in range(4):
#     for j in range(4):
#         sixteenth = res[i*16:(i+1)*16,j*16:(j+1)*16]
#         sixteenth.reshape(256)
#         if i == 0 and j == 0:
#             result = sixteenth
#         else:
#             result = torch.cat((result, sixteenth), dim=0)

# print(result.reshape(4096))

a = torch.tensor([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11., 12.,  13.,  14.,  15., 256., 257., 258., 259., 260., 261., 262., 263., 264., 265., 266., 267., 268., 269., 270., 271.])
b= torch.tensor([  0.,  16.,  32.,  48.,  64.,  80.,  96., 112., 128., 144., 160., 176., 192., 208., 224., 240., 256., 272., 288., 304., 320., 336., 352., 368., 384., 400., 416., 432., 448., 464., 480., 496.])
print(a.matmul(b))