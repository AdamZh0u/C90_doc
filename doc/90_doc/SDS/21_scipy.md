# scipy.sparse
## Sparse matrix classes
- 1，COO (Coordinate List Format)：**座标格式**，容易创建但是不便于矩阵计算，用 `coo_matrix`
- 2，CSR (Compressed Sparse Row)：**压缩行格式**，不容易创建但便于矩阵计算，用 `csr_matri`
- 3，CSC (Compressed Sparse Column)：**压缩列格式**，不容易创建但便于矩阵计算，用 `csc_matrix`
- 4，LIL (List of List): **内嵌列表格式**，支持切片但也不便于矩阵计算，用 `lil_matrix`
- 5，DIA (Diagnoal)：**对角线格式**，适合矩阵计算，用 `dia_matrix`

$负数小数会怎样？
![COO matrix](https://mmbiz.qpic.cn/mmbiz_gif/114Mib4UdUMeKLl2VDK1k29icZabw4KvCMiaGE8In9UVIHF4hGQE6jicsZPxtz3uw4j6g9Gqp4ZzfAr2Jg3SKVQycg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)
### coo_matrix
```python

```

### dia_matrix
```python
from scipy.sparse import dia_matrix
n = 10
ex = np.ones(n)
data = np.array([ex, 2 * ex, ex])

array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
```


```
offsets = np.array([-1, 0, 1])
dia_matrix((data, offsets), shape=(n, n)).toarray()
```

```python
array([[2., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 2., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 2., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 2., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 2., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 2., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 2., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 2., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 2., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 2.]])
```

### csr_matrix
#### eliminate_zeros
```python
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros() # 不存储 0 
```
