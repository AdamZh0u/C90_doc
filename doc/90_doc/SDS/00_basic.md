https://github.com/vinta/awesome-python
## Python Basic

100_0000
[下划线（_）在python中的作用 | 酷python](http://www.coolpython.net/python_senior/senior_feature/underline_effect.html)

# OS
```python
import os 
base_dir = os.path.dirname(os.path.realpath('__file__')) 
print(base_dir)

base_dir = os.getcwd()
```

# argparse
```python

```

# pickle
```python
with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
	u = pkl._Unpickler(rf)
	u.encoding = 'latin1'
	cur_data = u.load()
	objects.append(cur_data)
```

### cpickel
>The `pickle` module has an transparent optimizer (`_pickle`) written in C. It is used whenever available. Otherwise the pure Python implementation is used.

[what-difference-between-pickle-and-pickle-in-python-3](https://stackoverflow.com/questions/19191859/what-difference-between-pickle-and-pickle-in-python-3)

### 速度对比
* 为什么pkl.Unpickler更快？

```python
%%timeit
import pickle as pkl

objects = []
with open("data/ind.{}.{}".format("cora", "x"), 'rb') as rf:
    u = pkl.Unpickler(rf,encoding = "latin1")
    cur_data = u.load()
    objects.append(cur_data)
```
## datetime
```python
date_data_begin = datetime.datetime(2020,3,1)
date_data_end = datetime.datetime(2020,4,21)

# 计算间隔
num_days_data = (date_data_end-date_data_begin).days

# 日期加减
date_data_begin - datetime.timedelta(days = diff_data_sim)
```

## Gallery
* 下载数据或使用本地数据
```python
def get_jhu_confirmed_cases():
    """
        Attempts to download the most current data from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.
        Only works if the module is located in the repo directory.

        Returns
        -------
        : confirmed_cases
            pandas table with confirmed cases
    """
    try:
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        confirmed_cases = pd.read_csv(url, sep=",")
    except Exception as e:
        print("Failed to download current data, using local copy.")
        this_dir = os.path.dirname(__file__)
        confirmed_cases = pd.read_csv(
            this_dir + "/../data/confirmed_global_fallback_2020-04-28.csv", sep=","
        )

    return confirmed_cases
```

## 读取合并多个表
```python
import glob

files = glob.glob("./data/shenzhen POI CSV版本/*.csv")

# 读取合并多个poi表
ls_dfs = {}

for i in range(len(files)):
    # print(files[i])
    df = pd.read_csv(files[i], encoding="gbk",low_memory=False)
    
    if "WGS84_Lat" in df.columns:
        df = df.rename(columns={"WGS84_Lat":"WGS84_纬度","WGS84_Lng":"WGS84_经度"})
    
    ls_dfs[i] = df.dropna(how="all") ## 行全部为空
    # print(ls_dfs[i].shape)
    
df_concated = pd.concat(ls_dfs,ignore_index=True).drop({"Unnamed: 0","Unnamed: 0.1"},axis=1)

df_concated["类"] = df_concated.apply(lambda x: x["大类"]+"|"+x["中类"]+"|"+x["小类"], axis=1)
```

## Pyzotero
```python
library_type = "user"
library_id = "6486920"
api_key = "gqbyGYLWO4Gi3Cl0Xi3W5Ach"

from pyzotero import zotero
zot = zotero.Zotero(library_id, library_type, api_key)
items = zot.top(limit=5)
# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
    print('Item: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))
```

# Print  table to file 

```python
import docx
from docx.shared import Pt

#Print to file
table = pd.DataFrame(d)

doc = docx.Document()
t = doc.add_table(table.shape[0]+1,table.shape[1])
for j in range(table.shape[-1]):
    t.cell(0,j).text = table.columns[j]

for i in range(table.shape[0]):
    for j in range(table.shape[-1]):
        t.cell(i+1,j).text = str(table.values[i,j])
for row in t.rows:
    for cell in row.cells:
        paragraphs = cell.paragraphs
        for paragraph in paragraphs:
            for run in paragraph.runs:
                font = run.font 
                font.name = 'Helvetica 55 Roman'
                font.size = Pt(7)
doc.save("../outputs/Extended_data/Extended_data_5_D1.docx")
```


# itertools


# 高阶函数
- 输入是函数的函数
	- [引入一行代码让python提速50倍:两个jit库介绍_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1k44y1E7iW)

```python
from numba import jit
@jit(nopython=True)

# gpu tpu 更快
from jax import jit as jax_jit
@jax_jit
jit_mat = jax_it(mat)
```


# Jupyter notebook

# Jupyter Lab

```bash
jupyter lab
```

* 使用脚本

```
%%cmd
where python

## %cd 和!cd的区别
%cd

%ls
```

## 配置
- 未关闭的jupyter导致开启错误
	- [how to close running jupyter notebook servers? · Issue #2844 · jupyter/notebook · GitHub](https://github.com/jupyter/notebook/issues/2844)
	- [how to close running jupyter notebook servers? · Issue #2844 · jupyter/notebook · GitHub](https://github.com/jupyter/notebook/issues/2844)
```bash
## 显示正在跑的server
jupyter lab list

##Check where your runtime folder is located:  
jupyter --paths

## Remove all files in the runtime folder:  
rm -r [path to runtime folder]/*

## Then relaunch your notebook on the desired ip and port:  
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &
```


# Conda

##  Basic
* 创建
```bash
conda create -n py3 python=3.5
```
* 删除

```bash
conda env remove -n py3
```

* 重命名--复制删除

```bash
# 复制
conda create -n conda-new --clone conda-old
# 删除
conda remove -n conda-old --all
```

* 常用包的安装

```bash
conda install -c conda-forge jupyterlab numpy pandas matplotlib
```

* 更新包

```bash
## 重新安装
conda uninstall 
conda install package==version

## 更新conda
conda update conda

```
* 换源
	* 
* 理解conda
	* 就像一个诊所和医院，单个python是一个诊所，可以看所有的病，但是如果病情复杂，不同的功能之间会有矛盾，这时候需要专科门诊，在医院，每一个门诊的功能不相同，病人的要求也不同。

* 清理
```bash
conda clean -p //删除没有用的包（推荐） 

conda clean -t //tar打包

conda clean -y -all //删除全部的安装包及cache

conda clean --all
```

## 环境文件
- 生成requirements.txt文件
```bash
pip freeze > requirements.txt
```
- 从文件安装
```python
conda env create --file=myfile.yaml


```
- 安装requirements.txt依赖
```bash
# pip
pip install -r environment.yml

# 使用conda
conda env update -n <env> --file environment.yml
```

## Q&A
### colab 安装conda
```
!pip install -q condacolab
import condacolab
condacolab.install()
```
### `-c conda-forge `什么意思？
