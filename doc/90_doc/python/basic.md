https://github.com/vinta/awesome-python

## OS
```python
import os 
base_dir = os.path.dirname(os.path.realpath('__file__')) 
print(base_dir)

base_dir = os.getcwd()
```

## argparse
```python

```

## pickle
```python
with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
	u = pkl._Unpickler(rf)
	u.encoding = 'latin1'
	cur_data = u.load()
	objects.append(cur_data)
```

### cpickel
>The `pickle` module has an transparent optimizer (`_pickle`) written in C. It is used whenever available. Otherwise the pure Python implementation is used.

[19191859](https://stackoverflow.com/questions/19191859/what-difference-between-pickle-and-pickle-in-python-3)

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

# Gallery
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

# Pyzotero
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