## Sklearn

# Preprocess

### LabelEncoder

```python

le_poi_second_level.classes_
inverse_transform
transform
```
# DataSet
## Load iris
```python
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
```

# Utils
## safe indexing

```python
sklearn.utils._safe_indexing(df, np.array(range(10)), axis=0)

sklearn.utils._safe_indexing(df, cols, axis=1)
```




space 的概念阐述crime space和流的信息重构


双曲嵌入
式子奥利奥