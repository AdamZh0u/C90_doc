# Basic
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

# 环境文件
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

# Q&A
## colab 安装conda
```
!pip install -q condacolab
import condacolab
condacolab.install()
```
## `-c conda-forge `什么意思？
