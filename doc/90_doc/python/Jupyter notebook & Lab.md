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


