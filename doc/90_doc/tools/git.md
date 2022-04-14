## settings
```bash

git config --global init.defaultBranch main

git config --local core.symlinks true
git config --global user.name ""
git config --global user.email ""
git config --global http.proxy "127.0.0.1:10809"
```

## 配置SSH
- 手动打开ssh服务，并在命令行初始化
	- [Starting ssh-agent on Windows 10 fails: "unable to start ssh-agent service, error :1058" - Stack Overflow](https://stackoverflow.com/questions/52113738/starting-ssh-agent-on-windows-10-fails-unable-to-start-ssh-agent-service-erro)
	- [linux - ssh-add returns with: "Error connecting to agent: No such file or directory" - Unix & Linux Stack Exchange](https://unix.stackexchange.com/questions/464574/ssh-add-returns-with-error-connecting-to-agent-no-such-file-or-directory)


```ad-note
title:配置ssh


1. [多个Github账号如何配置SSH Key？ - 简书](https://www.jianshu.com/p/e50aeb57ea57)
2. 创建本地密钥
	- `ssh-keygen -t rsa -b 4096 -C ""  #个人账号`
3. 修改配置文件
	~~~python
	Host AdamZh0u.github.com  
	HostName github.com 
	User git  
	IdentityFile /Users/smile/.ssh/id_rsa
	~~~
4. 添加公钥到github
5. 添加到ssh agent
	- `ssh-add /Users/smile/.ssh/id_rsa`
6. 测试配置`ssh -T git@github.com`
```


## 将本地仓库上传到github

在Github上new一个repository；
进入本地的项目目录下，建立git仓库：
git init
将项目所有文件添加到仓库中：
git add . #全部
git add filename #指定文件
将添加的文件提交到仓库：
git commit -m "文件描述"
将本地仓库关联到Github上：
git remote add origin url_of_your_newrepository
上传代码到Github远程仓库：
git push -u origin main


[如何上传到GitHub的main分支而不是master分支_m0_46419510的博客-CSDN博客_git上传到main分支](https://blog.csdn.net/m0_46419510/article/details/112543544)

```bash
git config --global init.defaultBranch main

git init
git add .
git commit -m ""

git remote add origin git@AdamZh0u.github.com:AdamZh0u/C90_doc.git

git pull --rebase origin main

git push -u origin main
```


github pages 添加domain 

阿里云设置CNMAE 
```
CNAME
doc
adamzh0u.github.io
```

actions
