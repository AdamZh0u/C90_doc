## Sublime Text 3


# github 配置同步

```ad-note
title:替换默认配置路径

- 默认C盘路径：C\User\%AppData\Roaming%\Sublime Text 3\文件夹

- 安装路径下创建`Data`文件夹，放入配置即可替换·

- Data\packages\User下创建snippets文件夹可以识别自定义的片段
```

## package control

# Packages

- `Spacegrey` theme
- `material` Theme
- `sidebar enhancement` 替代`siderbar tools`
- 


# Latex

## 环境配置

- sublime text
- texlive or miktex
- 安装package control
    + 安装textools
    + 安装latex cwl 命令自动补全
- 安装sumatraPDF
- 配置反向定位
    + 将sumatraPDF加入系统path
    + cmd运行`sumatrapdf.exe -inverse-search "\"D:\Sublime Text 3\sublime_text.exe\" \"%f:%l\""`
      开启反向搜索
    +  现在sublime的设置有了反向搜索的配置，更改路径就行
- 配置latextools
    + "texpath" : "D:\\MikTex\\miktex\\bin\\x64;$PATH",
    + "sumatra": "C:\\Users\\zhouz\\AppData\\Local\\SumatraPDF\\SumatraPDF.exe",
    + "builder": "simple",
    + windows 上还需要修改"sublime_executable": "D:\\Sublime Text 3\\subli.exe",
        ref:    [4109 always opens a new windows after build? - Technical Support - Sublime Forum](https://forum.sublimetext.com/t/4109-always-opens-a-new-windows-after-build/59133/18)
- 配置snippets
    + `Packages\Usersnippets`文件夹下
- axmath
    + 每次复制会生成右侧栏笔记
    + 选中后拖动可以生成磁贴

```ad-note

[博士汪倾力整理！全网最强大的LaTeX+Sublime Text写作环境-第二集 手把手教你安装配置整套环境_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1p44y1P7P4/)
```


## shortcuts
```ad-note


```
```dem
