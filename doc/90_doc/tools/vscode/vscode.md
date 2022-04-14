---
title: vscode
---

[Documentation for Visual Studio Code](https://code.visualstudio.com/docs)

## 完全卸载

- 打开文件所在位置>双击unins000.exe
- 打开我的电脑>C盘>用户>AppData>Roaming>选择code
  - 删不掉的话用管理员命令行 rmdir
- 打开我的电脑>C盘>用户>选择.vscode

## 安装

CP 命令面板

## vscode get started

- quick open c+p
  - ? view suggestions
  - `right` add multiple files
  - c+r open folders and workspaces
- github同步配置,打开文件夹
- zen mode `c+K z`
- 快捷键设置
  - C+k C+s

### 界面快捷键

- cs+m messages
- cs+g git
- cs+x extensions
- cs+f find
- cs+d test
- c+\` terminal
- c+\ 右分屏
- c+1-5切换界面
- c+w 关闭
- c+j panel
- c+B
- alt 切换编辑模式,编辑模式下有些快捷键不能用
- c 切换界面中的文件

### 命令行启动

```bash
# open code with current directory
code .

# open the current directory in the most recently used code window
code -r .

# create a new window
code -n

# change the language
code --locale=es

# open diff editor
code --diff <file1> <file2>

# open file at specific line and column <file:line[:character]>
code --goto package.json:10:5

# see help options
code --help

# disable all extensions
code --disable-extensions .
.vscode folder#
```

### 编辑模式

- Multi-cursor editing
  - Box selection
    - shift + alt + 鼠标
  - add a cursor
    - ctrl + alt + up/dowm
    - alt + click
  - all occurrences
    - ctrl + shift + L
- IntelliSense
  - Ctrl + Space
- Line actions
  - 移动一行 a+up
  - 删除一行 cs+k
  - 复制一行 sa+up
- Rename
  - f2 重命名所有同名变量
- formatting
  - `sa+f` 格式化整个文件
  - c+k c+f 格式化选中
- code folding
  - cs+\[/\] 折叠选中
  - c+k c+0/j 折叠所有
  - c+k c+1-5 折叠级别
- errors and warnings
  - f8

```ad-note
- f1 CP 
- f2 reference
- f3 find next `s+f3`
- f4 rename
- f7 next difference / next highlight
- f8 next problem 
- f5 启动调试  `c+f5`不 启动调试运行

~~~admonition
- 调试模式
 - f5 continue  到下一个断点
 - f6 debug pause
 - f10 step over
 - f11 step into
 - `s+f11` step out 
~~~

- f12 reference  `s+f12` defination on top

```

### workspace

- 设置: 用户设置<工作区设置<项目设置
- 打开文件夹设置存储在.vscode下,而打开workspace,存储在.code-workspace文件中
- 一个workspace中,如果有多个文件夹,不同文件夹中的settings可以不一样,覆盖工作区设置
- 配置方案
  - 先把所有扩展都禁用，然后打开项目，再把需要的扩展，对该工作区开放。

### snippets

- snippets
  - tab建
  - emmet方式- 输入emm
- type check
- 自定义snippets

### 配置文件夹

- task.json

## extensions

- `markdown lint`
  - CP+lint 开关提示,设置lint
  - f8
- `x/Path Intellisense`
- [How We Made Bracket Pair Colorization 10,000x Faster In Visual Studio Code](https://code.visualstudio.com/blogs/2021/09/29/bracket-pair-colorization)
  - 实现彩虹括号的算法
- kite
	- kite不提供下载,从github issue找到源,保存在网盘

### theme

## settings
- 自动补全
	- "python.analysis.completeFunctionParens": true, 补全括号 
	- server 要改成pylance
- 多环境配置 [Feature Request: Enable/disable extensions from config file · Issue #40239 · microsoft/vscode · GitHub](https://github.com/microsoft/vscode/issues/40239)  一个长达四年的issue
- line number - relative
- files.exclude - _private 避免显示
- zen mode / screencast mode 录屏模式
- set proxy to load entensions properly `http://127.0.0.1:10809`
- title配置 `${dirty}${rootName}${separator}${activeEditorMedium}`
- font settings
  - [GitHub - beichensky/Font: FiraCode 和 Operator Mono 字体](https://github.com/beichensky/Font)
  - [为VSCode 设置好看的字体：Operator Mono_浮沉半生的博客-CSDN博客_vscode 字体设置](https://blog.csdn.net/zgd826237710/article/details/94137781#_19)
  - 字体配置
   ```json
  "editor.fontLigatures": true,
  "editor.fontFamily": "Operator Mono",
  "editor.tokenColorCustomizations": {
   "textMateRules": [
    {
     "name": "italic font",
     "scope": [
      "comment",
      "keyword",
      "storage",
      "keyword.control.import",
      "keyword.control.default",
      "keyword.control.from",
      "keyword.operator.new",
      "keyword.control.export",
      "keyword.control.flow",
      "storage.type.class",
      "storage.type.function",
      "storage.type",
      "storage.type.class",
      "variable.language",
      "variable.language.super",
      "variable.language.this",
      "meta.class",
      "meta.var.expr",
      "constant.language.null",
      "support.type.primitive",
      "entity.name.method.js",
      "entity.other.attribute-name",
      "punctuation.definition.comment",
      "text.html.basic entity.other.attribute-name.html",
      "text.html.basic entity.other.attribute-name",
      "tag.decorator.js entity.name.tag.js",
      "tag.decorator.js punctuation.definition.tag.js",
      "source.js constant.other.object.key.js string.unquoted.label.js",
     ],
     "settings": {
      "fontStyle": "italic",
     }
    },
   ]
   ```

### latex

- 安装perl
- 安装workshop
- 配置方案
	- [VSCode LatexWorkshop on WSL with MikTex for XeLaTeX · GitHub](https://gist.github.com/Querela/2da0ac0975ce5d84a948ab734485acec)

```json
autoclean 
subfolder 

```

### python环境配置

[Get Started Tutorial for Python in Visual Studio Code](https://code.visualstudio.com/docs/python/python-tutorial#_run-hello-world)

- 运行
  - `s+enter`运行片段
  - `CP+repl` **调出交互面板**
- 代码片段
- 自动补全
- formatting
  - autopep8
- autodocstring
  - 配置函数输入,输入三个双引号 自动生成  
  - `cs+2`
- Python Test Explorer for Visual Studio Code
- Visual Studio IntelliCode
  - 需要打开设置，更改python server 为 pylance

## 调试与测试配置

### 调试

- 在workspace中设置lunch
- "python": "D:\\Miniconda\\envs\\sds20\\python.exe",
- pdb调试

### 测试


悬浮窗：hover
hover翻页
