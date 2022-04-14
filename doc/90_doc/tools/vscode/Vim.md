[[PyCharm]] 
[技术|将 Vim 配置成一个写作工具](https://linux.cn/article-13607-1.html)
[高效做笔记:vim + markdown - 知乎](https://zhuanlan.zhihu.com/p/84773275)

## VimEverywhere
- 三种模式 
## Vim Movement
- normal insert visual 模式
- Normal模式 移动
	- 按字符移动 HJKL
	- 按单词移动 WE word end  大写WE：以空格分割的单词 bB 按单次
	- ^$ 开始和结尾	--> 大写IA A行尾 append I 行头insert
	- tab+hjkl --> 方向键

C-o 下面新增一行
C-o 回退一格
u 撤销
d 剪切 delete
y yank 粘贴
ctrl z + fg
:bd buffer delete
:source % excute current
:PlugInstall


**find and replace**
在normal模式下按下`/`即可进入查找模式，输入要查找的字符串并按下回车。 Vim会跳转到第一个匹配。按下`n`查找下一个，按下`N`查找上一个。



**Cut and paste:**
1.  Position the cursor where you want to begin cutting.
2.  Press `v` to select characters, or uppercase `V` to select whole lines, or `Ctrl-v` to select rectangular blocks (use `Ctrl-q` if `Ctrl-v` is mapped to paste).
3.  Move the cursor to the end of what you want to cut.
4.  Press `d` to cut (or `y` to copy).
5.  Move to where you would like to paste.
6.  Press `P` to paste before the cursor, or `p` to paste after.
## vimrc
[GitHub - junegunn/vim-plug: Minimalist Vim Plugin Manager](https://github.com/junegunn/vim-plug)
windows 和unix设置不同

[Your first VimRC: How to setup your vim's vimrc - YouTube](https://www.youtube.com/watch?v=n9k9scbTuvQ&list=RDCMUC8ENHE5xdFSwx71u3fDH5Xw&start_radio=1)
vim ~/.vimrc

* source % 
* python 配置
	*  [如何使 Vim 下开发 Python 调试更方便？ - 知乎](https://www.zhihu.com/question/20271508)
```vim
syntax on 

set noerrorbells " 取消 Vim 的错误警告铃声，关闭它以免打扰到我们 "
set textwidth=100 " 确保每一行不超过 100 字符 "
set tabstop=4 softtabstop=4
set shiftwidth=4
set expandtab
set smartindent 
set linebreak 
set number
set showmatch 
set showbreak=+++
set smartcase
set nowrap
set noswapfile
set nobackup
set undodir=~/.vim/undodir
set undofile
set incsearch 
set spell 
set showmatch 
set confirm 
set ruler 
set autochdir 
set autowriteall 
set undolevels=1000
set backspace=indent,eol,start

set colorcolumn=80
highlight ColorColumn ctermbg=0 guibg=lightgrey

call plug#begin('~/.vim/plugged')
`Plug 'gruvbox-community/gruvbox'`

call plug#end
```
## NeoVim
* 安装plugin
	* 
```ps
iwr -useb https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim |`
    ni "$(@($env:XDG_DATA_HOME, $env:LOCALAPPDATA)[$null -eq $env:XDG_DATA_HOME])/nvim-data/site/autoload/plug.vim" -Force
```
* 路径
	* autoload C:\Users\Administrator\AppData\Local\nvim-data\site\autoload
	* plugged C:\Users\Administrator\.config\nvim\init.vim ## 相当于vimrc
		* call plug#begin('~/vimfiles/plugged') # windows
		* ~/.vim/ # unix
	* `nvim %HOME%/.config/nvim/init.vim` 配置
	* 装到`C:\Users\Administrator\AppData\Local\nvim`下成功
* plugin 安装网络
	
* waketime
	* 5974f467-0d8a-4a8a-a68e-da5995e49b34