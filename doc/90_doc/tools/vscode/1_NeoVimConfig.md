## 设置代理和conda环境
[[_settings#^9ac28b]]

## 一些示例
[https://github.com/jdhao/nvim-config/blob/master/docs/README.md](https://github.com/jdhao/nvim-config)


# 安装 pre-requisite
[https://github.com/jdhao/nvim-config/blob/master/docs/README.md](https://github.com/jdhao/nvim-config/blob/master/docs/README.md)

## 创建软链接  symbolic link
```link
mklink /D C:\Users\Administrator\AppData\Local\nvim E:\311_VIM\nvim_config\nvim1
```

### Pynvim
-U, --upgrade
```bash
pip install -U pynvim
```
### python-lsp-server
```bash
pip install 'python-lsp-server[all]' pylsp-mypy pyls-isort
```
只在powershell下管用？
[https://github.com/python-lsp/python-lsp-server/issues/150](https://github.com/python-lsp/python-lsp-server/issues/150)
改成双引号成功
### nodejs

### vim-language-server
```bash
npm install -g vim-language-server
```

###  ctags
```bash
choco install universal-ctags
```
###  Ripgrep
[GitHub - BurntSushi/ripgrep: ripgrep recursively searches directories for a regex pattern while respecting your gitignore](https://github.com/BurntSushi/ripgrep)
面向行的搜索工具，它递归地在当前目录中搜索正则表达式模式
### Linters
```bash
pip install pylint
pip install vim-vint
```

# 设置Nvim
## 安装packer.vim
```
git clone --depth=1 https://github.com/wbthomason/packer.nvim "$env:LOCALAPPDATA\nvim-data\site\pack\packer\opt\packer.nvim"
```

* 创建init.vim
```vim
let g:config_files = [
\ 'globals.vim',
\ ]

for s:fname in g:config_files
execute printf('source %s/settings/%s', stdpath('config'), s:fname)
endfor
```
## 

* 运行python
	* ==!==表示在命令行运行