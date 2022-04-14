---
title:LaTex
---
## Basic
- texlive vs miktex
	- [texlive - What are the advantages of TeX Live over MiKTeX? - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/20036/what-are-the-advantages-of-tex-live-over-miktex)
- **PDFLaTeX** 编译模式与 **XeLaTeX** 区别如下：
	1. PDFLaTeX 使用的是TeX的标准字体，所以生成PDF时，会将所有的非 TeX 标准字体进行替换，其生成的 PDF 文件默认嵌入所有字体；而使用 XeLaTeX 编译，如果说论文中有很多图片或者其他元素没有嵌入字体的话，生成的 PDF 文件也会有些字体没有嵌入。
	2. XeLaTeX 对应的 XeTeX 对字体的支持更好，允许用户使用操作系统字体来代替 TeX 的标准字体，而且对非拉丁字体的支持更好。  
	3. PDFLaTeX 进行编译的速度比 XeLaTeX 速度快。
- 星号是干嘛的
	- *-ed form just insists that the space appear, while the un-*-ed form allows the space to be dropped in many cases

## Learn Tex

- [(La)TeX tutorials | The TeX FAQ](https://texfaq.org/FAQ-tutbitslatex)
- [CTAN: Package Catalogue](https://ctan.org/pkg/catalogue)
- [latexsheet](http://wch.github.io/latexsheet/)

# Awesome LaTex

## Themes
- [Clean Thesis — A LaTeX Style for Thesis Documents • Developed by Ricardo Langner](http://cleanthesis.der-ric.de/)

## Templetes
- [Microtype document](https://mirrors.sustech.edu.cn/CTAN/macros/latex/contrib/microtype/microtype.pdf#page=1&zoom=100,0,0)


# VSCode Latex Workshopsettings


配置参考
- [Visual Studio Code (vscode)配置LaTeX - 知乎](https://zhuanlan.zhihu.com/p/166523064)
- 使用Latexmk编译
	- [xelatex 以及 latexmk 命令行编译 - 知乎](https://zhuanlan.zhihu.com/p/256370737)
	- overleaf默认也是用这个
	- [tex core - What is the difference between "-interaction=nonstopmode" and "-halt-on-error"? - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/258814/what-is-the-difference-between-interaction-nonstopmode-and-halt-on-error)

## Snippets

```latex
% header 
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\rightmark}
\fancyfoot[C]{\thepage}
```

```latex
\usepackage[style=apa,]{biblatex}
\addbibresource{references.bib}
% 修改bib utl去掉 doi_i url_l
```
