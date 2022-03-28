```python
import matplotlib as mpl
import matplotlib.pyplot as plt
```

## Modules

* https://matplotlib.org/3.2.1/api/index.html#modules
* [matplotlib](#matplotlib)
* [matplotlib.font_manager](#font_manager)


```python
mpl.rcdefaults()
mpl.rcParams## defalut
```




 


### font_manager

* 字体并不等于text
* 颜色，bbox是text对象的属性
* 默认是用family下的排序，因此先要修改排序，再查找，再使用
* 关于网络字体的渲染 ：https://medium.com/gsoc-2k19-with-mozilla/text-web-fonts-math-text-36ba884fb476


```python
## 修改font
test_font = "1234567890 \nabcdefghijklmnopqrstuvwxyz \nABCDEFGHIJKLMNOPQRSTUVWXYZ"
fig,ax = plt.subplots()
ax.set_axis_off()


## default font settings
font1 = {'family':'sans-serif',
         "name":'DejaVu Sans',
        "style":"normal", # normal (or roman), italic  or oblique
        "variant":"normal", # normal or small-caps(smaller for truetype font)
        "weight":"normal"}#normal, bold, bolder, lighter, 100, 200, 300, ..., 900
#sans-serif, normal, normal, normal, normal, scalable.


fig.text(0,1,'sans-serif'+test_font,font1)

#findfont: Font family ['Computer Modern Roman'] not found. Falling back to DejaVu Sans.


#font.serif      : DejaVu Serif, Bitstream Vera Serif, Computer Modern Roman, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif
#font.sans-serif : DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
#font.cursive    : Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, cursive
#font.fantasy    : Comic Neue, Comic Sans MS, Chicago, Charcoal, ImpactWestern, Humor Sans, xkcd, fantasy
#font.monospace  : DejaVu Sans Mono, Bitstream Vera Sans Mono, Computer Modern Typewriter, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace

## 选定一个family，顺序选择知道存在

font2 = {'family':'monospace','size' : 20}
fig.text(0,0.7,'monospace'+test_font,font2)

font3 = {'family':'fantasy','size' : 20}
fig.text(0,0.4,'fantasy'+test_font,font3)

font4 = {'family':'cursive','size' : 20}
fig.text(0,0.1,"cursive"+ test_font,font4)

font5 = {'family':'serif','size' : 20}
fig.text(0,-0.2,"serif"+ test_font,font5);
```

![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_font.png)


#### font props



```python
# https://matplotlib.org/3.1.1/api/font_manager_api.html#matplotlib.font_manager.FontProperties

## class FontProperties
default = mpl.font_manager.FontProperties(family="sans-serif", style="normal", variant="normal", 
                                          weight="normal", stretch="normal", size=10, fname=None)
default.get_fontconfig_pattern()
```




    'sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=10.0'




```python
## 查找系统字体
mpl.font_manager.findSystemFonts(fontpaths="D:\\ENVI\\IDL85\\resource\\fonts\\tt", fontext='ttf')
```




    ['D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0419m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0011m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt9830z_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt9831z_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0582m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0583m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\envisym_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0014m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0012m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0005m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0611m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0006m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0013m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0004m_.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\DejaVuSans.ttf',
     'D:\\ENVI\\IDL85\\resource\\fonts\\tt\\tt0003m_.ttf']




```python
# 类FontManager 用来管理字体路径
fm = mpl.font_manager.FontManager(size=None, weight='normal') 
fm.defaultFont
```




    {'ttf': 'd:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\DejaVuSans.ttf',
     'afm': 'd:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\pdfcorefonts\\Helvetica.afm'}



#### 查找后使用


```python
# 使用fm查找目录下的字体
## FontManager.findfont(self, prop, fontext='ttf', directory=None, fallback_to_default=True, rebuild_if_missing=True)

mpl.rcParams['font.fantasy'] = 'Comic Neue'

la = mpl.font_manager.FontManager()# 实例化一个
lu = mpl.font_manager.FontProperties(family = 'monospace')# 可以不要
la.findfont(lu,directory= r"d:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf")
```




    'd:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\DejaVuSansMono.ttf'



#### 使用指定ttf


```python
## 指定一个ttf
lu = mpl.font_manager.FontProperties(
    fname ="d:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\sitka.ttc")
mpl.font_manager.FontManager().findfont(lu)
```




    'd:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\sitka.ttc'



### 实现对每个部分分别赋予字体

#### example1


```python
# https://matplotlib.org/gallery/api/font_file.html
import os
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fpath = os.path.join(rcParams["datapath"], "fonts/ttf/cmr10.ttf")
prop = fm.FontProperties(fname=fpath)
fname = os.path.split(fpath)[1]
ax.set_title('This is a special font: {}'.format(fname), fontproperties=prop)
ax.set_xlabel('This is the default font')

plt.show()
```


![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_setfont.png)



```python
fig,ax  = plt.subplots()
fpath = "d:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\sitka.ttc"
prop = fm.FontProperties(fname=fpath) 
fname = os.path.split(fpath)[1]

ax.set_title('This is a special font: {}'.format(fname), fontproperties=prop)
labels = ax.get_xticklabels() + ax.get_yticklabels() ## text class

[label.set_font_properties(prop) for label in labels]## set_font_prop
[label.set_bbox(dict(facecolor='red', alpha=1)) for label in labels]
[label.set_rotation("0") for label in labels]
[label.set_y(0.5) for label in labels]
[label.set_x(0.5) for label in labels];
```


![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_setfont2.png)


#### example2


```python
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
ax.set_axis_off()

font0 = FontProperties()
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
# Show family options

families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

font1 = font0.copy()
font1.set_size('large')

t = plt.figtext(0.1, 0.9, 'family', fontproperties=font1, **alignment)

yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

for k, family in enumerate(families):
    font = font0.copy()
    font.set_family(family)
    t = plt.figtext(0.1, yp[k], family, fontproperties=font, **alignment)

# Show style options

styles = ['normal', 'italic', 'oblique']

t = plt.figtext(0.3, 0.9, 'style', fontproperties=font1, **alignment)

for k, style in enumerate(styles):
    font = font0.copy()
    font.set_family('sans-serif')
    font.set_style(style)
    t = plt.figtext(0.3, yp[k], style, fontproperties=font, **alignment)

# Show variant options

variants = ['normal', 'small-caps']

t = plt.figtext(0.5, 0.9, 'variant', fontproperties=font1, **alignment)

for k, variant in enumerate(variants):
    font = font0.copy()
    font.set_family('serif')
    font.set_variant(variant)
    t = plt.figtext(0.5, yp[k], variant, fontproperties=font, **alignment)

# Show weight options

weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = plt.figtext(0.7, 0.9, 'weight', fontproperties=font1, **alignment)

for k, weight in enumerate(weights):
    font = font0.copy()
    font.set_weight(weight)
    t = plt.figtext(0.7, yp[k], weight, fontproperties=font, **alignment)

# Show size options

sizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = plt.figtext(0.9, 0.9, 'size', fontproperties=font1, **alignment)

for k, size in enumerate(sizes):
    font = font0.copy()
    font.set_size(size)
    t = plt.figtext(0.9, yp[k], size, fontproperties=font, **alignment)

# Show bold italic

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-small')
t = plt.figtext(0.3, 0.1, 'bold italic', fontproperties=font, **alignment)

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('medium')
t = plt.figtext(0.3, 0.2, 'bold italic', fontproperties=font, **alignment)

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-large')
t = plt.figtext(0.3, 0.3, 'bold italic', fontproperties=font, **alignment)

plt.show()
```


![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_setfont3.png)


#### 使用tex


```python
import os
import sys
import re
import gc
import matplotlib.pyplot as plt
import numpy as np

stests = [
    r'$\mathcircled{123} \mathrm{\mathcircled{123}}'
    r' \mathbf{\mathcircled{123}}$',
    r'$\mathsf{Sans \Omega} \mathrm{\mathsf{Sans \Omega}}'
    r' \mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard \pi}$',
    r'$\mathrm{\mathbb{Blackboard \pi}}$',
    r'$\mathbf{\mathbb{Blackboard \pi}}$',
    r'$\mathfrak{Fraktur} \mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$']
tests = stests

fig,ax = plt.subplots(figsize=(8, (len(tests) * 1) + 2))

ax.plot([0, 0], 'r')
ax.grid(False)
ax.axis([0, 3, -len(tests), 0])
ax.set_yticks(np.arange(len(tests)) * -1)

for i, s in enumerate(tests):
    ax.text(0.1, -i, s, fontsize=32)
# 使用tex创建text

```


![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_font_tex2.png)


#### tex2


```python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

def setfont(font):
    return r'\font\a %s at 14pt\a ' % font

for y, font, text in zip(range(5),
                         ['ptmr8r', 'ptmri8r', 'ptmro8r', 'ptmr8rn', 'ptmrr8re'],# font
                         ['Nimbus Roman No9 L ' + x for x in
                          ['', 'Italics (real italics for comparison)',
                           '(slanted)', '(condensed)', '(extended)']]):# text
    plt.text(0, y, setfont(font) + text)

plt.ylim(-1, 5)
plt.xlim(-0.2, 0.6)
plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=()) #Set a property on an artist object.
plt.title('Usetex font effects')
plt.show()
```


![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/mpl_font_tex1.png)



