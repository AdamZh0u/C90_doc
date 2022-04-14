
## Workflow
=== "Setup"
	```python 
	def setup_mpl(as_default=1):
		# My mpl setup

		if as_default: mpl.rcdefaults()
		else:
			# FONT
			mpl.rcParams["font.size"] = 7 # 10 default
			mpl.rcParams['font.family']='Helvetica 55 Roman' # sans-serif
			mpl.rcParams['legend.fontsize'] = 'small' # medium

			# TICK 
			mpl.rcParams['xtick.labelsize'] = 'small' # medium
			mpl.rcParams['ytick.labelsize'] = 'small' # medium
			mpl.rcParams['xtick.major.width'] = 2/3. # 0.8
			mpl.rcParams['ytick.major.width'] = 2/3.
			mpl.rcParams['xtick.minor.width'] = 2/3. # 0.6
			mpl.rcParams['ytick.minor.width'] = 2/3.
			mpl.rcParams['xtick.major.size'] = 3 # 3.5
			mpl.rcParams['ytick.major.size'] = 3
			mpl.rcParams['xtick.minor.size'] = 1.5 # 2
			mpl.rcParams['ytick.minor.size'] = 1.5
			mpl.rcParams['xtick.major.pad']='2.3' # 3.5
			mpl.rcParams['ytick.major.pad']='2.3'
			mpl.rcParams['xtick.minor.pad']='2.3' # 3.5
			mpl.rcParams['ytick.minor.pad']='2.3'
			mpl.rcParams['ytick.direction'] = 'in' # out
			mpl.rcParams['xtick.direction'] = 'in'
			mpl.rcParams['xtick.top']=True
			mpl.rcParams['ytick.right']=True

			# 
			mpl.rcParams['axes.linewidth'] = 2/3. # 0.8
			mpl.rcParams['axes.labelpad']= 2 # 4
			mpl.rcParams['lines.linewidth'] = 1 # 1.5
			mpl.rcParams['mathtext.default']='regular'

			# EXPORT 
			mpl.rcParams['figure.dpi'] = 400 # 100
			mpl.rcParams['svg.fonttype'] = "none"
			mpl.rcParams['figure.autolayout'] = True # tight_layout

			# PARAMS
			alpha = 0.6
			to_rgba = mpl.colors.ColorConverter().to_rgba
	```
=== "Functions"
	```python
	import matplotlib.pyplot as plt
	from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter, MaxNLocator

	fig, ax = plt.subplots(figsize = (3.60236*2,3.5))
	ax = axes.flatten()
	plt.subplots_adjust(hspace = 0.35,
						left=0.05,
						right=0.99,
						top=0.95,
						bottom=0.13)#, wspace = 0.35, right = 0.98, left = 0.1, bottom = 0.35, top = 0.9)

	fig.subplots_adjust(left=0, right=1, top=1, bottom=0,hspace=0.1, wspace=0.01)

	## ===============================================
	ax.axvline(ls = '--', color = 'C{}'.format(n-2))
	ax.axvspan(0, 48, alpha=0.3, color='grey')

	## ===============================================
	axes[r].tick_params("x",which = "major",direction = "in",
						length=1.5,width = 0.5 ,labelsize=10,rotation=0)

	labels = [item.get_text() for item in axes[r].get_xticklabels()]
	labels[1:-1] = ls_x
	axes[r].set_xticklabels(labels)

	## ===============================================
	ax.set_xscale('log')

	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_xlim(0.01,1000)
	ax.set_ylim(0,1)
	ax.get_ylim()
	ax.text(-0.1, 1.1, 'C', 
			transform=ax.transAxes, 
			fontsize = 8, 
			weight="bold", 
			fontdict={ 'family':'Helvetica 55 Roman'})

	plt.figlegend(ncol = 3,
						handlelength=1.5,
						handletextpad=0.4,
						bbox_transform = fig.transFigure,
						loc = 'lower center',
						mode = 'expand',
						bbox_to_anchor=(x1,0.01,x3-x1,0.1),
						borderaxespad=0)

	plt.savefig()
	```

###  Hist
```python
for n, group in data_size.groupby('level'):
    values = np.log10(group['size'].values)
    
    hist, edges = np.histogram(values, bins = np.linspace(2,7,25), density = True)
    x,y = zip(*[(k,k2) for (k,k2) in zip(edges,hist) if k2>0])
    ax2.plot([10**i for i in x], y, label =r'$s={}$'.format(n), color = 'C{}'.format(n-2))    
    ax2.axvline(10**np.mean(values), color = 'C{}'.format(n-2),label = r'$\overline{x}=%.1f$ Km'%(10**np.mean(values)/1000), ls = '--')
    ys.append(hist)
```

###
```python
import matplotlib.pyplot as plt
    
plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices, color='red')

x_lines = []
y_lines = []

for p1,p2 in edges:
    x1,y1 = points[p1]
    x2,y2 = points[p2]
    plt.plot([x1,x2],[y1,y2], color='blue')

plt.scatter(points[:,0],points[:,1])

plt.show()

```

## WHU

### loglogplot
=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/loglogplot.png)	
=== "Code"

	``` python
	def loglogplot(x,y,labelx="LogPop",labely="LogArea"):
		plt.style.use('ggplot')
		xd,yd=np.log10(x),np.log10(y)
		# make the scatter plot
		fig, ax = plt.subplots(figsize=(5,4))

		ax.set_yscale("log")
		ax.set_xscale("log")
		#ax.set_ylim(1e1,10**3.4)
		#ax.set_xlim(10**(-0.1),10**(2.5))
		# determine best fit line
		par = np.polyfit(xd, yd, 1,full=True)

		slope=par[0][0]
		intercept=par[0][1]

		xl = [min(xd), max(xd)]
		yl = [slope*xx + intercept  for xx in xl]
		ax.plot([10**xx for xx in xl],[10**yy for yy in yl], '-m')

		# coefficient of determination, plot text
		variance = np.var(yd)#方差
		residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])#残差

		Rsqr = np.round(1-residuals/variance, decimals=2)
		#ax.text(0.3*max(x),1.2*max(y),r'$R^2 ={0.2f} \n Slope={0.2f}'.format{Rsqr,slope.round(2)}, fontsize=15)
		ax.text(min(x),1.1*max(y),'%s =%0.2f\nSlope=%0.2f\nIntercept=%0.2f'%("$\sf{R^2}$",Rsqr,slope.round(2),intercept), fontsize=8)
		print(variance,residuals)
		plt.xlabel(labelx)
		plt.ylabel(labely)

		# error bounds
		yerr = [abs(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)]
		par = np.polyfit(xd, yerr, 2, full=True)
		erro_x2 = par[0][0]
		erro_x1 = par[0][1]
		erro_x0 = par[0][2]
		yerrUpper = [(xx*slope+intercept)+(erro_x2*xx**2 + erro_x1*xx + erro_x0) for xx,yy in zip(xd,yd)]
		yerrLower = [(xx*slope+intercept)-(erro_x2*xx**2 + erro_x1*xx + erro_x0) for xx,yy in zip(xd,yd)]
		print(erro_x2, erro_x1, erro_x0)

		#     ax.plot([10**xx for xx in xd],[10**yy for yy in yerrLower], 'm')
		#     ax.plot([10**xx for xx in xd],[10**yy for yy in yerrUpper], 'm')
		ax.scatter(x, y, s=10, alpha=1, marker='h',color="orangered")

		yline = [slope*xl[0] + intercept,xl[1] + intercept]
		#yline2 = [xl[0] + intercept,slope*xl[1] + intercept]
		ax.plot([10**xx for xx in xl],[10**yy for yy in yline], '--m')
		#ax.plot([10**xx for xx in xl],[10**yy for yy in yline2], '--m')
		ax.text(250,660, "Slope=1", size = 5,\
				family = "fantasy", color = "r", style = "italic", weight = "light",\
				)#bbox = dict(facecolor = "r", alpha = 0.2)
		return slope,ax
	```



### Scientific style plot
=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/CA_FI-rat0_2.0.png)
=== "Code"
	``` python
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import matplotlib.ticker as mtick
	from mpltools import annotation

	plt.style.use("science")
	fig = plt.figure(figsize=(3.5, 2.625),dpi=600)
	ax = fig.add_subplot(1,1,1)

	ax.set_xlim((-0.6,0.6))
	ax.set_ylim((-0.1,0.1))
	ax.plot([-0.6,0.6],[0,0], linewidth=0.8, color='black' )
	ax.plot([0,0],[-0.1,0.1], linewidth=0.8, color='black' )
	#ax.scatter(x_CA,y_FI,s=1,marker='o')
	ax.plot(x_FI,y_CA, linewidth=0,ms=2,
			marker='o', markerfacecolor='w',markeredgecolor='k',markeredgewidth=0.5,zorder=30)

	ax.set_xlabel("FI")
	ax.set_ylabel("CA")

	plt.annotate('9', xy=(-0.20, 0.03), xytext=(-0.30, 0.05),
					arrowprops=dict(facecolor='black',arrowstyle="->"))
	plt.annotate('10', xy=(0.23, -0.08), xytext=(0.051, -0.09),
					arrowprops=dict(facecolor='black',arrowstyle="->"))
	plt.annotate('18', xy=(0.548, -0.029), xytext=(0.41,-0.015),
					arrowprops=dict(facecolor='black',arrowstyle="->"))
	# plt.annotate('23', xy=(0.099, 0.006), xytext=(0.2,0.02),
	#              arrowprops=dict(facecolor='black',arrowstyle="->"))
	plt.annotate('5', xy=(0.27, -0.06), xytext=(0.39,-0.07),
					arrowprops=dict(facecolor='black',arrowstyle="->"))


	ax.grid(linestyle="--", linewidth=0.2, color='.25', zorder=50,alpha=0.5)
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:3.0f}\%'.format(x*100) for x in vals])
	vals = ax.get_xticks()
	ax.set_xticklabels(['{:3.0f}\%'.format(x*100) for x in vals])

	par = np.polyfit(x_FI, y_CA, 1,full=True)
	slope=par[0][0]
	intercept=par[0][1]
	xl = [-0.5, max(x_FI)]
	yl = [slope*xx + intercept  for xx in xl]
	ax.plot([xx for xx in xl],[yy for yy in yl], '--k',zorder=20)

	variance = np.var(y_CA)#方差
	residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(x_FI,y_CA)])#残差
	Rsqr = np.round(1-residuals/variance, decimals=2)
	ax.text(0.35,0.08,'%s=%0.2f\nSlope=%0.2f'%("${R^2}$",Rsqr,slope.round(2)), fontsize=6)

	# annotation.slope_marker((-0.4, 0.03), -0.11,
	#                         text_kwargs={'color': 'k'},
	#                         poly_kwargs={'facecolor': "k"})

	# \sf
	plt.show()
	fig.savefig("Four-quadrant.png",dpi=600)
	```

### HDI-LDI


=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/HDI-LDI-plot_8_31.png)
=== "Code"
	``` python
	import matplotlib.pyplot as plt
	import numpy as np
	import scipy.stats as stats
	import pandas as pd
	from matplotlib.font_manager import FontProperties
	from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

	fig, ax1 = plt.subplots(figsize = (3.5,2.625),dpi=200)
	#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params
	B,=ax1.plot(i, x,"^k",ls="",lw=1,ms=2,label="HDI")

	#ax1.minorticks_on()
	ax1.tick_params("x",which = "major",direction = "in" ,
					length=3,width = 0.5,labelrotation=90,labelsize=6)
	ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
	# ax1.tick_params("x",which = "minor",direction = "in",
	#                 length=3,width = 0.5, bottom = True, top=True,
	#                 labelbottom=True)
	xticks = [i for i in range(1,37)]
	ax1.set_xlim(0,37)
	ax1.set_ylim(0.3,1.3)
	ax1.set_xticklabels(a,size=6)


	# ax1 y
	ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
	ax1.tick_params("y",which = "major",direction = "in",
				length=3,width = 0.5,right=True ,labelsize=6)
	# def minor_tick(x, pos):
	#     if not x % 1.0:
	#         return ""
	#     return "%.2f" % x

	# ax1.yaxis.set_minor_formatter(FuncFormatter(minor_tick))
	ax1.tick_params("y",which = "minor",direction = "in",
				length=1.5,width = 0.5,right=True ,labelsize=6)
	labels = ax1.get_xticklabels() + ax1.get_yticklabels()
	[label.set_fontname('Times New Roman') for label in labels]


	## ax2
	ax2 = ax1.twinx()
	A,=ax2.plot(i,y,"d--k",lw=1,ms=2,label="LDI")
	ax2.set_ylim(0.3,1.3)
	ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
	ax2.tick_params("y",which = "major",direction = "in",
				length=3,width = 0.5,right=True ,labelsize=6)
	labels = ax2.get_yticklabels()
	[label.set_fontname('Times New Roman') for label in labels]
	# def minor_tick(x, pos):
	#     if not x % 1.0:
	#         return ""
	#     return "%.2f" % x

	# ax1.yaxis.set_minor_formatter(FuncFormatter(minor_tick))
	ax2.tick_params("y",which = "minor",direction = "in",
				length=1.5,width = 0.5,right=True ,labelsize=5)


	ax1.grid(which="major",axis="y",lw=0.4)

	font1 = {'family' : 'Times New Roman',
	'weight' : 'normal',
	'size'   : 6}

	ax1.set_ylabel("HDI",font1,size=8)
	ax2.set_ylabel("LDI",font1,size=8)
	ax1.set_xlabel("Hotspots",font1,size=8)

	ax1.legend(handles=[A,B],prop=font1,frameon=False,loc="lower left")
	fig.savefig("HDI-LDI-plot_8_31.png",dpi=1000)
	```

### Proportion and change rate
=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/Nature_change-9.2.png)
=== "Code"
	``` python
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import gridspec
	from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter,MaxNLocator

	df1_gp=pd.DataFrame()
	df2=pd.DataFrame()
	su={}
	for i in range(36):
		df1 = pd.read_excel("path_to_file.xlsx",sheet_name="ID_{}".format(i)).set_index("ID")
		S1 = df1.loc[6,:].rename(i)# nature
		df2 = df2.append(S1)
		S = (df1.loc[6,:]/df1.loc[6,1992]-1).rename(i)
		df1_gp=df1_gp.append(S)
		S2=sum(df1.loc[[5,6,7],1992])
		su[i] =S2

		df2_num=df2
		for i in range(36):
			df2_num.loc[i,1992]=df2_num.loc[i,1992]/su[i]

			df2_num=df2_num.sort_values(by=1992)
			index = list(df2_num.index)
			# df1_reindex=df1_gp.reindex(index)
			# df1_reindex.describe()

			def ax_y_settings(ax, var_name, x_min, x_max):
				ax.set_xlim(x_min,x_max)
				#ax.set_ylim(y_min,y_max)
				ax.set_yticks([])
				#ax.spines['left'].set_visible(False)
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				#ax.spines['bottom'].set_visible(False)
				#ax.spines['bottom'].set_edgecolor(='#444444')
				ax.spines['bottom'].set_linewidth(0)
				ax.spines['left'].set_linewidth(0.3)
				ax.text(0.01, 0.3, var_name, font1, transform = ax.transAxes)
				return None

			fig = plt.figure(figsize=(3.267,4.5),dpi=1000)

			number_gp=36
			gs0 = gridspec.GridSpec(nrows=1,
									ncols=2,
									figure=fig,
									width_ratios= [1,3],
									wspace=0, hspace=0
								   )
			gs0.tight_layout(fig,pad=0)
			#height_ratios= [1]*number_gp

			ax = [None]*(number_gp + 1)## important

			font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 6}
			cmap1 = plt.get_cmap("summer")

			gs01=gs0[1].subgridspec(number_gp,1)
			#https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py

			##ax0
			ax[0] = fig.add_subplot(gs0[0])
			ax[0].spines['right'].set_visible(False)
			ax[0].spines['left'].set_visible(False)
			ax[0].spines['top'].set_visible(False)
			ax[0].spines['bottom'].set_linewidth(0.3)

			perc= df2_num.iloc[:,0]
			features = [i+1 for i in range(number_gp)]

			ax[0].barh(features, -1*perc, color=cmap1(0.1), height=0.4)

			ax[0].invert_yaxis()
			ax[0].set_yticks([])
			ax[0].set_ylim([36.5,0.5])

			ax[0].xaxis.set_major_locator(MultipleLocator(0.5))
			b = ["","100%","50%","-----"]
			ax[0].set_xticklabels(b,font1)
			ax[0].tick_params("x",which = "major",direction = "in",
							  length=1.5,width = 0.5 ,labelsize=6,rotation=90)
			ax[0].set_title("Proportion",font1)


			## ax36
			for i in range(number_gp):
				ax[i+1] = fig.add_subplot(gs01[i,0])
				ax_y_settings(ax[i+1],index[i]+1,-0.6,0.16)

				rc = ax[i+1].scatter(df1_reindex.iloc[i,:],[0]*24,
									 c=[i for i in range(24)],cmap=cmap1,
									 marker = "o",s=4,
									 lw=0.1,edgecolors="k",
									 zorder=20)
				#ax[i].stackplot(df1_gp.columns,df1_gp.loc[i,:])
				#sns.kdeplot(data=df1_gp.loc[i,:],ax=ax[i], shade=True, color="blue",  bw=300, legend=False)
				#ax[i].plot([-0.08,0.05],[0,0],"--k",lw=0.1)
				ax[i+1].axhline(0,0.125,1,ls="--",c="k",lw=0.2,zorder=10)
				ax[i+1].plot([0,0],[-1,1],"-k",lw=0.2,zorder=10)
				if i < (number_gp - 1):#1-35
					ax[i+1].set_xticks([])
					if i == 0:
						ax[i+1].set_title("Change rate compared to 1992 ",font1)
						else:#36
							ax[i+1].spines['bottom'].set_linewidth(0.3)
							ax[i+1].spines['bottom'].set_edgecolor('k')

							a1 = [-0.08,-0.06,-0.04,-0.02,0,0.02,0.04]
							a = ["",""]+['{:3.0f}%'.format(x*100) for x in a1]
							ax[i+1].set_xticklabels(a,font1,size=6,)
							ax[i+1].xaxis.set_major_locator(MultipleLocator(0.02))
							ax[i+1].tick_params("x",which = "major",direction = "in",
												length=1.5,width = 0.5 ,labelsize=6,rotation=90)
							#ax[i].plot([1992,2015],[0,0],"--k")

							# colorbar

							cbar = fig.colorbar(rc,ax=[ax[i] for i in range(1,37)],shrink=0.3,
												drawedges=False)
							cbar.ax.get_yaxis().set_major_locator(MultipleLocator(23))
							#cbar.ax.get_yaxis().set_ticklabels(["","1992","2015",""])
							cbar.ax.set_yticklabels(["","1992","2015",""],font1,rotation=270)
							cbar.ax.set_ylabel('Year', font1,rotation=270)
							cbar.ax.tick_params("y",which = "major",direction = "in",
												length=0,width = 0.5 ,labelsize=6,rotation=270)

							fig.savefig("demo1.png",bbox_inches="tight",dpi=1200,pad_inches=0)

	```

### legend

``` python
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig, ax5 = plt.subplots()

x = [1,2,3]
y = [2,3,5]
pop = [2,4,5]

scatter=ax5.scatter(x=x, y=y, s=pop,c="white",edgecolor="black")
handles, labels = scatter.legend_elements(prop="sizes",c="black")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                        mec = "b",mfc='w', markersize=15),
                    Line2D([0], [0], marker='o', color='w', label='Scatter',
                    mec = "b",mfc='w', markersize=14),
                    Line2D([0], [0], marker='o', color='w', label='Scatter',
                    mec = "b",mfc='w', markersize=23)]

ax5.set_xlabel("Rank of Per capita built-up area",fontdict={'family':'Times New Roman','size':16})
ax5.set_ylabel("Rank of LIP",fontdict={'family':'Times New Roman','size':16})
ax5.legend(handles = legend_elements,
    frameon=False,
    loc='lower right',title="demo",ncol=2,fontsize=12,title_fontsize=12)
ax5.text(-0.12,0.95,"(e)",transform=ax5.transAxes,fontdict={'family':'Times New Roman','size':16})
plt.show()
```

### OLS log plot
$$
\begin{aligned}
\begin{eqnarray}
y &=& C\times x^\alpha + m  \\
y &\stackrel{\text{i.i.d}}{\sim}& \mathcal{N}(\hat{y}, \sigma^2) \\
\    \\
\log y &=& \alpha \log x+\log C \\
\log y &\not \stackrel{\text{i.i.d}}{\sim}& \mathcal{N}(\hat{\log y}, \sigma^2)
\end{eqnarray}
\end{aligned}
$$

=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/OLS_Log1.png)
	
=== "Code"

	``` python
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	from scipy.optimize import curve_fit

	def paramsPlot(xdata,ydata):
		## curve fit
		popt, pcov = curve_fit(func, xdata, ydata)

		yhat1 = [func(i, popt[0],popt[1]) for i in xdata]
		print("NLS Fit C:",popt[0],"\nNLS Fit Alpha===",popt[1])
		print("*"*60)

		## ols fit
		xd = np.log(xdata)
		yd = np.log(ydata)

		par= np.polyfit(xd,yd,1)
		k = par[0]
		m = par[1]

		logyhat = k*xd+m
		yhat2 = np.exp(logyhat)
		print("OLS Fit C:",m ,"\nOLS Fit Alpha===",k)
		print("*"*60)

		## plot
		fig,ax = plt.subplots(1,2,figsize = (12,5))

		ax[0].scatter(xdata,ydata,s=10)
		ax[0].plot(xdata,yhat1,'r--',label ="NLS line" )
		ax[0].plot(xdata,yhat2,"g--",label ="OLS line")## OLS fit
		ax[0].legend()
		ax[0].grid()
		ax[1].set_xlim(0.7*min(xdata),1.3*max(xdata))
		ax[1].set_ylim(0.7*min(ydata),1.3*max(ydata))

		ax[1].set_yscale("log")
		ax[1].set_xscale("log")
		ax[1].scatter(xdata,ydata,s=10)
		ax[1].loglog(xdata,yhat1,'r--',label ="NLS line" )
		ax[1].loglog(xdata,yhat2,"g--",label ="OLS line")## OLS fit
		ax[1].grid(which = "both")
		ax[1].legend()
		ax[1].set_xlim(0.7*min(xdata),1.3*max(xdata))
		ax[1].set_ylim(0.7*min(ydata),1.3*max(ydata))

		return fig,ax
	## 使用数据
	df = pd.read_excel("01全国96指标数据.xlsx",index_col = 0)
	data = df[["Pop","BuiltUpArea"]].dropna().sort_values(by = "Pop",ascending = False).reset_index(drop= True)
	data = df[["Pop","Employment"]].dropna().sort_values(by = "Pop",ascending = False).reset_index(drop= True)
	data = df[["Pop","GDP"]].dropna().sort_values(by = "Pop",ascending = False).reset_index(drop= True)

	xdata = data.iloc[:,0]
	ydata = data.iloc[:,1]

	fig,ax = paramsPlot(xdata,ydata)

	## 生成数据
	def func(x, a, b):
		return a * x**b

	a = 2.5
	b = 0.75

	noise_sigma = 100

	xdata = np.linspace(10, 10000, 100) # truex
	# xdata = np.logspace(1, 20, 100)

	y = func(xdata,a,b) # true y
	ydata = y + np.random.normal(0,noise_sigma,size=len(xdata)) # noise y
	index = ~(ydata<0)

	print("True C:",a,"\nTrue Alpha===",b)
	print("*"*60)

	## ols fit
	index = ~(ydata<0)

	fig,ax = paramsPlot(xdata[index],ydata[index])
	```

## CASA
## PKU


## Configures


### Colors

#### Named Colors

=== "Output"
	![](https://az-image-1310475420.cos.ap-guangzhou.myqcloud.com/pic/named_colors.png)
=== "Code"
	``` python
	"""
	========================
	Visualizing named colors
	========================
	Simple plot example with the named colors and its visual representation.
	"""
	from __future__ import division

	import matplotlib.pyplot as plt
	from matplotlib import colors as mcolors


	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	# Sort colors by hue, saturation, value and name.
	by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
					for name, color in colors.items())
	sorted_names = [name for hsv, name in by_hsv]

	n = len(sorted_names)
	ncols = 4
	nrows = n // ncols + 1

	fig, ax = plt.subplots(figsize=(8, 5))

	# Get height and width
	X, Y = fig.get_dpi() * fig.get_size_inches()
	h = Y / (nrows + 1)
	w = X / ncols

	for i, name in enumerate(sorted_names):
		col = i % ncols
		row = i // ncols
		y = Y - (row * h) - h

		xi_line = w * (col + 0.05)
		xf_line = w * (col + 0.25)
		xi_text = w * (col + 0.3)

		ax.text(xi_text, y, name, fontsize=(h * 0.8),
				horizontalalignment='left',
				verticalalignment='center')

		ax.hlines(y + h * 0.1, xi_line, xf_line,
				color=colors[name], linewidth=(h * 0.6))

	ax.set_xlim(0, X)
	ax.set_ylim(0, Y)
	ax.set_axis_off()

	fig.subplots_adjust(left=0, right=1,
						top=1, bottom=0,
						hspace=0, wspace=0)
	plt.show()
	```


## Default Settings

### Settings

```python
# https://matplotlib.org/api/matplotlib_configuration_api.html
# \Lib\site-packages\matplotlib\mpl-data\matplotlibrc
## backend
mpl.use("TkAgg")
mpl.interactive(False)

## Default values and styling
mpl.rcParams  # 显示rc配置
mpl.matplotlib_fname()   # 显示rc文件位置
mpl.rc(group, **kwargs)
mpl.rcdefaults()  # 默认rc
mpl.rc_file(fname, *, use_default_template=True) # "~\.matplotlib\matplotlibrc"
mpl.rc_context(rc=None, fname=None) # with mpl.rc_context(rc={'text.usetex': True}, fname='screen.rc'):plt.plot(x, a)

###  set rc 
mpl.rc('lines', linewidth=2, color='r') ## 
font = {'family' : 'monospace','weight' : 'bold','size' : 12}
mpl.rc('font', **font)  # pass in the font dict as kwargs

## logging
mpl.set_loglevel("info")# {"notset", "debug", "info", "warning", "error", "critical"}
```
	

### Styles

=== "Scientific style"
	``` python
	# .matplotlib\stylelib\science.mplstyle
	# Matplotlib style for general scientific plots

	# Set color cycle
	axes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])

	# Set default figure size
	figure.figsize : 3.5, 2.625

	# Set x axis
	xtick.direction : in
	xtick.major.size : 3
	xtick.major.width : 0.5
	xtick.minor.size : 1.5
	xtick.minor.width : 0.5
	xtick.minor.visible :   True
	xtick.top : True

	# Set y axis
	ytick.direction : in
	ytick.major.size : 3
	ytick.major.width : 0.5
	ytick.minor.size : 1.5
	ytick.minor.width : 0.5
	ytick.minor.visible :   True
	ytick.right : True

	# Set line widths
	axes.linewidth : 0.5
	grid.linewidth : 0.5
	lines.linewidth : 1.

	# Remove legend frame
	legend.frameon : False

	# Always save as 'tight'
	savefig.bbox : tight
	savefig.pad_inches : 0.05

	# Use serif fonts
	font.serif : Times New Roman
	font.family : serif

	# Use LaTeX for math formatting
	text.usetex : True
	text.latex.preamble : \usepackage{amsmath} \usepackage[T1]{fontenc}
	```
=== "IEEE style"
	``` python
	# Matplotlib style for IEEE plots
	# This style should work for most two-column journals

	# Set color cycle
	# Set line style as well for black and white graphs
	axes.prop_cycle : (cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.']))

	# Set default figure size
	figure.figsize : 3.3, 2.5
	figure.dpi : 600

	# Font sizes
	font.size : 8
	```
=== "Default style"
	```python
	### MATPLOTLIBRC FORMAT

	# NOTE FOR END USERS: DO NOT EDIT THIS FILE!
	#
	# This is a sample Matplotlib configuration file - you can find a copy
	# of it on your system in site-packages/matplotlib/mpl-data/matplotlibrc
	# (relative to your Python installation location).
	#
	# You should find a copy of it on your system at
	# site-packages/matplotlib/mpl-data/matplotlibrc (relative to your Python
	# installation location).  DO NOT EDIT IT!
	#
	# If you wish to change your default style, copy this file to one of the
	# following locations:
	#     Unix/Linux:
	#         $HOME/.config/matplotlib/matplotlibrc OR
	#         $XDG_CONFIG_HOME/matplotlib/matplotlibrc (if $XDG_CONFIG_HOME is set)
	#     Other platforms:
	#         $HOME/.matplotlib/matplotlibrc
	# and edit that copy.
	#
	# See https://matplotlib.org/users/customizing.html#the-matplotlibrc-file
	# for more details on the paths which are checked for the configuration file.
	#
	# Blank lines, or lines starting with a comment symbol, are ignored, as are
	# trailing comments.  Other lines must have the format:
	#     key: val  # optional comment
	#
	# Formatting: Use PEP8-like style (as enforced in the rest of the codebase).
	# All lines start with an additional '#', so that removing all leading '#'s
	# yields a valid style file.
	#
	# Colors: for the color values below, you can either use
	#     - a Matplotlib color string, such as r, k, or b
	#     - an RGB tuple, such as (1.0, 0.5, 0.0)
	#     - a hex string, such as ff00ff
	#     - a scalar grayscale intensity such as 0.75
	#     - a legal html color name, e.g., red, blue, darkslategray
	#
	# Matplotlib configuration are currently divided into following parts:
	#     - BACKENDS
	#     - LINES
	#     - PATCHES
	#     - HATCHES
	#     - BOXPLOT
	#     - FONT
	#     - TEXT
	#     - LaTeX
	#     - AXES
	#     - DATES
	#     - TICKS
	#     - GRIDS
	#     - LEGEND
	#     - FIGURE
	#     - IMAGES
	#     - CONTOUR PLOTS
	#     - ERRORBAR PLOTS
	#     - HISTOGRAM PLOTS
	#     - SCATTER PLOTS
	#     - AGG RENDERING
	#     - PATHS
	#     - SAVING FIGURES
	#     - INTERACTIVE KEYMAPS
	#     - ANIMATION

	#### CONFIGURATION BEGINS HERE


	# ***************************************************************************
	# * BACKENDS                                                                *
	# ***************************************************************************
	# The default backend.  If you omit this parameter, the first working
	# backend from the following list is used:
	#     MacOSX QtAgg Gtk4Agg Gtk3Agg TkAgg WxAgg Agg
	# Other choices include:
	#     QtCairo GTK4Cairo GTK3Cairo TkCairo WxCairo Cairo
	#     Qt5Agg Qt5Cairo Wx  # deprecated.
	#     PS PDF SVG Template
	# You can also deploy your own backend outside of Matplotlib by referring to
	# the module name (which must be in the PYTHONPATH) as 'module://my_backend'.
	#backend: Agg
	# The port to use for the web server in the WebAgg backend.
	webagg.port: 8988

	# The address on which the WebAgg web server should be reachable
	webagg.address: 127.0.0.1

	# If webagg.port is unavailable, a number of other random ports will
	# be tried until one that is available is found.
	webagg.port_retries: 50

	# When True, open the web browser to the plot that is shown
	webagg.open_in_browser: True

	# If you are running pyplot inside a GUI and your backend choice
	# conflicts, we will automatically try to find a compatible one for
	# you if backend_fallback is True
	backend_fallback: True

	interactive: False
	toolbar:     toolbar2  # {None, toolbar2, toolmanager}
	timezone:    UTC       # a pytz timezone string, e.g., US/Central or Europe/Paris


	# ***************************************************************************
	# * LINES                                                                   *
	# ***************************************************************************
	# See https://matplotlib.org/api/artist_api.html#module-matplotlib.lines
	# for more information on line properties.
	lines.linewidth: 1.5               # line width in points
	lines.linestyle: -                 # solid line
	lines.color:     C0                # has no affect on plot(); see axes.prop_cycle
	lines.marker:          None        # the default marker
	lines.markerfacecolor: auto        # the default marker face color
	lines.markeredgecolor: auto        # the default marker edge color
	lines.markeredgewidth: 1.0         # the line width around the marker symbol
	lines.markersize:      6           # marker size, in points
	lines.dash_joinstyle:  round       # {miter, round, bevel}
	lines.dash_capstyle:   butt        # {butt, round, projecting}
	lines.solid_joinstyle: round       # {miter, round, bevel}
	lines.solid_capstyle:  projecting  # {butt, round, projecting}
	lines.antialiased: True            # render lines in antialiased (no jaggies)

	# The three standard dash patterns.  These are scaled by the linewidth.
	lines.dashed_pattern: 3.7, 1.6
	lines.dashdot_pattern: 6.4, 1.6, 1, 1.6
	lines.dotted_pattern: 1, 1.65
	lines.scale_dashes: True

	markers.fillstyle: full  # {full, left, right, bottom, top, none}

	pcolor.shading: auto
	pcolormesh.snap: True  # Whether to snap the mesh to pixel boundaries. This is
							provided solely to allow old test images to remain
							unchanged. Set to False to obtain the previous behavior.

	# ***************************************************************************
	# * PATCHES                                                                 *
	# ***************************************************************************
	# Patches are graphical objects that fill 2D space, like polygons or circles.
	# See https://matplotlib.org/api/artist_api.html#module-matplotlib.patches
	# for more information on patch properties.
	patch.linewidth:       1      # edge width in points.
	patch.facecolor:       C0
	patch.edgecolor:       black  # if forced, or patch is not filled
	patch.force_edgecolor: False  # True to always use edgecolor
	patch.antialiased:     True   # render patches in antialiased (no jaggies)


	# ***************************************************************************
	# * HATCHES                                                                 *
	# ***************************************************************************
	hatch.color:     black
	hatch.linewidth: 1.0


	# ***************************************************************************
	# * BOXPLOT                                                                 *
	# ***************************************************************************
	boxplot.notch:       False
	boxplot.vertical:    True
	boxplot.whiskers:    1.5
	boxplot.bootstrap:   None
	boxplot.patchartist: False
	boxplot.showmeans:   False
	boxplot.showcaps:    True
	boxplot.showbox:     True
	boxplot.showfliers:  True
	boxplot.meanline:    False

	boxplot.flierprops.color:           black
	boxplot.flierprops.marker:          o
	boxplot.flierprops.markerfacecolor: none
	boxplot.flierprops.markeredgecolor: black
	boxplot.flierprops.markeredgewidth: 1.0
	boxplot.flierprops.markersize:      6
	boxplot.flierprops.linestyle:       none
	boxplot.flierprops.linewidth:       1.0

	boxplot.boxprops.color:     black
	boxplot.boxprops.linewidth: 1.0
	boxplot.boxprops.linestyle: -

	boxplot.whiskerprops.color:     black
	boxplot.whiskerprops.linewidth: 1.0
	boxplot.whiskerprops.linestyle: -

	boxplot.capprops.color:     black
	boxplot.capprops.linewidth: 1.0
	boxplot.capprops.linestyle: -

	boxplot.medianprops.color:     C1
	boxplot.medianprops.linewidth: 1.0
	boxplot.medianprops.linestyle: -

	boxplot.meanprops.color:           C2
	boxplot.meanprops.marker:          ^
	boxplot.meanprops.markerfacecolor: C2
	boxplot.meanprops.markeredgecolor: C2
	boxplot.meanprops.markersize:       6
	boxplot.meanprops.linestyle:       --
	boxplot.meanprops.linewidth:       1.0


	# ***************************************************************************
	# * FONT                                                                    *
	# ***************************************************************************
	# The font properties used by `text.Text`.
	# See https://matplotlib.org/api/font_manager_api.html for more information
	# on font properties.  The 6 font properties used for font matching are
	# given below with their default values.
	#
	# The font.family property can take either a concrete font name (not supported
	# when rendering text with usetex), or one of the following five generic
	# values:
	#     - 'serif' (e.g., Times),
	#     - 'sans-serif' (e.g., Helvetica),
	#     - 'cursive' (e.g., Zapf-Chancery),
	#     - 'fantasy' (e.g., Western), and
	#     - 'monospace' (e.g., Courier).
	# Each of these values has a corresponding default list of font names
	# (font.serif, etc.); the first available font in the list is used.  Note that
	# for font.serif, font.sans-serif, and font.monospace, the first element of
	# the list (a DejaVu font) will always be used because DejaVu is shipped with
	# Matplotlib and is thus guaranteed to be available; the other entries are
	# left as examples of other possible values.
	#
	# The font.style property has three values: normal (or roman), italic
	# or oblique.  The oblique style will be used for italic, if it is not
	# present.
	#
	# The font.variant property has two values: normal or small-caps.  For
	# TrueType fonts, which are scalable fonts, small-caps is equivalent
	# to using a font size of 'smaller', or about 83%% of the current font
	# size.
	#
	# The font.weight property has effectively 13 values: normal, bold,
	# bolder, lighter, 100, 200, 300, ..., 900.  Normal is the same as
	# 400, and bold is 700.  bolder and lighter are relative values with
	# respect to the current weight.
	#
	# The font.stretch property has 11 values: ultra-condensed,
	# extra-condensed, condensed, semi-condensed, normal, semi-expanded,
	# expanded, extra-expanded, ultra-expanded, wider, and narrower.  This
	# property is not currently implemented.
	#
	# The font.size property is the default font size for text, given in points.
	# 10 pt is the standard value.
	#
	# Note that font.size controls default text sizes.  To configure
	# special text sizes tick labels, axes, labels, title, etc., see the rc
	# settings for axes and ticks.  Special text sizes can be defined
	# relative to font.size, using the following values: xx-small, x-small,
	# small, medium, large, x-large, xx-large, larger, or smaller

	font.family:  sans-serif
	font.style:   normal
	font.variant: normal
	font.weight:  normal
	font.stretch: normal
	font.size:    10.0

	font.serif:      DejaVu Serif, Bitstream Vera Serif, Computer Modern Roman, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif
	font.sans-serif: DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
	font.cursive:    Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, Comic Neue, Comic Sans MS, cursive
	font.fantasy:    Chicago, Charcoal, Impact, Western, Humor Sans, xkcd, fantasy
	font.monospace:  DejaVu Sans Mono, Bitstream Vera Sans Mono, Computer Modern Typewriter, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace


	# ***************************************************************************
	# * TEXT                                                                    *
	# ***************************************************************************
	# The text properties used by `text.Text`.
	# See https://matplotlib.org/api/artist_api.html#module-matplotlib.text
	# for more information on text properties
	text.color: black

	# FreeType hinting flag ("foo" corresponds to FT_LOAD_FOO); may be one of the
	# following (Proprietary Matplotlib-specific synonyms are given in parentheses,
	# but their use is discouraged):
	# - default: Use the font's native hinter if possible, else FreeType's auto-hinter.
	#            ("either" is a synonym).
	# - no_autohint: Use the font's native hinter if possible, else don't hint.
	#                ("native" is a synonym.)
	# - force_autohint: Use FreeType's auto-hinter.  ("auto" is a synonym.)
	# - no_hinting: Disable hinting.  ("none" is a synonym.)
	text.hinting: force_autohint

	text.hinting_factor: 8  # Specifies the amount of softness for hinting in the
							 horizontal direction.  A value of 1 will hint to full
							 pixels.  A value of 2 will hint to half pixels etc.
	text.kerning_factor: 0  # Specifies the scaling factor for kerning values.  This
							 is provided solely to allow old test images to remain
							 unchanged.  Set to 6 to obtain previous behavior.
							 Values  other than 0 or 6 have no defined meaning.
	text.antialiased: True  # If True (default), the text will be antialiased.
							 This only affects raster outputs.


	# ***************************************************************************
	# * LaTeX                                                                   *
	# ***************************************************************************
	# For more information on LaTeX properties, see
	# https://matplotlib.org/tutorials/text/usetex.html
	text.usetex: False  # use latex for all text handling. The following fonts
						 are supported through the usual rc parameter settings:
						 new century schoolbook, bookman, times, palatino,
						 zapf chancery, charter, serif, sans-serif, helvetica,
						 avant garde, courier, monospace, computer modern roman,
						 computer modern sans serif, computer modern typewriter
	text.latex.preamble:   # IMPROPER USE OF THIS FEATURE WILL LEAD TO LATEX FAILURES
							AND IS THEREFORE UNSUPPORTED. PLEASE DO NOT ASK FOR HELP
							IF THIS FEATURE DOES NOT DO WHAT YOU EXPECT IT TO.
							text.latex.preamble is a single line of LaTeX code that
							will be passed on to the LaTeX system. It may contain
							any code that is valid for the LaTeX "preamble", i.e.
							between the "\documentclass" and "\begin{document}"
							statements.
							Note that it has to be put on a single line, which may
							become quite long.
							The following packages are always loaded with usetex,
							so beware of package collisions:
							  geometry, inputenc, type1cm.
							PostScript (PSNFSS) font packages may also be
							loaded, depending on your font settings.

	# The following settings allow you to select the fonts in math mode.
	mathtext.fontset: dejavusans  # Should be 'dejavusans' (default),
								   'dejavuserif', 'cm' (Computer Modern), 'stix',
								   'stixsans' or 'custom' (unsupported, may go
								   away in the future)
	# "mathtext.fontset: custom" is defined by the mathtext.bf, .cal, .it, ...
	# settings which map a TeX font name to a fontconfig font pattern.  (These
	# settings are not used for other font sets.)
	mathtext.bf:  sans:bold
	mathtext.cal: cursive
	mathtext.it:  sans:italic
	mathtext.rm:  sans
	mathtext.sf:  sans
	mathtext.tt:  monospace
	mathtext.fallback: cm  # Select fallback font from ['cm' (Computer Modern), 'stix'
							'stixsans'] when a symbol can not be found in one of the
							custom math fonts. Select 'None' to not perform fallback
							and replace the missing character by a dummy symbol.
	mathtext.default: it  # The default font to use for math.
						   Can be any of the LaTeX font names, including
						   the special name "regular" for the same font
						   used in regular text.


	# ***************************************************************************
	# * AXES                                                                    *
	# ***************************************************************************
	# Following are default face and edge colors, default tick sizes,
	# default font sizes for tick labels, and so on.  See
	# https://matplotlib.org/api/axes_api.html#module-matplotlib.axes
	axes.facecolor:     white   # axes background color
	axes.edgecolor:     black   # axes edge color
	axes.linewidth:     0.8     # edge line width
	axes.grid:          False   # display grid or not
	axes.grid.axis:     both    # which axis the grid should apply to
	axes.grid.which:    major   # grid lines at {major, minor, both} ticks
	axes.titlelocation: center  # alignment of the title: {left, right, center}
	axes.titlesize:     large   # font size of the axes title
	axes.titleweight:   normal  # font weight of title
	axes.titlecolor:    auto    # color of the axes title, auto falls back to
								 text.color as default value
	axes.titley:        None    # position title (axes relative units).  None implies auto
	axes.titlepad:      6.0     # pad between axes and title in points
	axes.labelsize:     medium  # font size of the x and y labels
	axes.labelpad:      4.0     # space between label and axis
	axes.labelweight:   normal  # weight of the x and y labels
	axes.labelcolor:    black
	axes.axisbelow:     line    # draw axis gridlines and ticks:
									 - below patches (True)
									 - above patches but below lines ('line')
									 - above all (False)

	axes.formatter.limits: -5, 6  # use scientific notation if log10
								   of the axis range is smaller than the
								   first or larger than the second
	axes.formatter.use_locale: False  # When True, format tick labels
									   according to the user's locale.
									   For example, use ',' as a decimal
									   separator in the fr_FR locale.
	axes.formatter.use_mathtext: False  # When True, use mathtext for scientific
										 notation.
	axes.formatter.min_exponent: 0  # minimum exponent to format in scientific notation
	axes.formatter.useoffset: True  # If True, the tick label formatter
									 will default to labeling ticks relative
									 to an offset when the data range is
									 small compared to the minimum absolute
									 value of the data.
	axes.formatter.offset_threshold: 4  # When useoffset is True, the offset
										 will be used when it can remove
										 at least this number of significant
										 digits from tick labels.

	axes.spines.left:   True  # display axis spines
	axes.spines.bottom: True
	axes.spines.top:    True
	axes.spines.right:  True

	axes.unicode_minus: True  # use Unicode for the minus symbol rather than hyphen.  See
							   https://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes
	axes.prop_cycle: cycler('color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'])
					  color cycle for plot lines as list of string color specs:
					  single letter, long name, or web-style hex
					  As opposed to all other parameters in this file, the color
					  values must be enclosed in quotes for this parameter,
					  e.g. '1f77b4', instead of 1f77b4.
					  See also https://matplotlib.org/tutorials/intermediate/color_cycle.html
					  for more details on prop_cycle usage.
	axes.xmargin:   .05  # x margin.  See `axes.Axes.margins`
	axes.ymargin:   .05  # y margin.  See `axes.Axes.margins`
	axes.zmargin:   .05  # z margin.  See `axes.Axes.margins`
	axes.autolimit_mode: data  # If "data", use axes.xmargin and axes.ymargin as is.
								If "round_numbers", after application of margins, axis
								limits are further expanded to the nearest "round" number.
	polaraxes.grid: True  # display grid on polar axes
	axes3d.grid:    True  # display grid on 3D axes


	# ***************************************************************************
	# * AXIS                                                                    *
	# ***************************************************************************
	xaxis.labellocation: center  # alignment of the xaxis label: {left, right, center}
	yaxis.labellocation: center  # alignment of the yaxis label: {bottom, top, center}


	# ***************************************************************************
	# * DATES                                                                   *
	# ***************************************************************************
	# These control the default format strings used in AutoDateFormatter.
	# Any valid format datetime format string can be used (see the python
	# `datetime` for details).  For example, by using:
	#     - '%%x' will use the locale date representation
	#     - '%%X' will use the locale time representation
	#     - '%%c' will use the full locale datetime representation
	# These values map to the scales:
	#     {'year': 365, 'month': 30, 'day': 1, 'hour': 1/24, 'minute': 1 / (24 * 60)}

	date.autoformatter.year:        %Y
	date.autoformatter.month:       %Y-%m
	date.autoformatter.day:         %Y-%m-%d
	date.autoformatter.hour:        %m-%d %H
	date.autoformatter.minute:      %d %H:%M
	date.autoformatter.second:      %H:%M:%S
	date.autoformatter.microsecond: %M:%S.%f
	# The reference date for Matplotlib's internal date representation
	# See https://matplotlib.org/examples/ticks_and_spines/date_precision_and_epochs.py
	date.epoch: 1970-01-01T00:00:00
	# 'auto', 'concise':
	date.converter:                  auto
	# For auto converter whether to use interval_multiples:
	date.interval_multiples:         True

	# ***************************************************************************
	# * TICKS                                                                   *
	# ***************************************************************************
	# See https://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
	xtick.top:           False   # draw ticks on the top side
	xtick.bottom:        True    # draw ticks on the bottom side
	xtick.labeltop:      False   # draw label on the top
	xtick.labelbottom:   True    # draw label on the bottom
	xtick.major.size:    3.5     # major tick size in points
	xtick.minor.size:    2       # minor tick size in points
	xtick.major.width:   0.8     # major tick width in points
	xtick.minor.width:   0.6     # minor tick width in points
	xtick.major.pad:     3.5     # distance to major tick label in points
	xtick.minor.pad:     3.4     # distance to the minor tick label in points
	xtick.color:         black   # color of the ticks
	xtick.labelcolor:    inherit # color of the tick labels or inherit from xtick.color
	xtick.labelsize:     medium  # font size of the tick labels
	xtick.direction:     out     # direction: {in, out, inout}
	xtick.minor.visible: False   # visibility of minor ticks on x-axis
	xtick.major.top:     True    # draw x axis top major ticks
	xtick.major.bottom:  True    # draw x axis bottom major ticks
	xtick.minor.top:     True    # draw x axis top minor ticks
	xtick.minor.bottom:  True    # draw x axis bottom minor ticks
	xtick.alignment:     center  # alignment of xticks

	ytick.left:          True    # draw ticks on the left side
	ytick.right:         False   # draw ticks on the right side
	ytick.labelleft:     True    # draw tick labels on the left side
	ytick.labelright:    False   # draw tick labels on the right side
	ytick.major.size:    3.5     # major tick size in points
	ytick.minor.size:    2       # minor tick size in points
	ytick.major.width:   0.8     # major tick width in points
	ytick.minor.width:   0.6     # minor tick width in points
	ytick.major.pad:     3.5     # distance to major tick label in points
	ytick.minor.pad:     3.4     # distance to the minor tick label in points
	ytick.color:         black   # color of the ticks
	ytick.labelcolor:    inherit # color of the tick labels or inherit from ytick.color
	ytick.labelsize:     medium  # font size of the tick labels
	ytick.direction:     out     # direction: {in, out, inout}
	ytick.minor.visible: False   # visibility of minor ticks on y-axis
	ytick.major.left:    True    # draw y axis left major ticks
	ytick.major.right:   True    # draw y axis right major ticks
	ytick.minor.left:    True    # draw y axis left minor ticks
	ytick.minor.right:   True    # draw y axis right minor ticks
	ytick.alignment:     center_baseline  # alignment of yticks


	# ***************************************************************************
	# * GRIDS                                                                   *
	# ***************************************************************************
	grid.color:     b0b0b0  # grid color
	grid.linestyle: -       # solid
	grid.linewidth: 0.8     # in points
	grid.alpha:     1.0     # transparency, between 0.0 and 1.0


	# ***************************************************************************
	# * LEGEND                                                                  *
	# ***************************************************************************
	legend.loc:           best
	legend.frameon:       True     # if True, draw the legend on a background patch
	legend.framealpha:    0.8      # legend patch transparency
	legend.facecolor:     inherit  # inherit from axes.facecolor; or color spec
	legend.edgecolor:     0.8      # background patch boundary color
	legend.fancybox:      True     # if True, use a rounded box for the
									legend background, else a rectangle
	legend.shadow:        False    # if True, give background a shadow effect
	legend.numpoints:     1        # the number of marker points in the legend line
	legend.scatterpoints: 1        # number of scatter points
	legend.markerscale:   1.0      # the relative size of legend markers vs. original
	legend.fontsize:      medium
	legend.labelcolor:    None
	legend.title_fontsize: None    # None sets to the same as the default axes.

	# Dimensions as fraction of font size:
	legend.borderpad:     0.4  # border whitespace
	legend.labelspacing:  0.5  # the vertical space between the legend entries
	legend.handlelength:  2.0  # the length of the legend lines
	legend.handleheight:  0.7  # the height of the legend handle
	legend.handletextpad: 0.8  # the space between the legend line and legend text
	legend.borderaxespad: 0.5  # the border between the axes and legend edge
	legend.columnspacing: 2.0  # column separation


	# ***************************************************************************
	# * FIGURE                                                                  *
	# ***************************************************************************
	# See https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
	figure.titlesize:   large     # size of the figure title (``Figure.suptitle()``)
	figure.titleweight: normal    # weight of the figure title
	figure.figsize:     6.4, 4.8  # figure size in inches
	figure.dpi:         100       # figure dots per inch
	figure.facecolor:   white     # figure face color
	figure.edgecolor:   white     # figure edge color
	figure.frameon:     True      # enable figure frame
	figure.max_open_warning: 20   # The maximum number of figures to open through
								   the pyplot interface before emitting a warning.
								   If less than one this feature is disabled.
	figure.raise_window : True    # Raise the GUI window to front when show() is called.

	# The figure subplot parameters.  All dimensions are a fraction of the figure width and height.
	figure.subplot.left:   0.125  # the left side of the subplots of the figure
	figure.subplot.right:  0.9    # the right side of the subplots of the figure
	figure.subplot.bottom: 0.11   # the bottom of the subplots of the figure
	figure.subplot.top:    0.88   # the top of the subplots of the figure
	figure.subplot.wspace: 0.2    # the amount of width reserved for space between subplots,
								   expressed as a fraction of the average axis width
	figure.subplot.hspace: 0.2    # the amount of height reserved for space between subplots,
								   expressed as a fraction of the average axis height

	# Figure layout
	figure.autolayout: False  # When True, automatically adjust subplot
							   parameters to make the plot fit the figure
							   using `tight_layout`
	figure.constrained_layout.use: False  # When True, automatically make plot
										   elements fit on the figure. (Not
										   compatible with `autolayout`, above).
	figure.constrained_layout.h_pad:  0.04167  # Padding around axes objects. Float representing
	figure.constrained_layout.w_pad:  0.04167  # inches. Default is 3/72 inches (3 points)
	figure.constrained_layout.hspace: 0.02     # Space between subplot groups. Float representing
	figure.constrained_layout.wspace: 0.02     # a fraction of the subplot widths being separated.


	# ***************************************************************************
	# * IMAGES                                                                  *
	# ***************************************************************************
	image.aspect:          equal        # {equal, auto} or a number
	image.interpolation:   antialiased  # see help(imshow) for options
	image.cmap:            viridis      # A colormap name, gray etc...
	image.lut:             256          # the size of the colormap lookup table
	image.origin:          upper        # {lower, upper}
	image.resample:        True
	image.composite_image: True  # When True, all the images on a set of axes are
								  combined into a single composite image before
								  saving a figure as a vector graphics file,
								  such as a PDF.


	# ***************************************************************************
	# * CONTOUR PLOTS                                                           *
	# ***************************************************************************
	contour.negative_linestyle: dashed  # string or on-off ink sequence
	contour.corner_mask:        True    # {True, False, legacy}
	contour.linewidth:          None    # {float, None} Size of the contour line
										 widths. If set to None, it falls back to
										 `line.linewidth`.


	# ***************************************************************************
	# * ERRORBAR PLOTS                                                          *
	# ***************************************************************************
	errorbar.capsize: 0  # length of end cap on error bars in pixels


	# ***************************************************************************
	# * HISTOGRAM PLOTS                                                         *
	# ***************************************************************************
	hist.bins: 10  # The default number of histogram bins or 'auto'.


	# ***************************************************************************
	# * SCATTER PLOTS                                                           *
	# ***************************************************************************
	scatter.marker: o         # The default marker type for scatter plots.
	scatter.edgecolors: face  # The default edge colors for scatter plots.


	# ***************************************************************************
	# * AGG RENDERING                                                           *
	# ***************************************************************************
	# Warning: experimental, 2008/10/10
	agg.path.chunksize: 0  # 0 to disable; values in the range
							10000 to 100000 can improve speed slightly
							and prevent an Agg rendering failure
							when plotting very large data sets,
							especially if they are very gappy.
							It may cause minor artifacts, though.
							A value of 20000 is probably a good
							starting point.


	# ***************************************************************************
	# * PATHS                                                                   *
	# ***************************************************************************
	path.simplify: True  # When True, simplify paths by removing "invisible"
						  points to reduce file size and increase rendering
						  speed
	path.simplify_threshold: 0.111111111111  # The threshold of similarity below
											  which vertices will be removed in
											  the simplification process.
	path.snap: True  # When True, rectilinear axis-aligned paths will be snapped
					  to the nearest pixel when certain criteria are met.
					  When False, paths will never be snapped.
	path.sketch: None  # May be None, or a 3-tuple of the form:
						(scale, length, randomness).
							- *scale* is the amplitude of the wiggle
								perpendicular to the line (in pixels).
							- *length* is the length of the wiggle along the
								line (in pixels).
							- *randomness* is the factor by which the length is
								randomly scaled.
	path.effects:


	# ***************************************************************************
	# * SAVING FIGURES                                                          *
	# ***************************************************************************
	# The default savefig parameters can be different from the display parameters
	# e.g., you may want a higher resolution, or to make the figure
	# background white
	savefig.dpi:       figure      # figure dots per inch or 'figure'
	savefig.facecolor: auto        # figure face color when saving
	savefig.edgecolor: auto        # figure edge color when saving
	savefig.format:    png         # {png, ps, pdf, svg}
	savefig.bbox:      standard    # {tight, standard}
									'tight' is incompatible with pipe-based animation
									backends (e.g. 'ffmpeg') but will work with those
									based on temporary files (e.g. 'ffmpeg_file')
	savefig.pad_inches:   0.1      # Padding to be used when bbox is set to 'tight'
	savefig.directory:    ~        # default directory in savefig dialog box,
									leave empty to always use current working directory
	savefig.transparent: False     # setting that controls whether figures are saved with a
									transparent background by default
	savefig.orientation: portrait  # Orientation of saved figure

	## tk backend params
	tk.window_focus:   False  # Maintain shell focus for TkAgg

	## ps backend params
	ps.papersize:      letter  # {auto, letter, legal, ledger, A0-A10, B0-B10}
	ps.useafm:         False   # use of AFM fonts, results in small files
	ps.usedistiller:   False   # {ghostscript, xpdf, None}
								Experimental: may produce smaller files.
								xpdf intended for production of publication quality files,
								but requires ghostscript, xpdf and ps2eps
	ps.distiller.res:  6000    # dpi
	ps.fonttype:       3       # Output Type 3 (Type3) or Type 42 (TrueType)

	## PDF backend params
	pdf.compression:    6  # integer from 0 to 9
							0 disables compression (good for debugging)
	pdf.fonttype:       3  # Output Type 3 (Type3) or Type 42 (TrueType)
	pdf.use14corefonts: False
	pdf.inheritcolor:   False

	## SVG backend params
	svg.image_inline: True  # Write raster image data directly into the SVG file
	svg.fonttype: path      # How to handle SVG fonts:
								 path: Embed characters as paths -- supported
									   by most SVG renderers
								 None: Assume fonts are installed on the
									   machine where the SVG will be viewed.
	svg.hashsalt: None      # If not None, use this string as hash salt instead of uuid4

	## pgf parameter
	# See https://matplotlib.org/tutorials/text/pgf.html for more information.
	pgf.rcfonts: True
	pgf.preamble:  # See text.latex.preamble for documentation
	pgf.texsystem: xelatex

	## docstring params
	docstring.hardcopy: False  # set this when you want to generate hardcopy docstring


	# ***************************************************************************
	# * INTERACTIVE KEYMAPS                                                     *
	# ***************************************************************************
	# Event keys to interact with figures/plots via keyboard.
	# See https://matplotlib.org/users/navigation_toolbar.html for more details on
	# interactive navigation.  Customize these settings according to your needs.
	# Leave the field(s) empty if you don't need a key-map. (i.e., fullscreen : '')
	keymap.fullscreen: f, ctrl+f   # toggling
	keymap.home: h, r, home        # home or reset mnemonic
	keymap.back: left, c, backspace, MouseButton.BACK  # forward / backward keys
	keymap.forward: right, v, MouseButton.FORWARD      # for quick navigation
	keymap.pan: p                  # pan mnemonic
	keymap.zoom: o                 # zoom mnemonic
	keymap.save: s, ctrl+s         # saving current figure
	keymap.help: f1                # display help about active tools
	keymap.quit: ctrl+w, cmd+w, q  # close the current figure
	keymap.quit_all:               # close all figures
	keymap.grid: g                 # switching on/off major grids in current axes
	keymap.grid_minor: G           # switching on/off minor grids in current axes
	keymap.yscale: l               # toggle scaling of y-axes ('log'/'linear')
	keymap.xscale: k, L            # toggle scaling of x-axes ('log'/'linear')
	keymap.copy: ctrl+c, cmd+c     # copy figure to clipboard


	# ***************************************************************************
	# * ANIMATION                                                               *
	# ***************************************************************************
	animation.html: none  # How to display the animation as HTML in
						   the IPython notebook:
							   - 'html5' uses HTML5 video tag
							   - 'jshtml' creates a JavaScript animation
	animation.writer:  ffmpeg        # MovieWriter 'backend' to use
	animation.codec:   h264          # Codec to use for writing movie
	animation.bitrate: -1            # Controls size/quality trade-off for movie.
									  -1 implies let utility auto-determine
	animation.frame_format: png      # Controls frame format used by temp files
	animation.ffmpeg_path:  ffmpeg   # Path to ffmpeg binary. Without full path
									  $PATH is searched
	animation.ffmpeg_args:           # Additional arguments to pass to ffmpeg
	animation.convert_path: convert  # Path to ImageMagick's convert binary.
									  On Windows use the full path since convert
									  is also the name of a system tool.
	animation.convert_args:          # Additional arguments to pass to convert
	animation.embed_limit:  20.0     # Limit, in MB, of size of base64 encoded
									  animation in HTML (i.e. IPython notebook)
	```
=== "Default settings"
	```python
	RcParams({
	'_internal.classic_mode': False,
  	'agg.path.chunksize': 0,
  	'animation.avconv_args': [],
  	'animation.avconv_path': 'avconv',
  	'animation.bitrate': -1,
  	'animation.codec': 'h264',
  	'animation.convert_args': [],
 	'animation.convert_path': 'convert',
	'animation.embed_limit': 20.0,
	'animation.ffmpeg_args': [],
	'animation.ffmpeg_path': 'ffmpeg',
	'animation.frame_format': 'png',
	'animation.html': 'none',
	'animation.html_args': [],
	'animation.writer': 'ffmpeg',
	'axes.autolimit_mode': 'data',
	'axes.axisbelow': 'line',
	'axes.edgecolor': 'black',
	'axes.facecolor': 'white',
	'axes.formatter.limits': [-5, 6],
	'axes.formatter.min_exponent': 0,
	'axes.formatter.offset_threshold': 4,
	'axes.formatter.use_locale': False,
	'axes.formatter.use_mathtext': False,
	'axes.formatter.useoffset': True,
	'axes.grid': False,
	'axes.grid.axis': 'both',
	'axes.grid.which': 'major',
	'axes.labelcolor': 'black',
	'axes.labelpad': 4.0,
	'axes.labelsize': 'medium',
	'axes.labelweight': 'normal',
	'axes.linewidth': 0.8,
	'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
	'axes.spines.bottom': True,
	'axes.spines.left': True,
	'axes.spines.right': True,
	'axes.spines.top': True,
	'axes.titlecolor': 'auto',
	'axes.titlelocation': 'center',
	'axes.titlepad': 6.0,
	'axes.titlesize': 'large',
	'axes.titleweight': 'normal',
	'axes.unicode_minus': True,
	'axes.xmargin': 0.05,
	'axes.ymargin': 0.05,
	'axes3d.grid': True,
	'backend': 'TkAgg',
	'backend_fallback': True,
	'boxplot.bootstrap': None,
	'boxplot.boxprops.color': 'black',
	'boxplot.boxprops.linestyle': '-',
	'boxplot.boxprops.linewidth': 1.0,
	'boxplot.capprops.color': 'black',
	'boxplot.capprops.linestyle': '-',
	'boxplot.capprops.linewidth': 1.0,
	'boxplot.flierprops.color': 'black',
	'boxplot.flierprops.linestyle': 'none',
	'boxplot.flierprops.linewidth': 1.0,
	'boxplot.flierprops.marker': 'o',
	'boxplot.flierprops.markeredgecolor': 'black',
	'boxplot.flierprops.markeredgewidth': 1.0,
	'boxplot.flierprops.markerfacecolor': 'none',
	'boxplot.flierprops.markersize': 6.0,
	'boxplot.meanline': False,
	'boxplot.meanprops.color': 'C2',
	'boxplot.meanprops.linestyle': '--',
	'boxplot.meanprops.linewidth': 1.0,
	'boxplot.meanprops.marker': '^',
	'boxplot.meanprops.markeredgecolor': 'C2',
	'boxplot.meanprops.markerfacecolor': 'C2',
	'boxplot.meanprops.markersize': 6.0,
	'boxplot.medianprops.color': 'C1',
	'boxplot.medianprops.linestyle': '-',
	'boxplot.medianprops.linewidth': 1.0,
	'boxplot.notch': False,
	'boxplot.patchartist': False,
	'boxplot.showbox': True,
	'boxplot.showcaps': True,
	'boxplot.showfliers': True,
	'boxplot.showmeans': False,
	'boxplot.vertical': True,
	'boxplot.whiskerprops.color': 'black',
	'boxplot.whiskerprops.linestyle': '-',
	'boxplot.whiskerprops.linewidth': 1.0,
	'boxplot.whiskers': 1.5,
	'contour.corner_mask': True,
	'contour.negative_linestyle': 'dashed',
	'datapath': 'd:\\anaconda\\envs\\py37\\lib\\site-packages\\matplotlib\\mpl-data',
	'date.autoformatter.day': '%Y-%m-%d',
	'date.autoformatter.hour': '%m-%d %H',
	'date.autoformatter.microsecond': '%M:%S.%f',
	'date.autoformatter.minute': '%d %H:%M',
	'date.autoformatter.month': '%Y-%m',
	'date.autoformatter.second': '%H:%M:%S',
	'date.autoformatter.year': '%Y',
	'docstring.hardcopy': False,
	'errorbar.capsize': 0.0,
	'figure.autolayout': False,
	'figure.constrained_layout.h_pad': 0.04167,
	'figure.constrained_layout.hspace': 0.02,
	'figure.constrained_layout.use': False,
	'figure.constrained_layout.w_pad': 0.04167,
	'figure.constrained_layout.wspace': 0.02,
	'figure.dpi': 100.0,
	'figure.edgecolor': 'white',
	'figure.facecolor': 'white',
	'figure.figsize': [6.4, 4.8],
	'figure.frameon': True,
	'figure.max_open_warning': 20,
	'figure.subplot.bottom': 0.11,
	'figure.subplot.hspace': 0.2,
	'figure.subplot.left': 0.125,
	'figure.subplot.right': 0.9,
	'figure.subplot.top': 0.88,
	'figure.subplot.wspace': 0.2,
	'figure.titlesize': 'large',
	'figure.titleweight': 'normal',
	'font.cursive': ['Apple Chancery',
				   'Textile',
				   'Zapf Chancery',
				   'Sand',
				   'Script MT',
				   'Felipa',
				   'cursive'],
	'font.family': ['sans-serif'],
	'font.fantasy': ['Comic Neue',
				   'Comic Sans MS',
				   'Chicago',
				   'Charcoal',
				   'Impact',
				   'Western',
				   'Humor Sans',
				   'xkcd',
				   'fantasy'],
	'font.monospace': ['DejaVu Sans Mono',
					 'Bitstream Vera Sans Mono',
					 'Computer Modern Typewriter',
					 'Andale Mono',
					 'Nimbus Mono L',
					 'Courier New',
					 'Courier',
					 'Fixed',
					 'Terminal',
					 'monospace'],
	'font.sans-serif': ['DejaVu Sans',
					  'Bitstream Vera Sans',
					  'Computer Modern Sans Serif',
					  'Lucida Grande',
					  'Verdana',
					  'Geneva',
					  'Lucid',
					  'Arial',
					  'Helvetica',
					  'Avant Garde',
					  'sans-serif'],
	'font.serif': ['DejaVu Serif',
				 'Bitstream Vera Serif',
				 'Computer Modern Roman',
				 'New Century Schoolbook',
				 'Century Schoolbook L',
				 'Utopia',
				 'ITC Bookman',
				 'Bookman',
				 'Nimbus Roman No9 L',
				 'Times New Roman',
				 'Times',
				 'Palatino',
				 'Charter',
				 'serif'],
	'font.size': 10.0,
	'font.stretch': 'normal',
	'font.style': 'normal',
	'font.variant': 'normal',
	'font.weight': 'normal',
	'grid.alpha': 1.0,
	'grid.color': '#b0b0b0',
	'grid.linestyle': '-',
	'grid.linewidth': 0.8,
	'hatch.color': 'black',
	'hatch.linewidth': 1.0,
	'hist.bins': 10,
	'image.aspect': 'equal',
	'image.cmap': 'viridis',
	'image.composite_image': True,
	'image.interpolation': 'antialiased',
	'image.lut': 256,
	'image.origin': 'upper',
	'image.resample': True,
	'interactive': True,
	'keymap.all_axes': ['a'],
	'keymap.back': ['left', 'c', 'backspace', 'MouseButton.BACK'],
	'keymap.copy': ['ctrl+c', 'cmd+c'],
	'keymap.forward': ['right', 'v', 'MouseButton.FORWARD'],
	'keymap.fullscreen': ['f', 'ctrl+f'],
	'keymap.grid': ['g'],
	'keymap.grid_minor': ['G'],
	'keymap.help': ['f1'],
	'keymap.home': ['h', 'r', 'home'],
	'keymap.pan': ['p'],
	'keymap.quit': ['ctrl+w', 'cmd+w', 'q'],
	'keymap.quit_all': ['W', 'cmd+W', 'Q'],
	'keymap.save': ['s', 'ctrl+s'],
	'keymap.xscale': ['k', 'L'],
	'keymap.yscale': ['l'],
	'keymap.zoom': ['o'],
	'legend.borderaxespad': 0.5,
	'legend.borderpad': 0.4,
	'legend.columnspacing': 2.0,
	'legend.edgecolor': '0.8',
	'legend.facecolor': 'inherit',
	'legend.fancybox': True,
	'legend.fontsize': 'medium',
	'legend.framealpha': 0.8,
	'legend.frameon': True,
	'legend.handleheight': 0.7,
	'legend.handlelength': 2.0,
	'legend.handletextpad': 0.8,
	'legend.labelspacing': 0.5,
	'legend.loc': 'best',
	'legend.markerscale': 1.0,
	'legend.numpoints': 1,
	'legend.scatterpoints': 1,
	'legend.shadow': False,
	'legend.title_fontsize': None,
	'lines.antialiased': True,
	'lines.color': 'C0',
	'lines.dash_capstyle': 'butt',
	'lines.dash_joinstyle': 'round',
	'lines.dashdot_pattern': [6.4, 1.6, 1.0, 1.6],
	'lines.dashed_pattern': [3.7, 1.6],
	'lines.dotted_pattern': [1.0, 1.65],
	'lines.linestyle': '-',
	'lines.linewidth': 1.5,
	'lines.marker': 'None',
	'lines.markeredgecolor': 'auto',
	'lines.markeredgewidth': 1.0,
	'lines.markerfacecolor': 'auto',
	'lines.markersize': 6.0,
	'lines.scale_dashes': True,
	'lines.solid_capstyle': 'projecting',
	'lines.solid_joinstyle': 'round',
	'markers.fillstyle': 'full',
	'mathtext.bf': 'sans:bold',
	'mathtext.cal': 'cursive',
	'mathtext.default': 'it',
	'mathtext.fallback_to_cm': True,
	'mathtext.fontset': 'dejavusans',
	'mathtext.it': 'sans:italic',
	'mathtext.rm': 'sans',
	'mathtext.sf': 'sans',
	'mathtext.tt': 'monospace',
	'mpl_toolkits.legacy_colorbar': True,
	'patch.antialiased': True,
	'patch.edgecolor': 'black',
	'patch.facecolor': 'C0',
	'patch.force_edgecolor': False,
	'patch.linewidth': 1.0,
	'path.effects': [],
	'path.simplify': True,
	'path.simplify_threshold': 0.1111111111111111,
	'path.sketch': None,
	'path.snap': True,
	'pdf.compression': 6,
	'pdf.fonttype': 3,
	'pdf.inheritcolor': False,
	'pdf.use14corefonts': False,
	'pgf.preamble': '',
	'pgf.rcfonts': True,
	'pgf.texsystem': 'xelatex',
	'polaraxes.grid': True,
	'ps.distiller.res': 6000,
	'ps.fonttype': 3,
	'ps.papersize': 'letter',
	'ps.useafm': False,
	'ps.usedistiller': None,
	'savefig.bbox': None,
	'savefig.directory': '~',
	'savefig.dpi': 'figure',
	'savefig.edgecolor': 'white',
	'savefig.facecolor': 'white',
	'savefig.format': 'png',
	'savefig.frameon': True,
	'savefig.jpeg_quality': 95,
	'savefig.orientation': 'portrait',
	'savefig.pad_inches': 0.1,
	'savefig.transparent': False,
	'scatter.edgecolors': 'face',
	'scatter.marker': 'o',
	'svg.fonttype': 'path',
	'svg.hashsalt': None,
	'svg.image_inline': True,
	'text.antialiased': True,
	'text.color': 'black',
	'text.hinting': 'auto',
	'text.hinting_factor': 8,
	'text.kerning_factor': 0,
	'text.latex.preamble': '',
	'text.latex.preview': False,
	'text.latex.unicode': True,
	'text.usetex': False,
	'timezone': 'UTC',
	'tk.window_focus': False,
	'toolbar': 'toolbar2',
	'verbose.fileo': 'sys.stdout',
	'verbose.level': 'silent',
	'webagg.address': '127.0.0.1',
	'webagg.open_in_browser': True,
	'webagg.port': 8988,
	'webagg.port_retries': 50,
	'xtick.alignment': 'center',
	'xtick.bottom': True,
	'xtick.color': 'black',
	'xtick.direction': 'out',
	'xtick.labelbottom': True,
	'xtick.labelsize': 'medium',
	'xtick.labeltop': False,
	'xtick.major.bottom': True,
	'xtick.major.pad': 3.5,
	'xtick.major.size': 3.5,
	'xtick.major.top': True,
	'xtick.major.width': 0.8,
	'xtick.minor.bottom': True,
	'xtick.minor.pad': 3.4,
	'xtick.minor.size': 2.0,
	'xtick.minor.top': True,
	'xtick.minor.visible': False,
	'xtick.minor.width': 0.6,
	'xtick.top': False,
	'ytick.alignment': 'center_baseline',
	'ytick.color': 'black',
	'ytick.direction': 'out',
	'ytick.labelleft': True,
	'ytick.labelright': False,
	'ytick.labelsize': 'medium',
	'ytick.left': True,
	'ytick.major.left': True,
	'ytick.major.pad': 3.5,
	'ytick.major.right': True,
	'ytick.major.size': 3.5,
	'ytick.major.width': 0.8,
	'ytick.minor.left': True,
	'ytick.minor.pad': 3.4,
	'ytick.minor.right': True,
	'ytick.minor.size': 2.0,
	'ytick.minor.visible': False,
	'ytick.minor.width': 0.6,
	'ytick.right': False})
	```


