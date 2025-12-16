import numpy as np

# =============================================================================
#   Load Data
# =============================================================================
year0,year1 = 1993,2021
grid_size_x, grid_size_y = 5, 5
lat  = np.arange(-90,90,grid_size_y)
lon  = np.arange(-180,180+grid_size_x,grid_size_x)
lonmesh, latmesh   = np.meshgrid(lon,lat)
lat_size, lon_size = lonmesh.shape

Trend_WOD    = np.load('Trend_of_IMHW_WOD.npy')
Trend_GLORYS = np.load('Trend_of_IMHW_GLORYS.npy')
Trend_CESM   = np.load('Trend_of_IMHW_CESM.npy')
P_WOD        = np.load('p_of_Trend_of_IMHW_WOD.npy')
P_GLORYS     = np.load('p_of_Trend_of_IMHW_GLORYS.npy')
P_CESM       = np.load('p_of_Trend_of_IMHW_CESM.npy')

sigmask = np.isnan(P_WOD) | (P_WOD>.05)
Trend_WOD_sig = Trend_WOD.copy()
Trend_WOD_sig[sigmask] = np.nan

sigmask = np.isnan(P_GLORYS) | (P_GLORYS>.05)
Trend_GLORYS_sig = Trend_GLORYS.copy()
Trend_GLORYS_sig[sigmask] = np.nan

sigmask = np.isnan(P_CESM) | (P_CESM>.05)
Trend_CESM_sig = Trend_CESM.copy()
Trend_CESM_sig[sigmask] = np.nan

depth_series = [0, 100, 200]#unit: m
Depth_WOD    = np.load('depth_IAP.npy')
Depth_GLORYS = np.load('depth_GLORYS.npy')
Depth_CESM   = np.load('depth_CESM.npy')
depth_indices_WOD    = [np.argmin(abs(Depth_WOD - d)) for d in depth_series]
depth_indices_GLORYS = [np.argmin(abs(Depth_GLORYS - d)) for d in depth_series]
depth_indices_CESM   = [np.argmin(abs(Depth_CESM - d)) for d in depth_series]


#%%
# =============================================================================
#   Plot Settings
# =============================================================================
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as feature
from matplotlib.colors import LinearSegmentedColormap

# colormap modification
cmax = 40
cmin = 10
cmid = 2
n_ratio = 7
n_positive = int((cmax-cmid)*n_ratio)
n_negative = int((cmin-cmid)*n_ratio)
n_zero     = int(2*cmid*n_ratio)
positive_colors = plt.cm.YlOrRd(np.linspace(0,1,n_positive))
negative_colors = plt.cm.BuPu_r(np.linspace(.8,1,n_negative))
zero_colors = [(0.0, '#F7FCFD'), (0.5, '#FFFFFF'), (1.0, '#FFFFCC')]
zero_cmap = LinearSegmentedColormap.from_list('zeros', zero_colors, N=n_zero)
zero_colors = zero_cmap(np.linspace(0,1,n_zero))
all_colors = np.r_[negative_colors, zero_colors, positive_colors]

cmap_IMHW = LinearSegmentedColormap.from_list('IMHW', all_colors)
cmap_IMHW.set_under(plt.cm.BuPu_r(.8))
cmap_IMHW.set_over('darkred')

levels = np.linspace(-cmin,cmax,50)
ticks  = np.arange(-cmin,cmax+10,10)

# Cartopy settings
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
proj0 = ccrs.PlateCarree()
proj  = ccrs.PlateCarree(central_longitude=180)

# Font settings
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 7}
font_cited = {'family': 'Arial',
        'weight': 'semibold',
        'size': 12}
font_label = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}
font_xlabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 9}
font_ylabel = {'family': 'Arial',
        'weight': 'semibold',
        'size': 12}
font_ticklabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 7}
font_xticklabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 6}
font_yticklabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 6}
font_unit = {'family': 'Arial',
        'weight': 'normal',
        'size': 6}
font_text = {'family': 'Arial',
        'weight': 'semibold',
        'size': 9,
        'color':(1,1,1)}
font_title = {'family': 'Arial',
        'weight': 'semibold',
        'size': 12}
font_legend = {'family': 'Arial',
        'weight': 'normal',
        'size': 9.5}
cites = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x']
ticklabelsize = 6
cbarticklabelsize = 12
ylabelpad = 5
titlepad = 5

x_cite, y_cite = 0, 1.02
c_lft_margin, c_rgt_margin = .14, .13
c_y0 = .038
c_wid= .035

color_WOD = (1,0,0)
color_GLY = (.1,.1,.1)
colors = [(1,0,0),(0,0,1),(0,.9,0),(.5,.5,.5)]
lw = 1.5
lw0 = 1
s = .2

figsize=(11,5)
nrow, ncol = 3, 3

#%%
# =============================================================================
#   Draw Fig. 1
# =============================================================================

fig = plt.figure(figsize=figsize, dpi=600)
gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1,1,1], hspace=.1, wspace=.13)
ax = []
# ========
for i in range(nrow):
    row_axes = []  # 临时存储当前行的所有子图
    for j in range(ncol):
        axi = fig.add_subplot(gs[i,j], projection=proj)  # 这里使用 proj
        axi.coastlines(lw=.7,zorder=3)
        axi.add_feature(feature.LAND,color=(.2,.2,.2),zorder=2)
        axi.set_extent([-180,180,-70,70], crs=proj)
        axi.set_xticks([-180,-120,-60,60,120,180],crs=proj0)
        axi.set_yticks(np.arange(-60, 80, 20),crs=proj0)
        axi.xaxis.set_major_formatter(lon_formatter)
        axi.yaxis.set_major_formatter(lat_formatter)
        axi.tick_params(which = 'both', direction = 'in', labelsize=ticklabelsize, length=2)
        axi.set_ylabel(f'{depth_series[i]}', fontdict=font_ylabel, labelpad=ylabelpad) if j==0 else 1
        axi.text(x_cite, y_cite, cites[nrow*j+i],ha='center',va='bottom',transform=axi.transAxes,fontdict=font_cited)
        row_axes.append(axi)
    ax.append(row_axes)

            
for d, D_idx in enumerate(depth_indices_WOD):
    D_idx = depth_indices_WOD[d]
    d_idx = depth_indices_GLORYS[d]
    C_idx = depth_indices_CESM[d]
    
    c=ax[d][0].pcolormesh(lon, lat, Trend_WOD[D_idx], cmap=cmap_IMHW, vmin=-cmin, vmax=cmax, transform=proj0)
    ax[d][1].pcolormesh(lon, lat, Trend_GLORYS[d_idx], cmap=cmap_IMHW, vmin=-cmin, vmax=cmax, transform=proj0) 
    ax[d][2].pcolormesh(lon,lat, Trend_CESM[C_idx],    cmap=cmap_IMHW, vmin=-cmin, vmax=cmax, transform=proj0) 
    
    Idx  = ~np.isnan(Trend_WOD_sig[:,:,D_idx])
    idx  = ~np.isnan(Trend_GLORYS_sig[d_idx])
    idxC = ~np.isnan(Trend_CESM_sig[C_idx])
    ax[d][0].scatter( lonmesh[Idx], latmesh[Idx], s=s, marker='.',alpha=.6, color=(.4,.4,.4),transform=proj0)
    ax[d][1].scatter( lonmesh[idx], latmesh[idx], s=s, marker='.',alpha=.4, color=(.4,.4,.4),transform=proj0)
    ax[d][2].scatter( lonmesh[idxC],latmesh[idxC],s=s, marker='.',alpha=.4, color=(.4,.4,.4),transform=proj0)


ax[0][0].set_title('in situ temperature profiles', fontdict=font_title, pad=titlepad)
ax[0][1].set_title('GLORYS', fontdict=font_title, pad=titlepad)
ax[0][2].set_title('CESM', fontdict=font_title, pad=titlepad)


ax_cb = fig.add_axes([c_lft_margin, c_y0,  1-c_rgt_margin-c_lft_margin,  c_wid])
cb = fig.colorbar(mappable=c, cax=ax_cb, orientation='horizontal', extend='both', shrink=1, pad=0.05)
cb.formatter.set_powerlimits((-2,2))
cb.ax.tick_params(axis='both', labelsize=cbarticklabelsize,direction='in', length=4, width=1)
cb.outline.set_linewidth(0.6)
cb.minorticks_off()
cb.set_ticks(ticks=ticks, labels=[f'{i}%' for i in ticks])

plt.savefig('Fig1.png',format='png',dpi=600,bbox_inches='tight')
plt.show()