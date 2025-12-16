import numpy as np

# =============================================================================
#   Load Data
# =============================================================================
year0,year1 = 1993,2021
grid_size_x, grid_size_y = 5, 5
lat  = np.arange(-90,90,grid_size_y)
lon  = np.arange(-180,180+grid_size_x,grid_size_x)
lonmesh, latmesh = np.meshgrid(lon,lat)

Trend_calibrated = np.load('Trend_of_IMHW_WOD_calibrated.npy')
Trend_fixed      = np.load('Trend_of_IMHW_WOD_fixed.npy')
P_calibrated     = np.load('p_of_Trend_of_IMHW_WOD_calibrated.npy')
P_fixed          = np.load('p_of_Trend_of_IMHW_WOD_fixed.npy')

sigmask = np.isnan(P_calibrated) | (P_calibrated>.05)
Trend_calibrated_sig = Trend_calibrated.copy()
Trend_calibrated_sig[sigmask] = np.nan

sigmask = np.isnan(P_fixed) | (P_fixed>.05)
Trend_fixed_sig = Trend_fixed.copy()
Trend_fixed_sig[sigmask] = np.nan

Depth_WOD    = np.load('depth_IAP.npy')
depth_series = [0, 100, 200]#unit: m
depth_indices_WOD = [np.argmin(abs(Depth_WOD - d)) for d in depth_series]

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

lw = 1.5
lw0 = 1
s = .2

x_cite, y_cite = 0, 1.02
c_lft_margin, c_rgt_margin = .14, .13
c_y0 = .038
c_wid= .035

nrow = len(depth_series)
ncol = 2
figsize=(7,5)

#%%
# =============================================================================
#   Draw Fig. 3
# =============================================================================
fig = plt.figure(figsize=figsize, dpi=600)
gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1,1], hspace=.1, wspace=.13)
ax = []
# ========
for i in range(nrow):
    row_axes = []
    for j in range(ncol):
        axi = fig.add_subplot(gs[i,j], projection=proj)
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
    
    c=ax[d][0].pcolormesh(lon, lat, Trend_calibrated[D_idx],  cmap=cmap_IMHW, vmin=-cmin, vmax=cmax, transform=proj0)
    ax[d][1].pcolormesh(lon, lat, Trend_fixed[D_idx], cmap=cmap_IMHW, vmin=-cmin, vmax=cmax, transform=proj0) 
    
    Idx  = ~np.isnan(Trend_calibrated_sig[:,:,d])
    idx  = ~np.isnan(Trend_fixed_sig[:,:,D_idx])
    ax[d][0].scatter( lonmesh[Idx], latmesh[Idx], s=s, marker='.',alpha=.4, color=(.4,.4,.4),transform=proj0)
    ax[d][1].scatter( lonmesh[idx], latmesh[idx], s=s, marker='.',alpha=.4, color=(.4,.4,.4),transform=proj0)


ax[0][0].set_title('bias calibrated', fontdict=font_title, pad=titlepad)
ax[0][1].set_title('fixed threshold', fontdict=font_title, pad=titlepad)


ax_cb = fig.add_axes([c_lft_margin, c_y0,  1-c_rgt_margin-c_lft_margin,  c_wid])
cb = fig.colorbar(mappable=c, cax=ax_cb, orientation='horizontal', extend='both', shrink=1, pad=0.05)
cb.formatter.set_powerlimits((-2,2))
cb.ax.tick_params(axis='both', labelsize=cbarticklabelsize,direction='in', length=4, width=1)
cb.outline.set_linewidth(0.6)
cb.minorticks_off()
cb.set_ticks(ticks=ticks, labels=[f'{i}%' for i in ticks])

plt.savefig('Fig3.png',format='png',dpi=600,bbox_inches='tight')
plt.show()