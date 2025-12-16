import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as st
from scipy.ndimage import uniform_filter

# =============================================================
#  Load Data
# =============================================================

Depth = np.load('depth_GLORYS.npy')
depth = 100
depth_index = np.argmin(abs(Depth-depth))
year0, year1 = 1993, 2021
Nyr = year1-year0+1
t_size = Nyr*12
smooth_wnd = 5

N_prof = np.load('N_prof.npy')
I_MHW  = np.load('I_MHW.npy')
Pop    = np.load('Temperature_anomaly_population_GLORYS_6050-6070_19932021.npy')

# true values of 95% quantile and mean above it
true_quantile = np.percentile(Pop, 95)
true_mean     = np.mean( Pop[Pop>=true_quantile] )
print(f" True 95% quantile = {true_quantile:.4f}")
print(f" True intensity (Mean above 95% quantile) = {true_mean:.4f}")
ks_stat, ks_p = st.kstest(Pop, 'norm', args=(np.mean(Pop), np.std(Pop)))
print(" Pop fits normal distribution") if ks_p<0.05 else print(" Pop does not fit normal distribution")

# sample sizes and repetitions
N_Samples = np.arange(2, 251, 2)
n_repeats = 1000

#%%
# =============================================================
#  Sampling Experiment
# =============================================================

# initialize result arrays
thrs = np.full((len(N_Samples), n_repeats),np.nan)
mean = np.full((len(N_Samples), n_repeats),np.nan)

for i, N_Sample in enumerate(tqdm(N_Samples)):
    for j in range(n_repeats):
        sample = np.random.choice(Pop, size=N_Sample, replace=True)
        thrs[i,j] = np.percentile(sample, 95)
        mean[i,j] = np.nanmean( sample[sample>=thrs[i,j]] )

thrs_mean = np.mean(thrs, axis=1)
mean_mean = np.mean(mean, axis=1)



#%%
# =============================================================
#  Figure Plotting
# =============================================================
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 7}
font_cited = {'family': 'Arial',
        'weight': 'semibold',
        'size': 16}
font_xlabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 12}
font_ylabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 11}
font_label = {'family': 'Arial',
        'weight': 'roman',
        'size': 7}
font_xticklabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}
font_yticklabel = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}
font_unit = {'family': 'Arial',
        'weight': 'normal',
        'size': 6}
font_text = {'family': 'Arial',
        'weight': 'normal',
        'size': 7}
font_thres = {'family': 'Arial',
        'weight': 'semibold','color':(1,0,0),
        'size': 13}
font_title = {'family': 'Arial',
        'weight': 'normal',
        'size': 13}
font_legend = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}
xticklabelsize = 10
yticklabelsize = 10
cites = ['a','b','c','d','e','f','g']
y_cite = 1.02
#
alpha_PDF = .5
color_PDF = (.5,.5,.5)
color_num = (0,0,0)
color_Ht = (255/255,152/255,0)
color_Hm = (211/255,47/255,47/255)
color_HT = (.9,.2,.2)
#
lw0 = 3
lw1 = 3
lw1M = 1
lw1_h = 2.5
lw2 = 1
lw3 = 1.3
lw3_t = 3
lw3_s = 1

# =========================================================

fig = plt.figure(figsize=(11,11), dpi=600)

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.2, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],  width_ratios=[1, 1], wspace=0.2)
ax2 = fig.add_subplot(row2_gs[0])
ax3 = fig.add_subplot(row2_gs[1])


ax1.spines['top'].set_visible(False) #设置坐标轴,下同
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

#  CITES  
ax0.text(0,y_cite, cites[0], fontdict=font_cited, ha='center', va='bottom', transform=ax0.transAxes)
ax1.text(0,y_cite, cites[1], fontdict=font_cited, ha='center', va='bottom', transform=ax1.transAxes)
ax2.text(0,y_cite, cites[2], fontdict=font_cited, ha='center', va='bottom', transform=ax2.transAxes)
ax3.text(-.02,y_cite, cites[3], fontdict=font_cited, ha='center', va='bottom', transform=ax3.transAxes)



# =======  ax0 PDF  =========
bins = np.linspace(-1.5,1.5,400)
ax0_xmax = 1

ax0.hist(Pop, bins=bins, density=True, alpha=alpha_PDF, color=color_PDF)
ax0.axvline(0,ls='--',lw=1,color=(0,0,0))

ax0_xticks = np.arange(-1,1+.5,.5)
ax0.set_xticks(ax0_xticks, ['%.1f'%i for i in ax0_xticks], fontdict=font_xticklabel)
ax0.set_xlabel('Temperature anomaly (K)', fontdict=font_xlabel)
ax0.set_title('PDF of temperature anomaly', fontdict=font_title)
ax0.set_xlim(-ax0_xmax,ax0_xmax)

# =======  ax1 TEST  =========
ax1_ymin, ax1_ymax = .2, .8
ax1.plot(N_Samples, thrs_mean, color=color_Ht, linewidth=lw1, label='EHTA threshold estimate')
ax1.plot(N_Samples, mean_mean, color=color_Hm, linewidth=lw1, label='EHTA intensity estimate')

ax1.axhline(y=true_quantile, color=color_Ht, linestyle=':', linewidth=lw1_h, label=f'EHTA threshold ({true_quantile:.2f})')
ax1.axhline(y=true_mean,     color=color_Hm, linestyle=':', linewidth=lw1_h, label=f'EHTA intensity ({true_mean:.2f})')

ax1.set_xlabel('Sample size', fontdict=font_xlabel)
ax1.set_ylabel('Estimate (K)', fontdict=font_ylabel)
ax1.set_title('Sample-size-dependent estimates', fontdict=font_title)
ax1.tick_params(axis='both', labelsize=xticklabelsize, direction='in',width=1,length=2)
ax1.grid(True, ls='--',alpha=0.3)
ax1.set_xlim(5,220)
ax1.set_ylim(ax1_ymin,ax1_ymax)
ax1_xticks = np.r_[[5],np.arange(20,220+20,20)]
ax1.set_xticks(ax1_xticks, [f'{x}' for x in ax1_xticks])

# =======  ax2 N_prof  =========
xticks_num = np.arange(0,t_size,12)
xticklabels_num = ['%d'%yr for yr in np.arange(1993,2021+1,1)]

ax2.plot(N_prof, color=color_num, lw=lw2)
ax2.set_xticks(xticks_num[::2], xticklabels_num[::2], rotation=45, fontdict=font_xticklabel)
ax2.tick_params(axis='both', labelsize=xticklabelsize, direction='in',width=1,length=2)
ax2.set_title(r'$N_{prof}$', fontdict=font_title)
ax2.set_xlim(0,t_size)
ax2.set_ylim(0,40)
ax2.grid(alpha=.3, ls='--')

# ======  ax3 I_MHW  =========
X = np.arange(t_size)/120

ax3.plot(X, I_MHW, color=(0,0,0), linewidth=lw3, label='MHW intensity')
ax3.set_title(r'$I_{MHW}$', fontdict=font_title)
ax3.set_xticks(xticks_num[::2]/120, xticklabels_num[::2], rotation=45, fontdict=font_xticklabel)
ax3.tick_params(axis='both', labelsize=xticklabelsize, direction='in',width=1,length=4)
ax3.set_xlim(0,t_size/120)
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('K',fontdict=font_ylabel,labelpad=2,va='bottom',ha='center')

plt.savefig('Fig2.png', dpi=600, bbox_inches='tight')
plt.savefig('Fig2.pdf', dpi=600, bbox_inches='tight')
plt.show()

