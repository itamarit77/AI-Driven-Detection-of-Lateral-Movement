"""Generate all report figures (static PNG) with a validated, colour-blind-safe palette."""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import sys, os, json, warnings
import numpy as np, json, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_features import FEATURE_COLS

FIG = ROOT + '/figures'; os.makedirs(FIG, exist_ok=True)
for f in os.listdir(FIG): os.remove(os.path.join(FIG, f))
BENIGN, ATTACK = '#2a78d6', '#e34948'; LMDC, MORC = '#2a78d6', '#eb6834'; AQUA = '#1baf7a'
INK='#0b0b0b'; INK2='#52514e'; SURF='#fcfcfb'; GRID='#e6e6e2'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 11,
    'font.family': 'DejaVu Sans', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
NICE = json.load(open(ROOT + '/artifacts/nice_names.json'))
def nice(c): return NICE.get(c, c)

dl = pd.read_parquet(ROOT + '/features/lmd_features.parquet')
dm = pd.read_parquet(ROOT + '/features/mordor_features.parquet')
rl = pd.read_csv(ROOT + '/artifacts/ranking_LMD.csv'); rm = pd.read_csv(ROOT + '/artifacts/ranking_Mordor.csv')
cross = pd.read_csv(ROOT + '/artifacts/ranking_cross.csv')

# ---------- Fig 1: class distribution ----------
fig, ax = plt.subplots(figsize=(7.2, 3.4))
lm = dl['label_multi'].value_counts().reindex([0,1,2]).fillna(0); mm = dm['label'].value_counts().reindex([0,1]).fillna(0)
cats = ['Normal','EoRS','EoHT','Background','LM (weak label)']; vals=[lm[0],lm[1],lm[2],mm[0],mm[1]]
cols=[BENIGN,ATTACK,'#a11f1e',BENIGN,ATTACK]; x=[0,1,2,3.8,4.8]
ax.bar(x, vals, color=cols, width=0.8, zorder=3); ax.set_yscale('log'); ax.set_ylabel('Event count (log)')
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
for xi,v in zip(x,vals): ax.text(xi, v*1.15, f'{int(v):,}', ha='center', fontsize=8.5)
ax.text(1.0, 4e6, 'LMD-2023 (n = 1,752,836)', ha='center', fontsize=9.5, color=INK2, style='italic')
ax.text(4.3, 4e6, 'Mordor APT29 (n = 783,367)', ha='center', fontsize=9.5, color=INK2, style='italic')
ax.set_ylim(1e4, 1.2e7); ax.axvline(3.4, color=GRID, lw=1)
ax.set_title('Figure 1  Class distribution in both datasets', fontsize=11, weight='bold', loc='left')
plt.tight_layout(); plt.savefig(f'{FIG}/fig01_class_dist.png'); plt.close()

# ---------- Fig 2: capture timeline (temporal disjointness vs interleaving) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios':[1.35,1]})
d = dl.groupby([dl['ts'].dt.date, 'label']).size().unstack(fill_value=0)
xx = np.arange(len(d)); ax = axes[0]
ax.bar(xx, d[0], color=BENIGN, label='Benign', zorder=3); ax.bar(xx, d[1], bottom=d[0], color=ATTACK, label='Attack', zorder=3)
ax.set_xticks(xx); ax.set_xticklabels([str(i)[2:] for i in d.index], rotation=90, fontsize=9)
ax.set_ylabel('Events per calendar day'); ax.set_title('LMD-2023: one calendar day contains both classes (sequentially)', fontsize=9.5)
ax.axhline(86400, color=INK2, lw=0.8, ls='--'); ax.text(-0.4, 86400*1.03, '86,400/day = 1 event/s', fontsize=7.5, color=INK2, ha='left', va='bottom')
ax.set_ylim(0, 100000); ax.legend(fontsize=8.5, loc='upper right')
m = dm.groupby([dm['ts'].dt.floor('5min'), 'label']).size().unstack(fill_value=0); ax = axes[1]; xx=np.arange(len(m))
ax.bar(xx, m[0], color=BENIGN, label='Background', zorder=3); ax.bar(xx, m[1], bottom=m[0], color=ATTACK, label='LM (weak)', zorder=3)
ax.set_xticks(xx); ax.set_xticklabels([t.strftime('%H:%M') for t in m.index], rotation=90, fontsize=9)
nboth=int(((m[0]>0)&(m[1]>0)).sum()); json.dump({'bins_total':int(len(m)),'bins_both':nboth}, open(ROOT + '/artifacts/mordor_bins.json','w'))
ax.set_ylabel('Events per 5-min bin'); ax.set_title(f'Mordor APT29: classes interleave in {nboth} of {len(m)} 5-min bins', fontsize=9.5); ax.legend(fontsize=8.5)
fig.suptitle('Figure 2  Capture structure: LMD-2023 classes are almost disjoint in time (49.5 s overlap); Mordor classes are interleaved', fontsize=11, weight='bold', x=0.01, ha='left')
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f'{FIG}/fig02_timeline.png'); plt.close()

# ---------- Fig 3: class-conditional box plots (LMD) ----------
feats_box = ['f_host_out_degree_1h','f_distinct_dstip_300s','f_inbound_adminport_60s','f_host_proc_create_60s']
fig, axes = plt.subplots(1,4, figsize=(12,3.6))
for ax,f in zip(axes, feats_box):
    b = dl.loc[dl['label']==0, f].clip(upper=dl[f].quantile(0.99)); a = dl.loc[dl['label']==1, f].clip(upper=dl[f].quantile(0.99))
    bp = ax.boxplot([b, a], patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color=INK,lw=1.5))
    bp['boxes'][0].set_facecolor(BENIGN); bp['boxes'][1].set_facecolor(ATTACK)
    for box in bp['boxes']: box.set_alpha(0.85)
    ax.set_xticklabels(['Benign','Attack'], fontsize=9); ax.set_title(nice(f), fontsize=9.5)
fig.suptitle('Figure 3  Class-conditional distributions of four engineered features (LMD-2023, 99th-pct clipped)', fontsize=11, weight='bold', x=0.01, ha='left')
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f'{FIG}/fig03_boxplots.png'); plt.close()

# ---------- Fig 4: Spearman correlation heatmap ----------
corr = pd.read_csv(ROOT + '/artifacts/corr_lmd_spearman.csv', index_col=0)
keep = [c for c in corr.columns if dl[c].std() > 0]; corr = corr.loc[keep, keep]; labels=[nice(c) for c in keep]
fig, ax = plt.subplots(figsize=(9.4,8.4)); im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8.5); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Spearman ρ', fontsize=9)
ax.set_title('Figure 4  Spearman correlation matrix (LMD-2023, non-constant features)', fontsize=10.5, weight='bold', loc='left')
plt.tight_layout(); plt.savefig(f'{FIG}/fig04_corr.png'); plt.close()

# ---------- Fig 5/6: importance ----------
def imp_fig(r, title, fname, topn=16):
    r = r.copy()
    for c in ['rf_mdi','rf_perm_auc_drop','lgbm_gain']:
        r[c] = r[c].clip(lower=0); s = r[c].sum(); r[c+'_n'] = r[c]/s if s>0 else r[c]
    r['score'] = r[['rf_mdi_n','lgbm_gain_n']].mean(axis=1); r = r.sort_values('score', ascending=True).tail(topn)
    y = np.arange(len(r)); h=0.26; fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.barh(y+h, r['rf_mdi_n'], height=h, color=LMDC, label='RF Gini/MDI', zorder=3)
    ax.barh(y, r['lgbm_gain_n'], height=h, color=MORC, label='LightGBM gain', zorder=3)
    ax.barh(y-h, r['rf_perm_auc_drop']/max(r['rf_perm_auc_drop'].max(),1e-9), height=h, color=AQUA, label='RF permutation (AUC drop, scaled)', zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([nice(c) for c in r['feature']], fontsize=9); ax.set_xlabel('Normalised importance')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9); ax.set_title(title, fontsize=11, weight='bold', loc='left')
    plt.tight_layout(); plt.savefig(f'{FIG}/{fname}'); plt.close()
imp_fig(rl, 'Figure 6  LMD-2023 tree-based feature importance (top 16)', 'fig06_imp_lmd.png')
imp_fig(rm, 'Figure 7  Mordor APT29 tree-based feature importance (top 16)', 'fig07_imp_mordor.png')

# ---------- Fig 7: cross-dataset shift ----------
UNIF = ['f_diurnal_sin','f_initiated','f_host_out_degree_1h','f_distinct_dstip_300s','f_event_category']
fig, axes = plt.subplots(2,3, figsize=(9.6,6.4)); axes = axes.ravel(); axes[5].axis('off')
for ax,f in zip(axes, UNIF):
    lv = dl[f].replace([np.inf,-np.inf],np.nan).dropna(); mv = dm[f].replace([np.inf,-np.inf],np.nan).dropna()
    if dl[f].nunique()<=6 or f=='f_event_category':
        lc = lv.value_counts(normalize=True).sort_index(); mc = mv.value_counts(normalize=True).sort_index()
        idx = sorted(set(lc.index)|set(mc.index)); xx=np.arange(len(idx)); w=0.4
        ax.bar(xx-w/2,[lc.get(i,0) for i in idx],width=w,color=LMDC,label='LMD',zorder=3); ax.bar(xx+w/2,[mc.get(i,0) for i in idx],width=w,color=MORC,label='Mordor',zorder=3)
        ax.set_xticks(xx); ax.set_xticklabels([str(int(i)) for i in idx], fontsize=8, rotation=90 if len(idx) > 6 else 0); ax.set_ylabel('Proportion')
        if len(idx) > 6: ax.set_xlabel('category code (names: artifacts/report_tables.json)', fontsize=8)
    else:
        hi = np.nanquantile(np.concatenate([lv.values,mv.values]),0.98); bins=np.linspace(min(lv.min(),mv.min()), hi, 30)
        ax.hist(lv.clip(upper=hi),bins=bins,density=True,color=LMDC,alpha=0.55,label='LMD',zorder=3); ax.hist(mv.clip(upper=hi),bins=bins,density=True,color=MORC,alpha=0.55,label='Mordor',zorder=3)
        ax.set_ylabel('Density')
    ax.set_title(nice(f), fontsize=10); ax.legend(fontsize=9)
fig.suptitle('Figure 8  Cross-dataset distribution shift of candidate features (LMD-2023 vs Mordor APT29)', fontsize=11, weight='bold', x=0.01, ha='left')
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f'{FIG}/fig08_shift.png'); plt.close()

# ---------- Fig 8: rank agreement ----------
fig, ax = plt.subplots(figsize=(7.4,6.0)); xr = cross['rank_LMD']; yr = cross['rank_Mordor']
ax.scatter(xr, yr, s=42, color=LMDC, zorder=3, edgecolor='white', linewidth=0.6); ax.plot([0,42],[0,42], color=INK2, lw=1, ls='--', zorder=2)
OFF = {'f_weekday':(-6,-14,'right'), 'f_naive_relational_pid':(6,6,'left'), 'f_host_out_degree_1h':(-10,8,'right'),
       'f_distinct_dstip_300s':(-8,-13,'right'), 'f_event_category':(-8,5,'right'), 'f_seconds_in_day':(6,4,'left'),
       'f_diurnal_sin':(6,-11,'left'), 'f_diurnal_cos':(-8,-11,'right'), 'f_host_netconn_60s':(4,-12,'left'),
       'f_inbound_peers_300s':(6,-11,'left'), 'f_image_is_lolbin':(6,4,'left'), 'f_sec_observed':(6,4,'left')}
for _,row in cross.iterrows():
    if row['rank_gap']>=12 or row['rank_LMD']<=3 or row['rank_Mordor']<=3:
        if row['feature']=='f_distinct_dstip_300s':   # crowded corner: leader line to an empty region
            ax.annotate(nice(row['feature']), (row['rank_LMD'],row['rank_Mordor']), xytext=(7,20.5), textcoords='data', fontsize=7.2, color=INK2, ha='center',
                        arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2, shrinkA=0, shrinkB=3)); continue
        dx,dy,ha = OFF.get(row['feature'], (5,4,'left'))
        ax.annotate(nice(row['feature']), (row['rank_LMD'],row['rank_Mordor']), fontsize=7.2, color=INK2, xytext=(dx,dy), textcoords='offset points', ha=ha)
ax.set_xlabel('Importance rank in LMD-2023 (1 = most important)'); ax.set_ylabel('Importance rank in Mordor APT29 (1 = most important)')
ax.invert_xaxis(); ax.invert_yaxis(); ax.set_title('Figure 9  Cross-dataset rank agreement (composite importance)', fontsize=11, weight='bold', loc='left')
plt.tight_layout(); plt.savefig(f'{FIG}/fig09_cross_imp.png'); plt.close()

# ---------- Fig 9: modality composition ----------
comp = pd.crosstab(dl['eventid'], dl['label'], normalize='columns')*100; order = comp.sort_values(1, ascending=False).head(8).index; comp = comp.loc[order]
y=np.arange(len(order)); h=0.38; fig, ax = plt.subplots(figsize=(8.0,4.2))
ax.barh(y+h/2, comp[0], height=h, color=BENIGN, label='Benign', zorder=3); ax.barh(y-h/2, comp[1], height=h, color=ATTACK, label='Attack', zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f'EID {int(e)}' for e in order], fontsize=9); ax.set_xlabel('Share of that class (%)'); ax.legend(fontsize=9)
ax.set_title('Figure 5  Event-ID composition: benign vs attack rows (LMD-2023)', fontsize=11, weight='bold', loc='left')
plt.tight_layout(); plt.savefig(f'{FIG}/fig05_modality.png'); plt.close()

# ---------- Fig 10/11: complete-pool class-divergence evidence (all features, full-row statistics) ----------
def evidence_fig(csv, title, fname):
    st = pd.read_csv(csv).copy(); st['mi_bits'] = st['mi_bits'].fillna(0)
    st = st.sort_values('mi_bits', ascending=True)
    y = np.arange(len(st)); fig, ax = plt.subplots(figsize=(8.4, 9.2))
    cols = [ATTACK if (r.p_holm < 0.05 and abs(r.effect_size) >= 0.1) else (LMDC if r.p_holm < 0.05 else '#b0b6c0') for r in st.itertuples()]
    ax.barh(y, st['mi_bits'], color=cols, zorder=3)
    for yi, r in zip(y, st.itertuples()):
        lab = ('δ=' if r.effect_type == 'CliffsDelta' else 'V=') + f'{r.effect_size:+.2f}' if r.effect_type == 'CliffsDelta' else ('V=' + f'{r.effect_size:.2f}')
        ax.text(r.mi_bits + 0.002, yi, lab + ('*' if r.p_holm < 0.05 else ''), va='center', fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([nice(c) for c in st['feature']], fontsize=8.5); ax.set_xlabel('Mutual information with the label (bits)')
    ax.set_xlim(0, max(st['mi_bits'].max() * 1.35, 0.05))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ATTACK, label='Holm p < .05 and |effect| ≥ 0.1'), Patch(color=LMDC, label='Holm p < .05, |effect| < 0.1'), Patch(color='#b0b6c0', label='not significant / constant')], fontsize=8.5, loc='lower right')
    ax.set_title(title, fontsize=11, weight='bold', loc='left'); plt.tight_layout(); plt.savefig(f'{FIG}/{fname}'); plt.close()
evidence_fig(ROOT + '/artifacts/eda_stats_LMD.csv', 'Figure 10  Complete candidate-pool evidence, LMD-2023 (all rows)', 'fig10_evidence_lmd.png')
evidence_fig(ROOT + '/artifacts/eda_stats_Mordor.csv', 'Figure 11  Complete candidate-pool evidence, Mordor APT29 (weak labels)', 'fig11_evidence_mordor.png')

from PIL import Image
dims = {f: Image.open(f'{FIG}/{f}').size for f in sorted(os.listdir(FIG)) if f.endswith('.png')}
json.dump(dims, open(f'{FIG}/dims.json','w')); print("figures:", list(dims))
