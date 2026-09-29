"""Export every table row rendered in the report (and code-derived counts) to artifacts/report_tables.json."""
import os as _os
ROOT = _os.environ.get('M2_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # package root: <ROOT>/{scripts,artifacts,figures,features,data}

import pandas as pd, json, math, sys, os
sys.path.insert(0,ROOT + '/scripts'); from common_features import FEATURE_COLS, FAMILY, CAT_ORDINAL
NICE=json.load(open(ROOT + '/artifacts/nice_names.json')); A=ROOT + '/artifacts/'
sl=pd.read_csv(A+'eda_stats_LMD.csv'); sm=pd.read_csv(A+'eda_stats_Mordor.csv')
def fmt(x,d=2):
    if pd.isna(x): return '—'
    return f'{x:,.0f}' if abs(x)>=1000 else f'{x:.{d}f}'
def pstr(pv): return '<0.001' if pv<1e-3 else f'{pv:.3f}'
def es(r): return ('δ='+fmt(r['effect_size'])) if r['effect_type']=='CliffsDelta' else ('V='+fmt(r['effect_size']))
CATEG={'f_event_category','f_weekday'}
def row(r):
    if r.feature in CATEG:   # nominal / cyclic-ordinal codes: mean and AUC are not meaningful
        return [NICE[r.feature], '—', '—', '—', es(r), fmt(r.mi_bits,3), pstr(r.p_holm)]
    return [NICE[r.feature], fmt(r.benign_mean,4 if abs(r.benign_mean)<0.01 else 2), fmt(r.attack_mean,4 if abs(r.attack_mean)<0.01 else 2), fmt(r.auc_attack_positive,3), es(r), fmt(r.mi_bits,3), pstr(r.p_holm)]
out={'lmd_stats':[row(r) for _,r in sl[~sl['constant']].head(10).iterrows()]}
top=list(sm[~sm['constant']].head(8)['feature']); sel=top+[f for f in FAMILY['auth_service'] if f not in top]
out['mordor_stats']=[row(sm[sm.feature==f].iloc[0]) for f in sel]
red=json.load(open(A+'redundancy.json')); out['redundancy']=[[NICE[a],NICE[b],str(c)] for a,b,c in red['redundant_pairs']]
out['zero_variance']={k:[x[2:] for x in v] for k,v in red['zero_variance'].items()}
def rank_rows(path,n=10):
    r=pd.read_csv(path); r['s']=(r['rf_mdi']+r['lgbm_gain'])/2; r=r.sort_values('s',ascending=False).head(n)
    return [[NICE[x.feature], fmt(x.rf_mdi,3), fmt(max(x.rf_perm_auc_drop,0),4), fmt(x.lgbm_gain,3)] for _,x in r.iterrows()]
out['rank_lmd']=rank_rows(A+'ranking_LMD.csv'); out['rank_mordor']=rank_rows(A+'ranking_Mordor.csv')
cr=pd.read_csv(A+'ranking_cross.csv'); cr=cr[cr['rank_gap']>=cr['rank_gap'].iloc[7]]   # top 8 plus ties at the cut
def rk(v): return str(int(v)) if float(v).is_integer() else f'{v:.1f}'
out['cross']=[[NICE[x.feature],rk(x.rank_LMD),rk(x.rank_Mordor),rk(x.rank_gap)] for _,x in cr.iterrows()]
out['summary']=json.load(open(A+'ranking_summary.json')); out['split']=json.load(open(A+'split_audit.json'))
out['meta']={'LMD':json.load(open(A+'eda_meta_LMD.json')),'Mordor':json.load(open(A+'eda_meta_Mordor.json'))}
out['n_pool']=len(FEATURE_COLS)
fams=['temporal','velocity','fanout','graph','user','lineage','cmdline','network','credential','tooltransfer','auth_service','modality','structural']
TIME5=('f_seconds_in_day','f_diurnal_sin','f_diurnal_cos','f_weekday','f_is_business_hours')
unified=[f for fam in fams for f in FAMILY[fam] if f not in TIME5 and f!='f_cmdline_len']
dl=pd.read_parquet(ROOT + '/features/lmd_features.parquet',columns=['f_event_category']); dm=pd.read_parquet(ROOT + '/features/mordor_features.parquet',columns=['f_event_category'])
ncat=len(set(dl.f_event_category.unique())|set(dm.f_event_category.unique()))
out['unified']=unified; out['n_unified']=len(unified); out['n_event_categories']=ncat; out['D_lstm']=len(unified)-1+ncat
from common_features import CAT_ORDINAL
inv={v:k for k,v in CAT_ORDINAL.items()}
cats=sorted(inv.get(c,'other') for c in (set(dl.f_event_category.unique())|set(dm.f_event_category.unique())))
out['event_category_names']=cats
# ---- derived quantities quoted in the prose (all from the artifact tables / parquet files) ----
SL=sl.set_index('feature'); SM=sm.set_index('feature')
out['mordor_labels']=json.load(open(A+'mordor_labels.json'))
out['stat']={
  'mordor_lolbin_V': round(float(SM.loc['f_image_is_lolbin','effect_size']),2),
  'mordor_fanout_auc': [round(float(SM.loc[f,'auc_attack_positive']),2) for f in ['f_host_out_degree_1h','f_distinct_dstport_300s','f_distinct_dstip_300s','f_host_netconn_60s']],
  'mordor_auth_V_max': round(float(SM.loc[FAMILY['auth_service'],'effect_size'].max()),3),
  'mordor_weekday_mi': round(float(SM.loc['f_weekday','mi_bits']),3), 'mordor_secday_mi': round(float(SM.loc['f_seconds_in_day','mi_bits']),3),
  'lmd_weekday_mi': round(float(SL.loc['f_weekday','mi_bits']),3), 'lmd_weekday_V': round(float(SL.loc['f_weekday','effect_size']),2),
  'lmd_secday_mi': round(float(SL.loc['f_seconds_in_day','mi_bits']),3), 'lmd_pid_mi': round(float(SL.loc['f_naive_relational_pid','mi_bits']),3),
  'lmd_pid_attack_mean': round(float(SL.loc['f_naive_relational_pid','attack_mean']),3),
  'lmd_credmask_V': round(float(SL.loc['f_cred_access_mask','effect_size']),2),
  'lmd_lsass_attack_pct': round(float(SL.loc['f_lsass_access','attack_mean'])*100,2), 'lmd_credmask_attack_pct': round(float(SL.loc['f_cred_access_mask','attack_mean'])*100,1),
  'mi_dstip': {'LMD': round(float(SL.loc['f_distinct_dstip_300s','mi_bits']),2), 'Mordor': round(float(SM.loc['f_distinct_dstip_300s','mi_bits']),2)},
  'lmd_max_mi': round(float(out['meta']['LMD']['max_mi_bits']),3), 'lmd_HY': round(float(out['meta']['LMD']['label_entropy_bits']),3),
}
rl_=pd.read_csv(A+'ranking_LMD.csv').set_index('feature'); perm_sorted=rl_['rf_perm_auc_drop'].sort_values(ascending=False)
out['stat']['lmd_perm_top']=[perm_sorted.index[0], round(float(perm_sorted.iloc[0]),4), round(float(perm_sorted.iloc[1]),4)]
cx=pd.read_csv(A+'ranking_cross.csv').set_index('feature')
flags=['f_initiated','f_admin_port_flag','f_is_ipv6']
out['stat']['ranks']={'weekday_LMD': float(cx.loc['f_weekday','rank_LMD']), 'weekday_Mordor': float(cx.loc['f_weekday','rank_Mordor']),
  'weekday_Mordor_composite': float(cx.loc['f_weekday','Mordor']), 'pid_LMD': float(cx.loc['f_naive_relational_pid','rank_LMD']), 'pid_Mordor_composite': float(cx.loc['f_naive_relational_pid','Mordor']),
  'flags_LMD': [float(cx.loc[flags,'rank_LMD'].min()), float(cx.loc[flags,'rank_LMD'].max())], 'flags_Mordor': [float(cx.loc[flags,'rank_Mordor'].min()), float(cx.loc[flags,'rank_Mordor'].max())],
  'agg_LMD_max': float(cx.loc[['f_host_out_degree_1h','f_distinct_dstip_300s','f_distinct_dstport_300s','f_host_evt_rate_60s','f_host_proc_create_60s','f_host_netconn_60s'],'rank_LMD'].max()),
  'agg_Mordor_max': float(cx.loc[['f_host_out_degree_1h','f_distinct_dstip_300s','f_distinct_dstport_300s','f_host_evt_rate_60s','f_host_proc_create_60s','f_host_netconn_60s'],'rank_Mordor'].max())}
# Mordor session attack densities (sessions = gaps > 10 min) and the two window lengths
_dm=pd.read_parquet(ROOT + '/features/mordor_features.parquet',columns=['ts','label']).sort_values('ts')
_sess=(_dm['ts'].diff().dt.total_seconds()>600).cumsum()
out['stat']['mordor_sessions']=[{'start':g['ts'].min().strftime('%H:%M'),'end':g['ts'].max().strftime('%H:%M'),'minutes':round((g['ts'].max()-g['ts'].min()).total_seconds()/60,1),'attack_pct':round(float(g['label'].mean()*100),1),'n':int(len(g))} for _,g in _dm.groupby(_sess)]
# LMD composition facts (label-independent of Mordor)
_dl=pd.read_parquet(ROOT + '/features/lmd_features.parquet',columns=['eventid','label'])
_b=_dl[_dl.label==0]; _a=_dl[_dl.label==1]
out['stat']['lmd_benign_eid3_pct']=round(float((_b.eventid==3).mean()*100),0)
_sh=(_a.eventid.value_counts(normalize=True)*100)
out['stat']['lmd_attack_eid_pct']={str(int(k)): round(float(v),0) for k,v in _sh.head(4).items()}
out['stat']['lmd_eid1_10_11_pct']=round(float(_dl.eventid.isin([1,10,11]).mean()*100),2)
out['stat']['lmd_benign_eid10_rows']=int((_b.eventid==10).sum())
out['stat']['lmd_attack_eid10_pct']=round(float((_a.eventid==10).mean()*100),1)
# ---- LMD temporal overlap between classes (Table 4 wording) ----
_dlt=pd.read_parquet(ROOT + '/features/lmd_features.parquet',columns=['ts','label'])
_fa=_dlt.loc[_dlt.label==1,'ts'].min(); _lb=_dlt.loc[_dlt.label==0,'ts'].max()
out['stat']['lmd_overlap']={'seconds': round(float((_lb-_fa).total_seconds()),1), 'benign_after_first_attack': int(((_dlt.label==0)&(_dlt.ts>=_fa)).sum()), 'attack_before_last_benign': int(((_dlt.label==1)&(_dlt.ts<=_lb)).sum())}
# ---- LMD reading numbers (Reading LMD paragraph) ----
def _r(f,col): return float(SL.loc[f,col])
out['stat']['lmd_read']={'outdeg_mi':round(_r('f_host_out_degree_1h','mi_bits'),2),'dstip_mi':round(_r('f_distinct_dstip_300s','mi_bits'),2),
  'low_auc':[round(_r(f,'auc_attack_positive'),2) for f in ['f_host_out_degree_1h','f_distinct_dstip_300s','f_host_netconn_60s']],
  'inbound_auc':round(_r('f_inbound_adminport_60s','auc_attack_positive'),2),'proc_auc':round(_r('f_host_proc_create_60s','auc_attack_positive'),2),
  'peers_auc':round(_r('f_inbound_peers_300s','auc_attack_positive'),2),'peers_mi':round(_r('f_inbound_peers_300s','mi_bits'),3)}
out['stat']['mordor_inbound_auc']=round(float(SM.loc['f_inbound_adminport_60s','auc_attack_positive']),2)
# ---- per-family empirical evidence flags (Table 12): significant (Holm p<.05) AND |effect| >= 0.1 in that dataset ----
NATT={'LMD': out['meta']['LMD']['attack_prevalence']*out['meta']['LMD']['n_rows'], 'Mordor': out['meta']['Mordor']['attack_prevalence']*out['meta']['Mordor']['n_rows']}
def _ev(S,f,ds=None):
    # evidence = Holm p < .05 and (|effect| >= 0.1, or for a binary flag: attack-rate lift >= 3 on >= 100 attack-row fires with a non-zero benign rate)
    r=S.loc[f]
    if not (r['p_holm']<0.05): return False
    if abs(r['effect_size'])>=0.1: return True
    if r['effect_type']=='CramersV' and ds is not None and r['benign_mean']>0:
        return bool(r['attack_mean']/r['benign_mean']>=3 and r['attack_mean']*NATT[ds]>=100)
    return False
out['family_evidence']={fam:{'LMD':any(_ev(SL,f,'LMD') for f in FAMILY[fam] if f in SL.index),'Mordor':any(_ev(SM,f,'Mordor') for f in FAMILY[fam] if f in SM.index)} for fam in fams}
out['feature_evidence']={f:{'LMD':_ev(SL,f,'LMD'),'Mordor':_ev(SM,f,'Mordor')} for f in FEATURE_COLS}
out['n_unvalidated']=sum(1 for f in unified if not (_ev(SL,f,'LMD') or _ev(SM,f,'Mordor')))
out['unvalidated']=[f for f in unified if not (_ev(SL,f,'LMD') or _ev(SM,f,'Mordor'))]
out['unvalidated_by_family']={fam:[f for f in FAMILY[fam] if f in out['unvalidated']] for fam in fams}
# ---- why each unvalidated column fails the criterion, per dataset (quoted in 5.1 / Appendix A) ----
NBEN={'LMD': (1-out['meta']['LMD']['attack_prevalence'])*out['meta']['LMD']['n_rows'], 'Mordor': (1-out['meta']['Mordor']['attack_prevalence'])*out['meta']['Mordor']['n_rows']}
def _why(S,f,ds):
    r=S.loc[f]; d={'p_holm':float(r['p_holm']),'effect_type':str(r['effect_type']),'effect':float(r['effect_size']),'constant':bool(r['constant'])}
    if r['effect_type']=='CramersV':
        d['attack_fires']=int(round(r['attack_mean']*NATT[ds])); d['benign_fires']=int(round(r['benign_mean']*NBEN[ds]))
        d['lift']=(float(r['attack_mean']/r['benign_mean']) if r['benign_mean']>0 else None)
    if d['constant']: d['reason']='constant'
    elif not d['p_holm']<0.05: d['reason']='not significant'
    elif abs(d['effect'])>=0.1: d['reason']='validated'
    elif r['effect_type']!='CramersV': d['reason']='effect below threshold'
    elif d['lift'] is None: d['reason']='no benign fires (lift undefined)'
    elif d['lift']<1: d['reason']='attack rate below benign'
    elif d['lift']<3: d['reason']='lift below 3'
    elif d['attack_fires']<100: d['reason']='fewer than 100 attack fires'
    else: d['reason']='validated'
    return d
out['unvalidated_why']={f:{'LMD':_why(SL,f,'LMD'),'Mordor':_why(SM,f,'Mordor')} for f in out['unvalidated']}
# ---- scaling audit (Chapter 5.3) ----
if os.path.exists(A+'scaling_audit.json'): out['scaling']=json.load(open(A+'scaling_audit.json'))
out['addr_kinds']={k:json.load(open(A+f'{k.lower()}_address_kinds.json')) for k in ['LMD','Mordor'] if os.path.exists(A+f'{k.lower()}_address_kinds.json')}


import os
out['mordor_bins']=json.load(open(A+'mordor_bins.json'))   # written by figures.py; must exist (no typed fallback)
_dme=pd.read_parquet(ROOT + '/features/mordor_features.parquet',columns=['eventid','f_initiated']); out['mordor_eid3_pct']=round(float((_dme.eventid==3).mean()*100),1); out['mordor_initiated_pct']=round(float(_dme.f_initiated.mean()*100),1)
_dle=pd.read_parquet(ROOT + '/features/lmd_features.parquet',columns=['eventid','f_initiated']); out['lmd_eid3_pct']=round(float((_dle.eventid==3).mean()*100),1); out['lmd_initiated_pct']=round(float(_dle.f_initiated.mean()*100),1)
META={'temporal':('Temporal','SystemTime per host','@timestamp per host','robust'),
 'velocity':('Velocity','per Computer','per Hostname','robust'),
 'fanout':('Fan-out','EID 3 (outbound dest.; inbound SourceIp)','EID 3 (outbound dest.); inbound SourceIp + Security client IpAddress','robust'),
 'graph':('Graph','EID 3, Initiated = true','EID 3, Initiated = true','robust / binary'),
 'user':('User','User','User','robust'),
 'lineage':('Lineage','EID 1; Image-only flags on any event','EID 1 (+4688); Image-only flags on any event','binary'),
 'cmdline':('Command line','EID 1','EID 1 / 4688','robust / binary'),
 'network':('Network','EID 3','EID 3','binary'),
 'credential':('Credential','EID 10','EID 10','binary'),
 'tooltransfer':('Tool transfer','EID 11','EID 11','binary'),
 'auth_service':('Auth / service','not collected → 0, sec_observed = 0','Security 4624, 4648, 4769, 5140, 5145; System 7045, Security 4697','binary'),
 'modality':('Modality','0 / cummax(EID 3 seen)','cummax(auth/service event seen) / cummax(EID 3 seen)','binary'),
 'structural':('Structural',f'EventID → {ncat} one-hot (observed levels; {len(CAT_ORDINAL)+1} registry slots for the LSTM, §6.1)',f'EventID → {ncat} one-hot (observed levels; {len(CAT_ORDINAL)+1} registry slots)','one-hot')}
schema=[]
SL_=sl.set_index('feature'); SM_=sm.set_index('feature')
def _evtxt(fam):
    fl=[f for f in FAMILY[fam] if f in unified]; a=any(_ev(SL_,f,'LMD') for f in fl); b=any(_ev(SM_,f,'Mordor') for f in fl)
    return 'LMD + Mordor' if (a and b) else ('LMD' if a else ('Mordor' if b else 'none (unvalidated)'))
for fam in fams:
    feats=[f[2:] for f in FAMILY[fam] if f in unified]
    if feats: schema.append([META[fam][0], ', '.join(feats), META[fam][1], META[fam][2], META[fam][3], _evtxt(fam)])
out['schema_rows']=schema
out['n_registry_levels']=len(CAT_ORDINAL)+1          # registry categories + 'other': the one-hot vocabulary the Milestone-3 code allocates
out['D_lstm_registry']=out['n_unified']-1+out['n_registry_levels']
out['n_by_family']={META[f][0]:len([x for x in FAMILY[f] if x in unified]) for f in fams}
def clean(o):
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    if isinstance(o,float) and (math.isnan(o) or math.isinf(o)): return None
    return o
json.dump(clean(out),open(A+'report_tables.json','w'),indent=1); print('pool',out['n_pool'],'unified',out['n_unified'],'D',out['D_lstm'])
