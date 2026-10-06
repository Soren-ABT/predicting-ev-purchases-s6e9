"""Export predeclared Round30 sprint candidates; all new public scores are unknown."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata

H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'work/round28/public/defiaudit__s6e9-realmlp-oof'
J=R/'work/round28/public/jazivxt__s6e9-zoom-zoom-baseline'
rank=lambda a:rankdata(a,method='average')/len(a)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PINNED={
 'member05':('submission_round29_member05.csv','64ab00f679ada83a886f752b19296a66f35133736068ab4e42adfb3f74d73b36'),
 'own':('submission_round28.csv','c1349315bdde776333b8ff3896c8a5ffb6e73884b65d94c5998544c769c60391'),
 'reference':('submission_round28_public_reference.csv','3db4a64c5fd8162b1a3999381a690a8304e7f92d026e6919855c0d809cc08399'),
 'mega':('work/round28/public/defiaudit__s6e9-realmlp-oof/r8_m00_mega_verbatim.csv','23052891f418c5b146693f5d811ceb81be4b560defe669972cdb8c3ddc664f78'),
 'base':('work/round28/public/defiaudit__s6e9-realmlp-oof/r6_rebuilt_94651.csv','7ee8a5698217b3b5783a07526dc887ec5f47d42f667e487d19d79e00f3e52457'),
 'public75':('work/round28/public/defiaudit__s6e9-realmlp-oof/r8_m75.csv','11dfdf3ed5614c36af06730e8397fcb1491c06c3796780efcb367e5dd85447f0'),
 'jazivxt':('work/round28/public/jazivxt__s6e9-zoom-zoom-baseline/submission_latest_best.csv','ab7f541144312abe335c3524b49757fdcb1e0eb87675d19be67f5921de3c42bf'),
}


def main():
    ids=pd.read_csv(R/'sample_submission.csv',usecols=['id']).id.to_numpy();raw={}
    for name,(relative,digest) in PINNED.items():
        path=R/relative;assert sha(path)==digest,(name,'source hash changed')
        df=pd.read_csv(path,float_precision='round_trip')
        assert df.columns.tolist()==['id','Will_Buy_EV'] and df.id.is_unique and np.array_equal(df.id,ids)
        raw[name]=df.Will_Buy_EV.to_numpy()
    scores=pd.read_csv(P/'r8_results.csv').set_index('file').public_score
    assert scores['r8_m75.csv']==.94657
    ledger=pd.read_csv(J/'latest_scored_results.csv')
    record=ledger.loc[(ledger.submission_id==56369364)&ledger.origin.eq('own')]
    assert len(record)==1 and record.iloc[0].sha256==PINNED['jazivxt'][1] and record.iloc[0].public_score==.94656
    M,C,O=rank(raw['mega']),rank(raw['base']),rank(raw['own']);B=rank(raw['member05'])
    assert np.array_equal(rank(.5*M+.45*C+.05*O),B)
    assert np.array_equal(.75*M+.25*C,raw['public75'])
    model=json.loads((H/'model_report.json').read_text());new=rank(np.load(H/'test_candidate.npy'))
    specs=[
        ('submission_round30_sprint.csv',rank(.625*M+.325*C+.05*O),'rank(0.625*M+0.325*C+0.05*O)','fixed_public_weight_midpoint',None),
        ('submission_round30_sprint_m75.csv',rank(.75*M+.20*C+.05*O),'rank(0.75*M+0.20*C+0.05*O)','fixed_public_weight_endpoint',None),
        ('submission_round30_sprint_public75.csv',raw['public75'],'byte-copy r8_m75.csv; raw 0.75*M+0.25*C','pinned_public_reference',.94657),
        ('submission_round30_sprint_consensus10.csv',rank(.9*B+.1*rank(raw['jazivxt'])),'rank(0.90*rank(member05)+0.10*rank(jazivxt_latest))','fixed_public_consensus',None),
        ('submission_round30_sprint_model.csv',new,'Frozen screen-selected model/rank blend in model_report.json','local_model_experiment',None),
        ('submission_round30_sprint_modelmix.csv',rank(.50*M+.45*C+.05*new),'rank(0.50*M+0.45*C+0.05*rank(new_model))','fixed_model_substitution',None),
    ]
    entries=[]
    for filename,pred,formula,kind,author_score in specs:
        assert len(pred)==len(ids) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        data=(R/PINNED['public75'][0]).read_bytes() if kind=='pinned_public_reference' else pd.DataFrame({'id':ids,'Will_Buy_EV':pred}).to_csv(index=False,lineterminator='\n').encode()
        path=R/filename
        if path.exists():assert path.read_bytes()==data,'Refuse changed overwrite'
        else:path.write_bytes(data)
        back=pd.read_csv(path,float_precision='round_trip');assert np.array_equal(back.Will_Buy_EV,pred) and np.array_equal(back.id,ids)
        difference=rank(pred)-B
        entries.append({'filename':filename,'sha256':sha(path),'rows':len(pred),'formula':formula,'kind':kind,
                        'public_score':None,'author_reported_source_score':author_score,'joint_oof_auc':None,
                        'own_component':'new_screen_choice' if filename in ['submission_round30_sprint_model.csv','submission_round30_sprint_modelmix.csv'] else 'confirmed_round28_main_or_public_only',
                        'rank_changed_rows_vs_member05':int(np.count_nonzero(difference)),
                        'mean_abs_rank_change_vs_member05':float(np.mean(np.abs(difference))),
                        'max_abs_rank_change_vs_member05':float(np.max(np.abs(difference))),
                        'spearman_vs_member05':float(np.corrcoef(rank(pred),B)[0,1])})
    for a in entries:
        a['same_rank_as_other_exports']=[b['filename'] for b in entries if a['filename']!=b['filename'] and np.array_equal(rank(pd.read_csv(R/a['filename'],float_precision='round_trip').Will_Buy_EV),rank(pd.read_csv(R/b['filename'],float_precision='round_trip').Will_Buy_EV))]
    sources=[H/'PLAN.md',H/'AMENDMENT_solver.md',H/'EXPORT_NAMING.md',Path(__file__),H/'model_report.json',H/'test_candidate.npy',P/'r8_results.csv',J/'latest_scored_results.csv']
    report={'main_submission':'submission_round30_sprint.csv','all_new_user_public_scores_unknown':True,'model_promoted_locally':model['passed_promotion_gate'],
            'confirmed_best_displayed_score':.94657,'preserved_latest_confirmed_submission':'submission_round29_member05.csv',
            'exports':entries,'pinned_inputs':PINNED,'source_hashes':{str(p.relative_to(R)):sha(p) for p in sources},
            'caveats':['No complete joint OOF exists for the public-reference mixtures.',
                       '50% and 75% source weights have author-reported 0.94657; an intermediate weight is an untested hypothesis.',
                       'Rank distance/correlation is not a performance estimate.','Filename-only Nina score is not verified and that file is excluded.']}
    (H/'export_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
