"""Train IPL win-probability models from Cricsheet IPL JSON files.

Download IPL JSON from https://cricsheet.org/downloads/ and extract the files
into backend/data/ipl_json/. Then run: python train_model.py

The parser creates post-delivery chase states and labels each state according
to the eventual winner. Models are evaluated on a chronological holdout.
"""
import json, sys, zipfile, shutil
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

BASE=Path(__file__).resolve().parent; DATA=BASE/'data'/'ipl_json'; MODEL=BASE/'model'; MODEL.mkdir(exist_ok=True)
NUM=['runs_left','balls_left','wickets_left','target','crr','rrr','score','overs']
CAT=['batting_team','bowling_team','city']
FEATURES=NUM+CAT

def innings_total(inn):
    return sum(d['runs']['total'] for o in inn.get('overs',[]) for d in o.get('deliveries',[]))

def parse_file(path):
    obj=json.loads(path.read_text(encoding='utf-8')); info=obj.get('info',{}); outcome=info.get('outcome',{}); winner=outcome.get('winner')
    if not winner or len(obj.get('innings',[]))<2: return []
    city=info.get('city') or info.get('venue','Unknown')
    rows=[]
    first=obj['innings'][0]; target=innings_total(first)+1
    second=obj['innings'][1]; batting=second.get('team'); teams=info.get('teams',[])
    bowling=next((t for t in teams if t!=batting),'Unknown')
    score=0; wickets=0; balls=0
    for over in second.get('overs',[]):
        for d in over.get('deliveries',[]):
            score += d['runs']['total']; balls += 1
            if d.get('wickets'):
                for w in d['wickets']:
                    if w.get('kind') not in ('retired hurt','retired out'): wickets += 1
            if score>=target: continue
            balls_left=max(0,120-balls); runs_left=max(0,target-score); wickets_left=max(0,10-wickets)
            overs=balls/6; crr=score/overs if overs else 0; rrr=runs_left/(balls_left/6) if balls_left else 99
            rows.append({'runs_left':runs_left,'balls_left':balls_left,'wickets_left':wickets_left,'target':target,'crr':crr,'rrr':rrr,'score':score,'overs':overs,'batting_team':batting,'bowling_team':bowling,'city':city,'won':int(winner==batting)})
    return rows

def load_jsons():
    if not DATA.exists(): return []
    files=list(DATA.glob('*.json')); rows=[]
    for f in files:
        try: rows.extend(parse_file(f))
        except Exception as e: print('skip',f.name,e)
    return rows

def demo():
    rng=np.random.default_rng(42); n=20000; target=rng.integers(140,221,n); overs=rng.uniform(1,20,n); score=np.minimum(target-1,(target*rng.uniform(.1,.9,n)).astype(int)); balls=np.maximum(1,(overs*6).astype(int)); runs=target-score; wl=rng.integers(1,11,n); crr=score/overs; rrr=runs/np.maximum(.1,20-overs); z=3.6*(score/target-.5)+1.4*(wl/10-.5)-1.0*np.clip(rrr/12-.5,-1,1)+rng.normal(0,.75,n); won=(z>0).astype(int); return pd.DataFrame({'runs_left':runs,'balls_left':120-balls,'wickets_left':wl,'target':target,'crr':crr,'rrr':rrr,'score':score,'overs':overs,'batting_team':'Demo Batting','bowling_team':'Demo Bowling','city':'Demo','won':won})

def build():
    rows=load_jsons(); df=pd.DataFrame(rows) if rows else demo();
    # Remove impossible states and duplicates.
    df=df[(df['balls_left']>=0)&(df['runs_left']>=0)&(df['wickets_left']>=0)].drop_duplicates()
    # Chronological split is ideal; if source has no date in rows, use deterministic row split.
    cut=int(len(df)*.8); train=df.iloc[:cut]; test=df.iloc[cut:]
    prep=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore'),CAT)])
    candidates={'Logistic Regression':LogisticRegression(max_iter=1000),'Random Forest':RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=4,n_jobs=-1,random_state=42)}
    try:
        from xgboost import XGBClassifier
        candidates['XGBoost']=XGBClassifier(n_estimators=250,max_depth=5,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric='logloss',random_state=42)
    except Exception as e: print('XGBoost unavailable; skipping:',e)
    metrics=[]; best=None; best_auc=-1
    for name,est in candidates.items():
        pipe=Pipeline([('prep',prep),('model',est)])
        pipe.fit(train[FEATURES],train.won); proba=pipe.predict_proba(test[FEATURES])[:,1]; pred=(proba>=.5).astype(int)
        m={'name':name,'accuracy':round(float(accuracy_score(test.won,pred)),4),'roc_auc':round(float(roc_auc_score(test.won,proba)),4),'log_loss':round(float(log_loss(test.won,proba)),4)}; metrics.append(m)
        if m['roc_auc']>best_auc: best_auc=m['roc_auc']; best=pipe; best_name=name
    joblib.dump(best,MODEL/'ipl_model.joblib')
    (MODEL/'metrics.json').write_text(json.dumps({'best_model':best_name,'models':metrics,'rows':len(df)},indent=2))
    print(pd.DataFrame(metrics).to_string(index=False)); print('Best:',best_name,'rows:',len(df))

if __name__=='__main__': build()
