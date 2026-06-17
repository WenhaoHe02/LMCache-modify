import json, statistics, time, urllib.request
from pathlib import Path
URL='http://127.0.0.1:8000/v1/completions'
MODEL='deepseek-v4-pro'
OUT=Path('/tmp/dsv4_csaon_hcaoff_extra_248k_hits_20260602.jsonl')
SENT='The quick brown fox jumped over the lazy dog near the river bank. '
PROMPT='CSAON_HCAOFF_248K_4HIT_20260602_UNIQUE_PREFIX. ' + SENT * 17745 + '\nFinal question: answer with one token.\n'
OUT.write_text('', encoding='utf-8')

def write(row):
    row['ts']=time.time()
    with OUT.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)

def req(label):
    payload={'model':MODEL,'prompt':PROMPT,'max_tokens':1,'temperature':0,'ignore_eos':True,'stream':False}
    r=urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    st=time.perf_counter()
    with urllib.request.urlopen(r, timeout=2400) as resp: obj=json.loads(resp.read())
    el=time.perf_counter()-st
    return {'label':label,'elapsed_s':el,'usage':obj.get('usage'),'text':(obj.get('choices') or [{}])[0].get('text')}
write({'event':'start','mode':'csa_on_hca_off_extra_248k_full_hits','prompt_chars':len(PROMPT),'out':str(OUT)})
vals=[]
for i in range(1,7):
    row=req(f'csaon_hcaoff_extra_248k_hit_{i}')
    write(row)
    vals.append(row['elapsed_s'])
    if i != 6:
        write({'event':'sleep_before_hit','hit_next':i+1,'seconds':60})
        time.sleep(60)
write({'event':'summary','hit_elapsed_s':vals,'hit_min_s':min(vals),'hit_mean_s':statistics.mean(vals),'hit_max_s':max(vals)})
write({'event':'done','out':str(OUT)})
