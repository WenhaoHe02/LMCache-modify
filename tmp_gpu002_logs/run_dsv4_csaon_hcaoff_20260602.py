import json, statistics, time, urllib.request
from pathlib import Path

URL = 'http://127.0.0.1:8000/v1/completions'
MODEL = 'deepseek-v4-pro'
OUT = Path('/tmp/dsv4_csaon_hcaoff_20260602.jsonl')
SENT = 'The quick brown fox jumped over the lazy dog near the river bank. '
OUT.write_text('', encoding='utf-8')

def write(row):
    row['ts'] = time.time()
    with OUT.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)

def request(label, prompt, stream=False):
    payload = {'model': MODEL, 'prompt': prompt, 'max_tokens': 1, 'temperature': 0, 'ignore_eos': True, 'stream': stream}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    if stream:
        events = 0; text = ''; ttft = None
        with urllib.request.urlopen(req, timeout=2400) as resp:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                if events == 0:
                    ttft = time.perf_counter() - start
                events += 1
                obj = json.loads(data)
                text += (obj.get('choices') or [{}])[0].get('text') or ''
        total = time.perf_counter() - start
        return {'label': label, 'ttft_s': ttft, 'total_s': total, 'events': events, 'text': text}
    with urllib.request.urlopen(req, timeout=2400) as resp:
        obj = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    return {'label': label, 'elapsed_s': elapsed, 'usage': obj.get('usage'), 'text': (obj.get('choices') or [{}])[0].get('text')}

write({'event': 'start', 'mode': 'csa_on_hca_off', 'out': str(OUT)})

prompt_248 = 'CSAON_HCAOFF_248K_4HIT_20260602_UNIQUE_PREFIX. ' + SENT * 17745 + '\nFinal question: answer with one token.\n'
write({'event': 'item_start', 'item': 'csaon_hcaoff_248k_4hit', 'prompt_chars': len(prompt_248)})
row = request('csaon_hcaoff_248k_cold_store', prompt_248, stream=False)
row.update({'item': 'csaon_hcaoff_248k_4hit', 'prompt_chars': len(prompt_248)})
write(row)
write({'event': 'sleep_before_hit', 'item': 'csaon_hcaoff_248k_4hit', 'seconds': 60})
time.sleep(60)
hits = []
for i in range(1, 5):
    row = request(f'csaon_hcaoff_248k_full_hit_{i}', prompt_248, stream=False)
    row.update({'item': 'csaon_hcaoff_248k_4hit', 'prompt_chars': len(prompt_248)})
    write(row)
    hits.append(row['elapsed_s'])
    if i != 4:
        write({'event': 'sleep_before_hit', 'item': 'csaon_hcaoff_248k_4hit', 'hit_next': i + 1, 'seconds': 60})
        time.sleep(60)
write({'event': 'summary', 'item': 'csaon_hcaoff_248k_4hit', 'hit_elapsed_s': hits, 'hit_min_s': min(hits), 'hit_mean_s': statistics.mean(hits), 'hit_max_s': max(hits)})

for target, reps in [('44K', 3150), ('248K', 17745)]:
    prompt = f'CSAON_HCAOFF_CURVE_{target}_20260602_UNIQUE_PREFIX. ' + SENT * reps + '\nFinal question: answer with one token.\n'
    write({'event': 'item_start', 'item': 'csaon_hcaoff_stream_curve', 'target': target, 'prompt_chars': len(prompt)})
    row = request(f'csaon_hcaoff_curve_{target}_cold_store', prompt, stream=True)
    row.update({'item': 'csaon_hcaoff_stream_curve', 'target': target, 'prompt_chars': len(prompt)})
    write(row)
    write({'event': 'sleep_before_hit', 'item': 'csaon_hcaoff_stream_curve', 'target': target, 'seconds': 60})
    time.sleep(60)
    vals = []
    for i in range(1, 3):
        row = request(f'csaon_hcaoff_curve_{target}_hit_{i}', prompt, stream=True)
        row.update({'item': 'csaon_hcaoff_stream_curve', 'target': target, 'prompt_chars': len(prompt)})
        write(row)
        vals.append(row['ttft_s'])
    write({'event': 'summary', 'item': 'csaon_hcaoff_stream_curve', 'target': target, 'hit_ttft_s': vals, 'hit_mean_s': statistics.mean(vals)})

write({'event': 'done', 'out': str(OUT)})
