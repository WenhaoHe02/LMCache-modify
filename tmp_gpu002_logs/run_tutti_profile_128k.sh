cat >/dev/shm/tutti_profile_128k.py <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_PROFILE_128K_20260610_UNIQUE_PREFIX"
payload = {
    "model": "deepseek-v4-pro",
    "prompt": prefix + "\n" + ("0123456789abcdef " * 20000),
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}

def post(label):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read())
    print(label, round(time.time() - start, 3), obj.get("usage", {}), flush=True)

post("cold")
time.sleep(35)
post("hit1_lazy")
time.sleep(10)
post("hit2_steady")
PY
sudo docker cp /dev/shm/tutti_profile_128k.py dsv4-256k-measure-tutti:/tmp/tutti_profile_128k.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/tutti_profile_128k.py | tee /dev/shm/tutti_profile_128k_client.txt
sudo docker logs --since 25m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "TUTTI_PROFILE|LMCACHE_RETRIEVE_PROFILE|Reqid:|Retrieved [0-9]" \
  > /dev/shm/tutti_profile_128k.log || true
python3 - <<'PY'
from pathlib import Path
import re, statistics
log=Path('/dev/shm/tutti_profile_128k.log').read_text(errors='ignore').splitlines()
print('client:')
print(Path('/dev/shm/tutti_profile_128k_client.txt').read_text(errors='ignore'))
# Print hit lines around retrieve profiles compactly
retrieve=[]; batched=[]; ensure=[]; create=[]; batch_detail=[]
for line in log:
    if 'LMCACHE_RETRIEVE_PROFILE' in line:
        retrieve.append(line)
    elif 'TUTTI_PROFILE batched_get' in line:
        batched.append(line)
    elif 'TUTTI_PROFILE ensure_loader' in line:
        ensure.append(line)
    elif 'TUTTI_PROFILE create' in line:
        create.append(line)
    elif 'TUTTI_PROFILE batch_detail' in line:
        batch_detail.append(line)
print('counts retrieve',len(retrieve),'batched',len(batched),'ensure',len(ensure),'create',len(create),'batch_detail',len(batch_detail))

def val(line, key):
    m=re.search(rf'{key}=([0-9.]+)', line)
    return float(m.group(1)) if m else None

def summarize(name, lines, keys):
    print('\n'+name)
    for k in keys:
        xs=[val(l,k) for l in lines]
        xs=[x for x in xs if x is not None]
        if xs:
            print(k, 'n',len(xs),'min',round(min(xs),3),'mean',round(statistics.mean(xs),3),'max',round(max(xs),3))
# last 8 retrieve/batched should be hit2 steady; first 8 with ensure are hit1 lazy usually
summarize('batched_all', batched, ['size_mb','metadata_ms','load_hbm_ms','total_ms'])
summarize('batched_last8', batched[-8:], ['size_mb','metadata_ms','load_hbm_ms','total_ms'])
summarize('retrieve_all', retrieve, ['size_mb','process_tokens_ms','broadcast_ms','to_gpu_ms','cleanup_ms','total_ms'])
summarize('retrieve_last8', retrieve[-8:], ['size_mb','process_tokens_ms','broadcast_ms','to_gpu_ms','cleanup_ms','total_ms'])
summarize('ensure', ensure, ['recover_ms','collect_paths_ms','fiemap_ms','unmount_ms','create_ms','total_ms'])
summarize('create', create, ['cuda_malloc_ms','session_bind_map_ms','aux_alloc_ms','total_ms'])
summarize('batch_detail_last80', batch_detail[-80:], ['bytes_mb','build_ms','extents_ms','arg_ms','submit_launch_ms','poll_sync_ms','status_ms','wrap_ms','total_ms'])
print('\nlast retrieve lines')
for l in retrieve[-8:]: print(l)
print('\nlast batched lines')
for l in batched[-8:]: print(l)
PY
