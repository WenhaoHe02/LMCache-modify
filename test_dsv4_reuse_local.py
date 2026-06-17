import json, time, requests, statistics

BASE_URL = "http://127.0.0.1:8000"
MODEL = "dsv4"
TIMEOUT = 120
CTX_LENGTHS = [256, 512, 1024, 2048, 3072, 3584]


def make_prompt(n, tag):
    hdr = "[bench:" + tag + "] Summarize:\n<text>\n"
    ftr = "\n</text>\nAnswer:"
    filler = "Bench filler line. " * 200
    body = (filler * 5)[:max(1, n * 4 - len(hdr) - len(ftr))]
    return hdr + body + ftr


def ttft(prompt):
    t0 = time.perf_counter()
    with requests.post(
        BASE_URL + "/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        },
        stream=True,
        timeout=TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line[6:] != "[DONE]":
                json.loads(line[6:])
                return time.perf_counter() - t0
    raise RuntimeError("no data received")


results = []
print(
    "{:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>10}  {:>8}".format(
        "ctx", "populate", "reuse_1", "reuse_2", "reuse_3", "avg_reuse", "speedup"
    )
)
print("-" * 80)

for ctx in CTX_LENGTHS:
    tag = "dsv4-{}-{}".format(ctx, int(time.time()))
    p = make_prompt(ctx, tag)
    t_pop = ttft(p)
    time.sleep(0.5)
    tr = []
    for _ in range(3):
        tr.append(ttft(p))
        time.sleep(0.5)
    avg = statistics.mean(tr)
    spd = t_pop / avg
    results.append(
        {"ctx": ctx, "populate": t_pop, "reuse": tr, "avg_reuse": avg, "speedup": spd}
    )
    print(
        "{:>6}  {:>10.3f}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>10.3f}  {:>8.2f}x".format(
            ctx, t_pop, tr[0], tr[1], tr[2], avg, spd
        )
    )

with open("/root/dsv4_bench.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /root/dsv4_bench.json")
