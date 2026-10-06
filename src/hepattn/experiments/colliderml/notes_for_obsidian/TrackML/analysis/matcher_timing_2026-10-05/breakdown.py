# Inside one matcher call (late-training costs): GPU prep, the one copy, the 4 solves (each and slowest), pool overhead.
import glob, sys, time
import numpy as np
import torch
from hepattn.models import matcher as m


def solve_timed(args):
    import os
    t = time.time()
    out = m._mp_match_task_persistent(args)
    return out, time.time() - t, t, os.getpid()


if __name__ == "__main__":
    assert "hepattn-matcher" in m.__file__
    for trim in ((True, False, True, False)):
        rows, shared, late_start = [], [], []
        for f in sorted(glob.glob(sys.argv[1] + "/costs_*.pt")):
            d = torch.load(f)
            c, ov, qv = d["costs"].cuda(), d["object_valid"].cuda(), d["query_valid"].cuda()
            for rep in range(4):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                lengths = ov.bool().sum(1).cpu().numpy().astype(np.int32)
                x = c.detach().float().masked_fill(~qv.bool().unsqueeze(-1), float(np.finfo(np.float32).max / 10))
                width = m._valid_query_width(qv.bool(), lengths, x.shape[1]) if trim else x.shape[1]
                xt = x[:, :width].transpose(1, 2)[:, : int(lengths.max())].contiguous()
                torch.cuda.synchronize(); t1 = time.perf_counter()
                shm = m._get_shm_buffer(xt.numel() * 4)
                torch.from_numpy(np.ndarray(tuple(xt.shape), dtype=np.float32, buffer=shm.buf)).copy_(xt)
                t2 = time.perf_counter(); w2 = time.time()
                tasks = [("lap1015_late", shm.name, tuple(xt.shape), i, int(lengths[i]), x.shape[1]) for i in range(len(xt))]
                res = m._get_process_pool(4).map(solve_timed, tasks, chunksize=1)
                t3 = time.perf_counter()
                solves = [r[1] for r in res]
                starts = [r[2] - w2 for r in res]; pids = [r[3] for r in res]
                if rep:
                    shared.append(len(set(pids)) < len(pids)); late_start.append(max(starts))  # skip the first (pool / buffer warm-up)
                    rows.append((t1 - t0, t2 - t1, max(solves), np.mean(solves), (t3 - t2) - max(solves), t3 - t0, xt.numel() * 4 / 1e6))
        r = np.array(rows)
        print(f"{'toggle 1+2' if trim else 'toggle 1  '}: total {r[:,5].mean()*1e3:5.0f} ms = GPU prep {r[:,0].mean()*1e3:4.1f} + copy {r[:,1].mean()*1e3:5.1f} ({r[:,6].mean():.0f} MB, {r[:,6].mean()/r[:,1].mean()/1e3:.1f} GB/s) "
              f"+ slowest solve {r[:,2].mean()*1e3:5.1f} (mean of 4: {r[:,3].mean()*1e3:5.1f}) + pool overhead {r[:,4].mean()*1e3:4.1f} ms | calls where 2 solves shared a worker: {100*np.mean(shared):.0f}% | last solve started {1e3*np.mean(late_start):.1f} ms after dispatch")
