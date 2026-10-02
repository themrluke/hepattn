# `test_encoder_or.py` explained

> [!summary]
> This note walks through **`tests/models/test_encoder_or.py`** line by line: what each test
> checks, why it is checked that way, and what every unfamiliar piece of machinery in the file
> actually does. Companion to [[step4-5-plan]] (step 4.11) and [[1. Per-Head Ordering]]. The
> MaskFormer-level tests that feed this one are explained in [[test_maskformer_or_coords]].

**File:** `tests/models/test_encoder_or.py` — three tests, ~200 lines.
**Run it:** from the repo root,

```bash
.pixi/envs/default/bin/python -m pytest tests/models/test_encoder_or.py -q
```

Expect `3 passed` in ~26 s on an A100, ~40 s on CPU. Almost all of that is one-off compilation,
not the tests themselves.

---

## 1. Why this file exists at all

Before these tests, everything we knew about the Route B encoder wiring was: *it runs, the output
is the right shape, and there are no NaNs.*

**That is a very weak statement.** A completely wrong mask would satisfy all three. If every head
attended to every token, or if head 3 were using head 5's ordering, the output would still be
`(1, 256, 32)` and still be full of perfectly finite numbers. Nothing would look wrong. The model
would just train slightly worse than it should, for reasons nobody would ever trace back here.

So the job of this file is to check the **numbers**, not the shapes. Each test does that by
constructing a situation where we independently know what the right answer is.

---

## 2. Vocabulary

Everything in the file that is not plain Python. Skip what you already know.

| Term | What it means here |
|---|---|
| **pytest** | The test runner. It imports the file, runs every function whose name starts with `test_`, and reports a pass unless an `assert` fails or an exception escapes. Functions *not* starting with `test_` (our helpers and stubs) are never run directly. |
| **`assert X, "message"`** | Plain Python. If `X` is false, raise `AssertionError` with that message. pytest prints the message, so it is worth writing a useful one. |
| **stub** (or *double*, *fake*) | A stand-in object that replaces a real one during a test, so the test can control what it returns. Ours replace the random ordering grid with a hand-chosen ordering. |
| **monkeypatching** | Replacing an attribute on a live object or module at runtime. We use it to swap in a stub without changing any production code. |
| **rank table** | For one ordering, a list saying *"token `t` sits at position `p`"*. See [[1. Per-Head Ordering]]. Shape `(N,)` per ordering; the grid produces one per `(head, hash)` cell. |
| **ordering grid** | The `nn.Module` holding one frozen ordering recipe per `(head, hash)` pair, for one layer. Called `E2LSHOrderingGrid`. |
| **`BlockMask`** | The object `flex_attention` uses to decide which 128x128 tiles of the score matrix to compute. Built by `create_block_mask`. |
| **`nn.Module`** | PyTorch's base class for anything that lives inside a model. Needed because you cannot put a bare function into a `ModuleList`. |
| **buffer** | A tensor stored on a Module that is *not* trained but *is* saved in the checkpoint and moved by `.cuda()`. Registered with `register_buffer`. |
| **`state_dict`** | A dictionary of every parameter and buffer in a model, keyed by name. How weights are saved, loaded and copied. |
| **forward hook** | A callback PyTorch runs whenever a Module is called. A *pre*-hook runs just **before**, and can see the module's inputs. |
| **`head_dim`** | `dim // num_heads`. With `DIM=32` and 8 heads, each head works in 4 dimensions. |

---

## 3. The shared setup

```python
import torch
from torch import Tensor, nn

from hepattn.models import encoder as encoder_module
from hepattn.models.encoder import Encoder

DIM = 32
SEQ_LEN = 256
WINDOW = 64
```

Two different imports of the same file, on purpose:

* `Encoder` is the class, so we can build encoders.
* `encoder_module` is the **module object**, so test 3 can reach in and replace a function inside
  it. More on that in §6.

**Why the three constants are shared by all three tests.** The first time PyTorch runs
`flex_attention` for a given set of tensor shapes it *compiles* a GPU kernel for them, which takes
seconds. Every new combination of shapes triggers another compile. Keeping `SEQ_LEN`, `DIM` and
`WINDOW` identical across the tests means they reuse the same compiled kernels and the file stays
under a minute instead of several.

`SEQ_LEN = 256` is deliberately small but is still **2 blocks of 128** — enough that the block
machinery is genuinely exercised. `WINDOW = 64` means each token attends to 32 positions either
side of itself in its ordering.

---

## 4. Test 1 — identity ranks must reproduce the plain sliding window

**Function:** `test_identity_ranks_match_plain_sliding_window`
**This is the load-bearing test.** If only one of the three survived, keep this one.

### The idea

Route B's rule is: *two tokens may attend if their positions **in this head's ordering** are within
half a window of each other.*

```
admit (q, kv)   iff   |rank[q] - rank[kv]| <= window // 2
```

The old, plain sliding window's rule is: *two tokens may attend if their **token indices** are
within half a window.*

```
admit (q, kv)   iff   |q - kv| <= window // 2
```

Now force the ordering to be **the identity** — that is, token 0 sits at position 0, token 1 at
position 1, and so on, so `rank[t] == t`. Substituting that into the first rule gives you the
second rule *exactly*. Not approximately: the same inequality, and both use the same
`<= window // 2` convention (check `flex/sliding_window.py` against `flex/per_head_window.py` if
you want to see it).

So with one hash and an identity ordering, **Route B is the plain sliding window**, and the two
code paths must produce the same numbers. Any disagreement is a bug in the wiring.

### The stub

```python
class _IdentityRankGrid(nn.Module):
    def __init__(self, num_heads: int, n_hashes: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_hashes = n_hashes

    def forward(self, coords: Tensor) -> Tensor:
        num_tokens = coords.shape[-2]
        return torch.arange(num_tokens, device=coords.device).expand(self.num_heads, self.n_hashes, num_tokens)
```

The real grid derives its ordering from the hit coordinates using random projections, so we cannot
simply *ask* it for the identity. We replace it instead.

* **`torch.arange(N)`** is `[0, 1, 2, ..., N-1]` — literally the identity rank table.
* **`.expand(num_heads, n_hashes, N)`** repeats it for every cell without copying memory, so every
  head and every hash gets that same identity ordering.
* **It must be an `nn.Module`**, not a function, because `encoder.ordering_grids` is a `ModuleList`
  and a `ModuleList` only accepts Modules.
* **It must return `(num_heads, n_hashes, N)`**, because `Encoder.forward` immediately calls
  `.flatten(0, 1)` on the result to get `(num_heads * n_hashes, N)`. Return the wrong shape and you
  would silently be testing something else.
* `device=coords.device` keeps it on the same device as everything else — a CPU tensor meeting a
  CUDA tensor is a runtime error.

### Building the two encoders

```python
route_b = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=1)
plain   = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW)
```

Identical except `or_n_hashes=1`, which switches Route B on.

### Making the weights match

```python
missing, unexpected = plain.load_state_dict(route_b.state_dict(), strict=False)
assert not missing, f"plain encoder is missing weights: {missing}"
assert all("ordering_grids" in key for key in unexpected), f"unexpected non-grid keys: {unexpected}"
```

Two encoders are useless for comparison unless their weights are identical.

**Why copy rather than just seed both?** `torch.manual_seed(0)` then building both would only work
if the two constructions draw random numbers in exactly the same order. Route B builds extra stuff
(the ordering grids), so it does not. Copying is deterministic and does not rely on that.

**What `strict=False` does, and why the two asserts are essential.** `load_state_dict` normally
fails if the dictionaries do not match exactly. Ours legitimately do not match — Route B carries
extra frozen buffers for the grids — so we relax it. But `strict=False` silently tolerates
*anything*, including a typo that leaves half the model randomly initialised. So we check the two
lists it hands back:

* `missing` — keys `plain` expected but did not receive. Must be **empty**.
* `unexpected` — keys supplied that `plain` has no slot for. Must be **only** the grid buffers.

Without these two lines the test could pass while comparing two differently-weighted models.

**Direction matters.** We load *Route B's* dict into *plain*. The other way round would report the
grid buffers as `missing` and the check would be muddier.

### Swapping in the stub

```python
num_heads = route_b.layers[0].attn.fn.num_heads
route_b.ordering_grids[0] = _IdentityRankGrid(num_heads, n_hashes=1)
```

`layers[0].attn` is a `Residual` wrapper; `.fn` is the `Attention` inside it. We read `num_heads`
off the real thing rather than hard-coding 8, so the test follows any config change.

### Running both

```python
route_b.eval()
plain.eval()
x = torch.randn(1, SEQ_LEN, DIM)
coords = torch.randn(1, SEQ_LEN, 2)

with torch.no_grad():
    got = route_b(x, x_coords=coords)
    expected = plain(x)
```

* **`.eval()`** turns off training-only randomness such as dropout and drop-path. Without it the
  two encoders would disagree for reasons that have nothing to do with the mask.
* **`torch.no_grad()`** skips building the backward graph. Pure speed and memory; no effect on the
  numbers.
* **The coords are random and completely ignored** by the stub, which returns the identity whatever
  it is handed. They are only passed because `Encoder.forward` asserts they are present when the
  feature is on.

### The comparison

```python
torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
```

`assert_close` compares two tensors elementwise and fails with a readable report (worst element,
how far off) rather than a bare `False`.

**Why not demand exact equality?** With one hash, the machinery is *nearly* a no-op — the
replication step is a view, and averaging over a single hash gives that hash weight 1.0 — but it is
still a different sequence of GPU kernels. Floating-point addition is not associative, so the last
bits can differ. `rtol=1e-4` is loose enough to ignore that and far tighter than any real masking
bug, which would move the numbers by whole percent.

> [!success] What passing means
> The mask rule, the head layout, the replication and the merge all compose to the right answer in
> the one case where the right answer is known independently.

---

## 5. Test 2 — each head must get *its own* ordering

**Function:** `test_head_major_layout_pairs_each_head_with_its_own_ranks`

### The gap test 1 cannot cover

The `(head, hash)` grid is flattened into one axis before it reaches attention. We chose
**head-major**:

```
h = head * n_hashes + hash
```

With `num_heads=8, n_hashes=3` that is 24 rows: rows 0,1,2 are head 0's three hashes; rows 3,4,5
are head 1's; and so on. The q/k/v replication in `attention.py` uses the same convention, so the
two line up.

The alternative, **hash-major**, would be `h = hash * num_heads + head` — rows 0..7 are every head's
hash 0, rows 8..15 every head's hash 1.

**Now the problem.** With `n_hashes == 1`:

```
head-major:  h = head * 1 + 0      = head
hash-major:  h = 0 * num_heads + head = head
```

They are the *same mapping*. So if the grid and the replication ever disagreed about layout, test 1
would still pass — and every head would quietly be using another head's ordering. Catching that
needs **two things at once**: `n_hashes >= 2`, and heads that are distinguishable from each other.

### The stub

```python
class _HeadZeroIdentityRankGrid(nn.Module):
    def __init__(self, num_heads: int, n_hashes: int, num_tokens: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(1234)
        rows = [torch.arange(num_tokens).expand(n_hashes, num_tokens)]
        rows += [torch.randperm(num_tokens, generator=generator).expand(n_hashes, num_tokens)
                 for _ in range(num_heads - 1)]
        self.register_buffer("ranks", torch.stack(rows))  # (num_heads, n_hashes, N)
```

* **Head 0 gets the identity in *both* its hashes.** `torch.randperm(N)` is a random shuffle of
  `0..N-1`; every other head gets one.
* **Why give head 0 the identity twice?** Because then its two hashes produce *identical* results,
  so their merge collapses back to that single result. (The merge weights the hashes by how much
  attention weight each gathered; two identical hashes gather identically, so the weights are 0.5
  and 0.5, and `0.5*out + 0.5*out == out`.) That means head 0 must equal the plain sliding window
  **exactly**, not approximately.
* **The shuffle is generated once, in `__init__`, with a fixed seed.** If it were generated afresh
  on every call, an unlucky permutation could land close to the identity and the "these heads must
  differ" check would fail intermittently. A test that fails once a month is worse than no test.
* **`register_buffer`** rather than a plain attribute, so the tensor moves with `.cuda()` and
  behaves like real model state.

**How this catches the bug:** head-major puts head 0's two hashes in rows 0 and 1, and the
replication puts head 0's q/k/v in those same two rows — they match. Under hash-major, row 1 would
hold *head 1's shuffled ordering* while still receiving head 0's q/k/v, so head 0 would stop
matching the plain window and the first assertion fires.

### Getting at per-head outputs

Here is the awkward part. By the time the encoder returns, the heads have been glued back together
and passed through a linear layer (`out_proj`), which mixes them. You cannot recover head 0's
output from the encoder's return value.

But one step earlier, they are still separate:

```python
def _capture_per_head_output(encoder: Encoder) -> tuple[list, object]:
    captured: list[Tensor] = []

    def pre_hook(module, args):
        captured.append(args[0].detach())

    handle = encoder.layers[0].attn.fn.out_proj.register_forward_pre_hook(pre_hook)
    return captured, handle
```

* **A forward *pre*-hook** runs immediately **before** a Module is called, and receives its inputs
  as `args`. So `args[0]` is exactly what is about to be fed into `out_proj`.
* **Why that tensor is still head-separated:** the step before `out_proj` is `recombine_heads`,
  which is only `transpose(-3, -2)` followed by `flatten(-2)`. No mixing — just a reshape. So head
  `h` occupies columns `[h*head_dim : (h+1)*head_dim]`, which is what `_head_slice` does:

  ```python
  def _head_slice(packed: Tensor, head: int, head_dim: int) -> Tensor:
      return packed[..., head * head_dim : (head + 1) * head_dim]
  ```

* **`.detach()`** drops the tensor's connection to the autograd graph so nothing is kept alive.
* **The hook returns `None`**, which tells PyTorch "I am only observing, do not change the input."
* **`handle`** is the object that lets you remove the hook again.

```python
try:
    with torch.no_grad():
        route_b(x, x_coords=coords)
        plain(x)
finally:
    got_handle.remove()
    want_handle.remove()
```

**The `finally` is not decoration.** If an assertion failed while hooks were still attached, they
would stay attached for the rest of the pytest session and quietly affect later tests. That is a
genuinely horrible bug to track down. `finally` runs whether or not something goes wrong.

### The two assertions

```python
torch.testing.assert_close(_head_slice(got, 0, head_dim), _head_slice(want, 0, head_dim),
                           rtol=1e-4, atol=1e-5)
```

Head 0 has the identity ordering, so it must match the plain window.

```python
differs = [head for head in range(1, num_heads)
           if not torch.allclose(_head_slice(got, head, head_dim), _head_slice(want, head, head_dim),
                                 rtol=1e-4, atol=1e-5)]
assert differs, "every head matched the plain window; the shuffled orderings had no effect"
```

**This second one stops the test passing for the wrong reason.** Suppose a bug made the mask admit
everything for every head. Head 0 would then match the plain window by coincidence, the first
assertion would pass, and the test would report success while proving nothing. Requiring that at
least one shuffled head *visibly disagrees* rules that out.

(`torch.allclose` is the same comparison as `assert_close` but returns a bool instead of raising,
which is what we want inside a list comprehension.)

> [!tip] The general lesson
> Whenever a test asserts "these two things are equal", ask what else would make them equal. If a
> broken implementation would also pass, add the complementary assertion.

---

## 6. Test 3 — the smoke test

**Function:** `test_real_grids_build_one_distinct_mask_per_layer`

No stubs this time — the **real** `E2LSHOrderingGrid`, three layers, three hashes. This is the only
test that exercises the actual random-projection machinery end to end.

### Spying on a free function

```python
built = []
real_create_block_mask = encoder_module.create_block_mask

def spy(*args, **kwargs):
    mask = real_create_block_mask(*args, **kwargs)
    built.append(mask)
    return mask

encoder_module.create_block_mask = spy
try:
    with torch.no_grad():
        out = encoder(x, x_sort_value=coords[..., 1], x_coords=coords)
finally:
    encoder_module.create_block_mask = real_create_block_mask
```

We want to see the masks the encoder builds, but `create_block_mask` is a **plain function**, not a
Module, so there is nothing to hang a hook on. Instead we replace the name **in the module's own
namespace** — `Encoder.forward` looks up `create_block_mask` in `hepattn.models.encoder` every time
it is called, so replacing it there is enough. `spy` calls the real function, records the result,
and passes it through unchanged.

Restored in a `finally` for the same reason as the hooks: a patch left in place would follow the
whole test session around.

**Why `x_sort_value` is supplied.** The mask is defined over *token index*, so it only does anything
useful once the sequence has been sorted into geometric order. Without the sort the mask comes out
fully dense at every sequence length we tried, and the feature is a silent no-op. Passing the sort
value tests the configuration that will actually be used. (See the table in [[step4-5-plan]] under
step 4.8.)

### The assertions

```python
assert out.shape == (1, SEQ_LEN, DIM)
assert torch.isfinite(out).all(), "Route B produced non-finite activations"
```

Basic sanity. `torch.isfinite` is false for `NaN` and `±inf`; an empty softmax row would produce
`NaN` here.

```python
assert len(built) == num_layers, f"expected one BlockMask per layer, got {len(built)}"
for mask in built:
    assert mask.shape[1] == num_heads * n_hashes, mask.shape
```

**One mask per layer** — this is the whole point of giving each layer its own grid. If a cached mask
were reused, you would see 1 instead of 3.

**Height `num_heads * n_hashes` = 24** — this is the assertion that catches a regression to
`H=None` in `create_block_mask`, which would silently collapse all 24 cells onto a single shared
mask. That single change would undo the entire feature while leaving every other check green.

```python
ranks = [encoder.ordering_grids[i](coords[0]).flatten(0, 1) for i in range(num_layers)]
assert not any(torch.equal(ranks[0], other) for other in ranks[1:]), (
    "layers produced identical rank tables; the per-layer grids are not independent"
)
```

**Each layer must have drawn its own orderings.**

> [!warning] The trap this assertion originally fell into
> The first version compared the **masks** instead of the ranks, and **failed** — not because
> anything was broken, but because the assertion was asking at the wrong granularity.
>
> A `BlockMask` only records which 128x128 tiles survive. At `SEQ_LEN = 256` there are just 2
> blocks, every mask is fully dense, and three genuinely different orderings are *indistinguishable*
> at that resolution. Measured:
>
> | N | blocks | densities | masks identical? | ranks identical? |
> |---|---|---|---|---|
> | 256 | 2 | 1.000, 1.000, 1.000 | **yes** | no |
> | 1024 | 8 | 0.465, 0.486, 0.458 | no | no |
> | 4096 | 32 | 0.320, 0.324, 0.311 | no | no |
>
> Comparing *densities* is worse still: two different masks can land on the same density by
> coincidence, so that version would have been flaky rather than reliably wrong.
>
> **Assert a property where it actually lives.** Per-layer independence is a property of the rank
> tables; the mask is a lossy summary of them.

---

## 6b. The equivalence test (added when the sorted path landed)

`test_sorted_and_masked_implementations_agree` builds the same encoder twice, once per
`or_impl`, copies the weights across, and asserts the outputs match.

**Why it is the most important test in the file now.** `sorted` is what actually runs; `masked`
is the thing we reasoned about, tested first, and trust. The two admit exactly the same
query/key pairs -- "within `W//2` of each other in this cell's ordering" -- and differ only in
*how* they arrange for that: one permutes the tokens and applies a plain banded window, the other
leaves them still and encodes the ordering in the mask. So they must agree, and if they ever stop
agreeing the default is silently wrong.

Two companions to it:

* `test_sorted_is_the_default_and_keeps_the_shared_mask_cache` -- pins the default, and checks
  that `sorted` populates the encoder's position-only mask cache while `masked` does not. That
  cache is where most of the speedup comes from, so it is worth asserting rather than assuming.
* `test_unknown_or_impl_is_rejected` -- a typo in a config should fail loudly at construction,
  not silently pick a default.

> [!note] One older test had to be repinned
> `test_real_grids_build_one_distinct_mask_per_layer` asserted one BlockMask per layer. That is
> the *masked* path's defining behaviour -- the sorted path shares one mask across every layer and
> cell -- so once the default changed it was testing the wrong implementation. It is now
> `test_masked_impl_builds_one_distinct_mask_per_layer` and passes `or_impl="masked"` explicitly.

---

## 6c. The padding tests (step 5)

Two more, both parametrised over `or_impl` so they cover the sorted default and the masked
reference.

**`test_padding_does_not_change_the_answer_for_real_tokens`** runs the same real hits twice, once
alone and once with 56 padded slots attached, and demands the outputs match.

That framing is the whole trick. The obvious test — "does it run without NaNs with padding?" —
would pass against a badly broken implementation. The real risk is subtler: padded coordinates
feed into two *event-level* statistics inside the ordering (the projection's span, and the
equal-occupancy quantile bin edges), and either one shifts where the bins fall, which reorders
**genuine** hits. Comparing against the unpadded run is the only way to see that.

**`test_padded_queries_keep_a_finite_softmax_row`** uses a mostly-padding worst case (64 real,
192 padded). A padded query has no valid keys, so unless the mask always admits `q == kv` its row
is entirely `-inf`: empty softmax, NaN output, and a NaN `lse` that spreads through the merge into
the real tokens. The padded token's own output is meaningless and gets discarded downstream — the
diagonal exists purely to keep the arithmetic finite.

Both were checked by mutation: contaminating the statistics fails the first on both
implementations, and removing the diagonal escape fails the second.

> [!note] The stubs gained a `valid` argument
> `E2LSHOrderingGrid.forward` now takes `valid`, so both grid stubs accept and ignore it. Worth
> noticing when reading them: a stub has to track its subject's signature or the tests break for
> reasons that have nothing to do with what they are testing.

---

## 7. What the tests prove together

| Claim | Which test |
|---|---|
| The rank window equals the index window when ranks are the identity | 1 |
| The merge is a no-op for a single hash | 1 |
| The merge of two identical hashes returns that hash | 2 |
| Each head's q/k/v meets **its own** ordering, not another head's | 2 |
| The orderings actually change the result | 2 (the `differs` check) |
| The real random grid survives real coordinates | 3 |
| One mask per layer, of height `num_heads * n_hashes` | 3 |
| Layers draw independent orderings | 3 |

### What they do *not* prove

* **Nothing about speed.** `flex_attention` is currently running uncompiled and unfused on this path,
  so any timing taken from these tests is meaningless. See §2c of [[step4-5-plan]].
* **Nothing about padding.** Every sequence here is full; step 5 is where padded tokens are handled.
* **Nothing about real physics.** Coordinates are random Gaussians, not detector hits, so the
  densities you see are nothing like the ~0.24 measured on real events.
* **Nothing about `MaskFormer`.** These tests drive `Encoder` directly. The layer above — getting
  the coordinates down to the encoder still paired with their tokens — is step 4.10, covered
  separately in [[test_maskformer_or_coords]].

---

## 8. Reusable lessons

1. **"It runs and the shape is right" is not a test.** Construct a case where you know the answer.
2. **A degenerate case is often the known answer.** One hash + identity ordering turns a complicated
   new path into an old, trusted one.
3. **Ask what else would make your assertion pass.** If a broken implementation also passes, add the
   complementary check (test 2's `differs`).
4. **Assert properties where they live.** Not through a lossy summary (test 3's ranks-vs-masks).
5. **Always restore what you patch, in a `finally`.** Hooks and monkeypatches outlive the test that
   installed them.
6. **Seed anything random, once.** An occasionally-failing test gets ignored, then deleted.
7. **Never let `strict=False` pass unchecked.** Assert on what it hands back.
