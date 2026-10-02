# `test_maskformer_or_coords.py` explained

> [!summary]
> Walkthrough of **`tests/models/test_maskformer_or_coords.py`**, which covers step 4.10: getting
> the hit coordinates from `MaskFormer` down into `Encoder` **still paired with their tokens**.
> Companion to [[step4-5-plan]] (step 4.10), [[test_encoder_or]] (the encoder-level tests) and
> [[1. Per-Head Ordering]].

**File:** `tests/models/test_maskformer_or_coords.py` — three tests.
**Run it:**

```bash
.pixi/envs/default/bin/python -m pytest tests/models/test_maskformer_or_coords.py -q
```

Expect `3 passed` in ~55 s. Almost all of that is one-off compilation.

---

## 1. The bug this file exists to prevent

`Encoder` turns coordinates into a per-head ordering. That ordering is a statement about
**positions in the sequence**, so it is only meaningful if row `i` of the coordinates describes
the same hit as row `i` of the tokens.

`MaskFormer` is where that can quietly come apart, because there are **two dictionaries** in play:

* **`inputs`** — the raw per-input-type tensors handed to the model. Never reordered.
* **`x`** — the working dictionary, holding the merged sequence. **May be permuted** by a
  `Sorter` before the encoder is called.

So if a `Sorter` is active and you build the coordinates from `inputs`, you get the coordinates in
one order and the tokens in another. Every rank table then describes hits that have moved.

> [!danger] Why this is worth a test rather than care
> Nothing about it is loud. The shapes are right — both are `(1, N, 2)` and `(1, N, dim)`. No
> exception, no NaN. Attention simply operates on geometric nonsense and the model trains slightly
> worse, forever, for a reason nobody would ever trace back to this line.
>
> This is the same class of bug as step 4.6 inside the encoder, one level up.

---

## 2. What the production code does

Two pieces, in `maskformer.py`:

```python
# before the sorter block
if self.encoder.or_n_hashes is not None:
    for field in ("eta", "phi"):
        x[f"key_{field}"] = torch.concatenate([inputs[f"{name}_{field}"] for name in self.input_names], dim=-1)

# ... the existing sorter block, which may or may not run ...

# at the encoder call
x_coords = torch.stack([x["key_eta"], x["key_phi"]], dim=-1) if self.encoder.or_n_hashes is not None else None
```

The coordinates are put into `x` **before** the sorter runs, so they are carried along by whatever
happens next. That single placement covers both situations:

| | who permutes the coordinates | when |
|---|---|---|
| **no sorter** | `Encoder.forward` itself, using `x_sort_value` (step 4.6) | after they arrive |
| **sorter active** | `Sorter.sort_inputs`, along with everything else in `x` | before they arrive |

Two details that are not obvious:

* **They are stored as two 1-D fields (`key_eta`, `key_phi`), not one stacked `(B, N, 2)` tensor.**
  `sort_inputs` gathers with a `(B, N)` index (`sorter.py:54`), and `torch.gather` requires the
  index to have the same number of dimensions as the tensor. A 3-D tensor would raise there.
* **They are stacked into `(B, N, 2)` only at the very end**, with **eta first** —
  `E2LSHOrderingGrid` reads `coords[..., 0]` as eta.

---

## 3. The trick that makes misalignment visible

Here is the problem with testing this. A token embedding is an opaque vector; it says nothing
about which hit produced it. So "are the coordinates still paired with their tokens" is not a
question you can ask of the tensors as they normally exist.

The fix is to make the embedding **carry its own coordinates**:

```python
class _CoordEchoInputNet(nn.Module):
    """Embeds each hit so that channel 0 is its eta and channel 1 is its phi."""

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        eta = inputs[f"{self.input_name}_eta"]
        phi = inputs[f"{self.input_name}_phi"]
        pad = torch.zeros(*eta.shape, self.dim - 2, device=eta.device, dtype=eta.dtype)
        return torch.cat([eta.unsqueeze(-1), phi.unsqueeze(-1), pad], dim=-1)
```

A normal input net is a learned `Linear` that mixes the hit's fields into an embedding. This one
just copies eta and phi into the first two channels and zero-pads the rest. It is not a sensible
model component — it is a **tracer dye**.

Now the property we care about becomes a one-line assertion at the encoder's door:

```python
torch.testing.assert_close(coords[..., 0], x[..., 0])   # eta
torch.testing.assert_close(coords[..., 1], x[..., 1])   # phi
```

Whatever permutation happened on the way — none, the encoder's, or the sorter's — if the
coordinates and the tokens were reordered together, these hold. If either moved without the
other, they fail.

> [!tip] The general technique
> When a property is about *correspondence between two tensors* and neither tensor carries
> identity, put identity into one of them. Coordinates, an index, a serial number — anything that
> survives the transformation and can be checked at the other end.

---

## 4. Catching the arguments at the encoder's door

```python
def pre_hook(module, args, kwargs):
    seen["x"] = args[0].detach()
    seen["x_coords"] = None if kwargs["x_coords"] is None else kwargs["x_coords"].detach()
    seen["x_sort_value"] = kwargs["x_sort_value"]

handle = model.encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
```

A **forward pre-hook** is a callback PyTorch runs immediately *before* a Module's `forward`. That
is exactly the moment we want: after `MaskFormer` has done all its assembling, before the encoder
does anything of its own.

* **`with_kwargs=True`** is required. Without it the hook receives only positional arguments, and
  `x_coords` is passed by keyword, so it would be invisible.
* **`args[0]`** is `x["key_embed"]`, the merged token sequence.
* **`.detach()`** drops the autograd connection so nothing is kept alive.
* **Returning `None`** means "observing only, do not modify the arguments".
* **`handle.remove()`** in a `finally` — a hook left attached would follow the rest of the pytest
  session around and is a genuinely horrible bug to track down.

Capturing `x_sort_value` too lets each test confirm it is exercising the path it thinks it is.

---

## 5. The three tests

### `test_coords_reach_encoder_paired_without_a_sorter`

`input_sort_field="phi"`, no sorter. Asserts:

1. The coordinates arrive and are paired with their tokens.
2. **`x_sort_value` is not `None`** — the encoder was told to sort, so the 4.6 gather is what will
   do the permuting.
3. **The coordinates are *not* already in phi order.** This is the check that proves the test is
   really on this path and not accidentally testing the other one.

### `test_coords_reach_encoder_paired_with_a_sorter`

`Sorter(input_sort_field="phi")`, no `input_sort_field`. Asserts:

1. The coordinates arrive and are paired with their tokens.
2. **`x_sort_value` is `None`** — the sequence is already sorted, so the encoder must not sort it
   again.
3. **The coordinates *are* in phi order** — proof they were carried by `sort_inputs` rather than
   read out of the untouched `inputs`.

Note that assertions 3 in the two tests are **opposites**. That is deliberate: together they pin
down which mechanism ran, so neither test can silently drift onto the other's path.

### `test_coords_are_not_built_when_the_feature_is_off`

Swaps in an ordinary `attn_type="torch"` encoder with no `or_n_hashes`, and asserts `x_coords` is
`None`. Everything in step 4.10 is gated on the feature being on; this checks the gate actually
holds, so no existing config changes behaviour.

---

## 6. Fixture constraints worth knowing

These are not arbitrary; each one is forced by something real.

| Choice | Why |
|---|---|
| **Batch size 1 everywhere** | `attention.py:439` asserts batch size 1 whenever a `BlockMask` is present. Route B always has one. The existing `test_maskformer_sorting.py` uses batch 2, which is why this needed its own fixtures. |
| **`attn_type="flex"` + `window_size`** | Both are required by the `Encoder` guard asserts when `or_n_hashes` is set. |
| **Input names `inputone` / `inputtwo`** | `MaskFormer.__init__` asserts input names contain no underscores. |
| **phi values disjoint between the two inputs** (one 0-9, the other 100-114) | If the two input types' hits were ever mixed up with each other, the phi values would interleave visibly instead of looking plausible. |
| **phi deliberately unsorted** (`torch.randperm`) | If the input were already sorted, "did the sorter reorder this" would be unanswerable. |
| **A `_CoordEchoInputNet` rather than the existing `MockInputNet`** | The one in `test_maskformer_sorting.py` builds its input by taking every field that is not `valid` or `phi` and feeding them to a `Linear(2, dim)`. Adding `eta` fields would give it three inputs and a shape error. |

---

## 7. Proving the tests can fail

A test that cannot fail is worse than no test, because it produces false confidence. This one was
checked by **mutation**: the production line was temporarily changed back to the buggy version —
building the coordinates from the unsorted `inputs` dict — and the suite re-run:

```
E       AssertionError: eta is not paired with its token
1 failed, 2 passed
```

The sorter test failed, the other two passed (correctly — they do not exercise the sorter path).
The mutation was then reverted and all tests re-confirmed green.

> [!tip] Worth doing routinely
> After writing a test for a specific bug, reintroduce the bug and watch the test go red. It takes
> a minute and is the only direct evidence that the test does what you think.

---

## 8. What these tests do *not* cover

* **Nothing downstream of the encoder.** Once the coordinates are correctly paired at the encoder's
  door, [[test_encoder_or]] covers what happens to them inside.
* **Nothing about real data.** Coordinates are random; field names (`eta`, `phi`) are hard-coded in
  the production code, so a dataset using different names would fail at a `KeyError`, not here.
* **Nothing about padding.** All hits are valid. Step 5.
* **Nothing about batch size > 1**, which Route B does not currently support.
