# Speaker script — *Per-head orderings and OR amplification*

Read alongside `OR_amplification.pptx`. Slide numbers match the deck exactly (30 slides; the deck also prints the number bottom-right).

**Runtime:** ~30 min at a comfortable pace, ~20 min if you skip §6 (correctness).
**Shape of the argument:** one ordering is never enough → so use several → here is how you combine
them exactly → here are two ways to implement that → one is 11x cheaper → here is what we
measured → here is what we still do not know.

> **The one thing to land:** the physics idea (several orderings, merged exactly) and the
> engineering result (11x, from where the ordering lives) are *separate* findings. Do not let the
> second bury the first.

---

## 1 — Title

> "This is about a change to how attention decides which hits are allowed to talk to each other.
> There is a physics motivation and an engineering result, and I want to keep them apart, because
> the engineering one is more dramatic and it would be easy to leave thinking that was the point."

Set expectations: **no physics result yet.** You are presenting a mechanism, a correctness
argument and a cost. Say so now and nobody feels misled at slide 27.

---

## 2 — The problem in one slide

Walk the four lines slowly; everything later depends on them.

> "A pixel event is about a hundred thousand hits. Attention is quadratic, so that is ten to the
> ten operations per layer — not affordable. The standard fix is to let each hit attend only to a
> window of around five hundred neighbours, which is a two-hundred-fold saving.
>
> The catch is that attention doesn't see the detector. It sees a **list**. And 'nearby in the
> list' is only 'nearby in the detector' if somebody arranged the list that way."

Then the punchline:

> "Arranging it is called ordering, or serialization. And the fact this whole talk hangs on: **no
> single ordering can keep every track intact.**"

---

## 3 — Why we cannot afford full attention

Quick slide. Point at the two curves, then the arrow.

> "Quadratic versus linear-in-N. At our event size that gap is about two hundred times. Local
> attention isn't a tuning choice — without it this model does not run."

**If asked "why 512?"** — it is what the encoder already uses; it is a trade between how many
genuine neighbours you capture and cost. Not something this work changes.

---

## 4 — Serialization

The left panel is the detector, the right is what attention actually sees.

> "Same event twice. On the left, real geometry — one track highlighted, five hits sitting close
> together in eta-phi. On the right, the list. Unsorted, those five hits span fifty-one positions:
> a window of five would never see them together. Sorted by phi, they span seven. Same hits, same
> window, completely different outcome.
>
> That is all sorting does, and it is why it matters."

---

## 5 — The catch: every ordering cuts somewhere

The slide that motivates everything after it. Take your time.

> "Here is the problem with any single ordering. Top row: this track's hits get split — the window
> around the first one doesn't reach the rest. Bottom row: a different ordering keeps the same
> track together.
>
> This is not a bad ordering that we could fix by being cleverer. You cannot flatten two
> dimensions onto one without cutting somewhere. Every ordering cuts — they just cut in
> **different places**."

Then the pivot:

> "Which immediately suggests the fix: don't use one. Use several, and let a hit count as a
> neighbour if **any** of them puts it nearby. That is OR amplification."

---

## 6 — Section: background

> "Before the mechanism, one design question that turns out to be entangled with it."

---

## 7 — Two ways to define the window

> "Two flavours of local attention. Sliding window: every hit sits at the centre of its own
> window, the windows overlap, there are no boundaries. Block-sparse: chop the list into fixed
> blocks, everyone inside a block attends to everyone else, nobody attends outside.
>
> Look at the red lines. Hits either side of a block edge are adjacent in the list and **cannot
> see each other at all**. That is a seam. A track straddling one loses half of itself at that
> layer."

Worth noting: **HEPTv2 is block-sparse**, and we confirmed that in their code rather than
inferring it from the paper — there is no mask argument anywhere in their encoder.

---

## 8 — "BlockMask" is not "block-sparse"

A vocabulary slide. Include it — someone will look at the code, see `BlockMask`, and conclude we
went block-sparse.

> "Quick disambiguation, because these two sound identical and are not.
>
> **BlockMask** is a PyTorch data structure. It records, for each 128-by-128 tile of the score
> matrix, whether that tile contains any admitted pair. It is bookkeeping for deciding what to
> skip, and *any* mask rule can be stored in one.
>
> **Block-sparse** is an attention pattern: chop the sequence into fixed disjoint blocks, attend
> densely inside each, never across.
>
> We hand flex a sliding-window rule and it stores the resulting sparsity in a BlockMask. So we
> use the structure without using the pattern. Look at the two pictures — the orange grid is the
> same in both, that is the BlockMask tiling. Only the right-hand pattern is block-sparse."

**If asked which our model uses** — sliding window, in both implementations. I verified it by
evaluating the mask rules directly: the admitted set is exactly `|Δ| ≤ W/2`, adjacent queries'
windows overlap, and edge rows are truncated. All three are sliding-window signatures and none
are block-sparse ones.

---

## 9 — Which for us?

The interesting slide for a research audience, because the answer isn't "ours is better".

> "You would think seams are simply bad. But look at what blocks buy: gather, reshape, run dense
> attention, no mask at all. Perfect kernel utilisation. Sliding windows overlap, so they don't
> tile, so you cannot do that reshape — you need a real mask."

Then the insight, which is the bit worth them remembering:

> "Block-sparse is not better or worse in the abstract. It becomes **safe** once you have OR
> amplification, because a seam in one ordering is deep inside another, and the merge repairs it.
> HEPTv2's two choices are a matched pair — they use blocks *because* they built the
> multi-ordering that makes blocks tolerable.
>
> We kept the sliding window: it is what our encoder and benchmarks are already built around, and
> with no seams the whole OR budget goes into genuine recall rather than patching boundaries we
> introduced ourselves."

**If asked "shouldn't we switch to blocks then?"** — it is a principled fallback, not a
compromise, and it would give an apples-to-apples number against HEPTv2's block-size ablation.
It is on the list; it was not the default because it changes the model, not just the kernel.

---

## 10 — Section: the mechanism

---

## 11 — The LSE merge

The one piece of real mathematics. Lead with why the naive thing fails.

> "Each ordering gives the hit its own answer. The obvious move is to average them. That is wrong:
> a neighbour found by two orderings should count for more than one found by a single ordering,
> and averaging throws that away.
>
> Flex attention hands back a number called the log-sum-exp for free — the log of the softmax
> denominator. It is the total attention mass that neighbourhood gathered before normalisation.
> Weight each ordering's output by softmax of its lse, sum, and you get **exactly** what you would
> have got from one single attention over the union of all three neighbourhoods."

Stress **exactly**:

> "Not approximately. This is the standard exact method for combining attention computed over
> different subsets of keys. There is no approximation being introduced here."

**The dinner-party analogy, if the room wants one:** you attend three tables in an evening, and
afterwards fuse what you heard, weighting each table by how much genuinely relevant conversation
happened there. The result is as if everyone had sat at one table with you.

---

## 12 — Two axes that are easy to confuse

A correction slide. Say it was a mistake we made early, because it makes the point stick.

> "We initially thought ordinary multi-head recombination would give us the OR for free — different
> heads see different orderings, they get concatenated, job done. That is **false**, and it is
> worth being explicit because it is a natural thing to assume.
>
> There are two axes. The hash axis is merged by log-sum-exp *inside* each head, and that is OR
> amplification. The head axis is concatenated and projected, and that is ordinary multi-head
> capacity. Different mechanisms, different purposes."

Then close the door on the other tempting shortcut:

> "And you cannot do this with a cleverer global sort either. A list is one line — each hit has
> exactly one position and therefore exactly one set of neighbours. You physically cannot have hit
> A next to B in one arrangement and next to C in another *within the same list*. That is why the
> hash axis has to exist, and why this lives inside the attention layer."

---

## 13 — How one ordering is actually built

The mechanics behind everything so far. Worth the time — until now orderings have been a black box.

> "So what *is* one of these orderings? Three steps.
>
> **One**: chop the detector into bins in eta and phi. Not equal-width bins — equal
> *occupancy*. Every bin holds the same number of hits, so the edges sit wherever the hits happen
> to be dense. That is the grid on the left.
>
> **Two**: draw a random direction and project every hit onto it. That gives each hit a position
> along that line, which is the colour gradient in the middle panel.
>
> **Three**: combine them. Sort by bin first, and within a bin by the projection."

Then the part that makes it click:

> "The arithmetic makes that ordering strict. The bin terms move in steps of `D`, where `D` is the
> full span of the projection — and the projection can only vary *by* `D` in total. So the
> projection can never lift a hit out of its bin's slot. Bins give the coarse structure, the
> projection breaks ties inside them. You can see it on the right: contiguous runs of colour per
> bin, and the projection ramp restarting inside each one."

**If asked why a random direction rather than just eta or phi** — a fixed direction always makes
the same mistakes; hits separated perpendicular to it always look adjacent. Random directions per
cell means each cell makes *different* mistakes, which is the entire point.

---

## 14 — Why two cells disagree

> "Here is the same event through two different cells. Different number of bins, so different
> edges; different projection direction. Follow the six orange hits — physically the same six hits
> in both rows — and they end up spanning very different stretches of the sequence.
>
> That disagreement is not a defect to be tuned away. It is the raw material. If all the orderings
> agreed there would be nothing for the OR merge to recover."

---

## 15 — Where the diversity comes from

> "Every square is an independent ordering: four layers, eight heads, three hashes — ninety-six in
> the default config. The merge only unions the three hashes inside one head of one layer, but
> depth helps on top: a track cut at one layer meets a completely different partition at the next."

The subtlety worth spelling out:

> "These are **frozen** — drawn once when the model is built, never retrained, never resampled.
> But frozen means the *rule* is fixed, not the order. Think 'sort guests by surname': the rule
> never changes, but every party has a different guest list. Same event always serialises the same
> way, which is what makes inference reproducible."

**If asked about the downside:** if two hits on the same track are separated by all ninety-six
orderings for a given event, they can never attend — in any epoch, at inference, ever. That blind
spot is baked in. Resampling every batch would avoid it, but you would lose reproducibility and
the model would be chasing a target that reshuffles under it.

---

## 16 — Section: implementation

> "Everything so far is the idea. The rest is how you build it — and this is where it got
> interesting."

---

## 17 — masked vs sorted

The central engineering slide. Go slowly, the diagram carries it.

> "Two ways to implement exactly the same thing.
>
> **Masked**: leave the hits where they are, and hand each head a lookup table — 'in my ordering,
> hit A sits third'. The rule becomes 'are you within two places of me *in my table*'. Every head
> has a different table, so every head needs its own mask.
>
> **Sorted**: physically rearrange the hits into that head's order. Now position in memory *is*
> position in the ordering, so the rule becomes 'are you within two slots of me'. That sentence
> mentions no ordering at all — so it is the **same rule for every head**, and they can all share
> one mask."

Then the key framing:

> "Both admit exactly the same pairs and produce identical numbers. Sorted pays a shuffle to make
> the question simpler. We have a test that asserts they agree, and that test is what let us
> change the default safely."

---

## 18 — Background: how tiled attention actually runs

Context slide. Some of the room will not know why "tiles" are the unit at all — spend a minute
here or slide 16 lands as an assertion.

> "A word on how these kernels work, because the next slide depends on it.
>
> FlashAttention — and flex, which is the same family — never builds the full N-by-N score
> matrix. It cannot: at thirteen thousand hits that is 170 million numbers per head. Instead it
> walks the matrix in **tiles**, typically 128 by 128, sized so a tile fits in the GPU's on-chip
> memory. It loads a block of queries and a block of keys, does one dense matmul, and moves on."

Then the part that matters:

> "And here is the bit people find surprising: **the mask is applied after that matmul**. The
> scores for the whole tile have already been computed. Masking sets the rejected ones to minus
> infinity so the softmax ignores them — but the arithmetic has already happened. Masking
> discards, it does not save.
>
> So a tile is all-or-nothing. If the mask says *nothing* in this tile is admitted, the kernel
> skips it before doing any work — that is where the saving comes from. But if even one pair in
> the corner is admitted, the whole 16,384-entry tile is computed."

**If asked "why not skip per element?"** — the tensor cores process a tile in lockstep; there is
no per-lane branch. And a predicate per element would cost about as much as the multiply it was
trying to avoid.

---

## 19 — Why that matters: tiles

Now the consequence, with the two pictures.

> "Both panels admit exactly the same number of pairs. The only difference is **where** they sit
> relative to the tile grid.
>
> On the left the pairs hug the diagonal, so whole rows of tiles are empty and get skipped. On the
> right — same pairs, different ordering — they are spread out, so most tiles contain at least one
> and every one of those is computed in full."

Be explicit that the figure is schematic:

> "These percentages are for a 96-token toy so you can see the tiles. The ratio is what transfers,
> not the numbers. On real events it is 4.8% of tiles against 23.2%."

---

## 20 — How much of that computed work is wasted

The stat slide. This is the cleanest way to express "how bad is it".

> "Here is the same thing as an efficiency number. Of the whole score matrix, the window genuinely
> wants **3.9%** of the pairs — and that figure is identical for every ordering, because a rank
> table is just a permutation, so the number of admitted pairs never changes.
>
> The sorted implementation computes 4.8%, so it does about **1.2x** more work than strictly
> necessary. That is close to optimal — it is the unavoidable price of deciding in 128-wide chunks
> rather than per pair.
>
> The masked implementation computes 23.2%, which is **5.9x** more than necessary. Same pairs
> wanted, five times the work done to get them."

---

## 21 — Where it plugs in

> "For orientation: the dashed box is the new part. The global sort, the projections, the decoder,
> the losses — untouched. The rest of the model never learns that anything happened."

Flag the subtle bit:

> "The one genuinely dangerous line is permuting the coordinates alongside the tokens. Get it
> wrong and nothing crashes: right shapes, no NaNs, the model just trains on geometric nonsense
> forever."

---

## 22 — Section: results

---

## 23 — Step 4a, the go/no-go

Frame it as a question that could have killed the project.

> "Before building any of it we had to answer one question: is the per-head mask actually sparse?
> The worry was real — twenty-four different orderings over one physical layout could easily make
> every tile non-empty, and then you have paid for a mask that saves nothing.
>
> It came out at 0.24, not 1.0. The reason is that LSH orderings are not random permutations —
> they all sort the *same* geometry with different random grids, so they stay correlated."

The second finding:

> "And how you lay the sequence out matters as much as the orderings do — half a factor of two
> between raw and LSH-sorted. That was free."

---

## 24 — Where the time goes

> "This is the slide that decided the design, and it surprised us. Look at what dominates: not
> attention, **mask construction**.
>
> The masked implementation has to rebuild a twenty-four-row mask every layer of every event,
> because its rule depends on this event's ordering tables. Sorted's rule — 'within 256 slots' —
> depends only on sequence length, so it is built once and reused."

---

## 25 — The headline numbers

> "Eleven times cheaper, for bit-for-bit identical output. That is the whole case for the sorted
> implementation."

Then be careful with the comparison to today:

> "Against today's sliding window: the encoder is 5.6x more expensive. But per **training step** it
> is only 1.44x, because attention is a smaller share of the full model than the isolated benchmark
> suggests. If someone quotes 5.6x as the cost of this feature, that is the encoder only."

---

## 26 — Block size cannot rescue it

Quick.

> "Obvious question: could smaller tiles fix the masked version? No. The ratio sits at four to five
> times at every block size, because it is a property of how scattered the ranks are, not of the
> tiling."

---

## 27 — The methodology warning

Include this. It is the most transferable slide in the deck.

> "Flex attention has to be compiled. Uncompiled it silently falls back to materialising the whole
> N-by-N score matrix — 8.5 gigabytes where the fused version uses 0.1.
>
> And here is why I am showing you this rather than just fixing it quietly. Before we noticed,
> attention was taking 271 milliseconds a layer, and against *that* mask-building looked like under
> four percent of the cost. We concluded mask construction didn't matter. Fixing the compilation
> made attention twenty-seven times faster and the **same** mask cost became ninety-two percent.
>
> The measurement never changed. The thing it was divided by did — and it inverted the conclusion."

---

## 28 — Section: correctness

Optional if short on time; say "happy to go through how we tested this offline".

---

## 29 — Testing strategy

> "The pattern throughout was: construct a case where you already know the answer.
>
> One hash with an identity ordering *must* reproduce the plain sliding window, because the rank
> window and the index window become the same inequality. That collapses the new path onto an old
> trusted one.
>
> The second test exists because the first structurally cannot catch a head-ordering mix-up — with
> one hash, the two possible layouts are the same mapping. You need two hashes and heads you can
> tell apart."

The habit worth advertising:

> "Every one of these was checked by putting the bug back and confirming the test goes red. A test
> you have never seen fail is not evidence."

---

## 30 — Bugs found

> "Five bugs, and the thing they have in common is that **none of them raise**."

Pick two to tell properly — the fp16 one lands best with an ML audience:

> "Flex returns the lse in float32 whatever you give it. Multiply a half-precision output by that
> and it promotes, and the next layer's half-precision weights refuse. Fifteen passing tests never
> saw it, because they all run in float32. Training runs in half — so this would have crashed on
> the very first real step.
>
> It was found by running the thing at realistic size and precision, not by testing. That is worth
> internalising: the test suite told us nothing here."

---

## 31 — Section: status

---

## 32 — Training comparison

Be straight about this; it is the honest core of the talk.

> "Two runs going now, one per arm, same seed. And the result so far is: they are
> indistinguishable. Which means this comparison currently tells us **nothing** about OR
> amplification.
>
> The metrics are also poor in absolute terms — and they are poor for *both*, so it is not our
> feature. The baseline config is pixel-barrel only, eta below one, which leaves about four and a
> half hits per particle. Matching at fifty percent means getting two of four right; perfect-match
> efficiency is zero everywhere. That config is called 'example' and I do not think anyone has
> tuned it for physics."

**If asked "so is the feature useless?"** — we do not know. That is exactly the point. What we
know is that it is correct and what it costs. Whether it helps needs a baseline that reconstructs
tracks in the first place.

---

## 33 — Status and next

> "Built, tested, benchmarked, audited. The sorted implementation is the default; masked stays in
> as the reference we check against.
>
> What is left is not engineering. It is: fix the baseline config so the comparison can mean
> something, then run the A/B properly — ideally two seeds per arm, because with one seed you
> cannot separate a real effect from noise."

Close on the honest note:

> "Nothing measured so far shows this feature *helps*. It shows it is correct, and what it costs.
> The interesting experiment hasn't happened yet."

---

# Questions you should expect

**"Why not just use a bigger window?"**
Cost is linear in window size, and it does not fix the problem — a bigger window still cuts, just
further out. Several orderings attack the cut itself rather than paying to reach past it.

**"Is the LSE merge really exact, or exact up to something?"**
Exact. It reconstructs softmax attention over the multiset union of the per-hash windows. Keys
reachable by several orderings are counted several times, and that multiplicity is intended — it
is the amplification.

**"Why frozen and not learned?"**
The ordering depends only on (eta, phi), never on learned features, so it can be precomputed and
does not drift as the model trains. Learning it would mean the neighbourhood structure moving
under the model while it tries to learn from it.

**"What is the memory cost?"**
A wash: 0.21 GB vs 0.19 GB at N=13k. Masked already materialises a replicated copy of q/k/v, so
sorted's permuted tensors are not extra.

**"Does this work with batch size > 1?"**
Not currently — flex asserts batch size 1 whenever a block mask is present. ColliderML already
runs at batch size 1 so it has not bitten us.

**"How confident are you in the 11x?"**
It is measured on real event geometry, on an A100, in steady state with warm-up absorbed, and both
arms are asserted to produce identical output. The number I would treat with more caution is the
comparison to today's window, because it depends on how much of the model is attention — 5.6x for
the encoder, 1.44x per training step.

**"What would change your mind about the sorted implementation?"**
If mask construction stopped dominating — a much larger window, or far fewer orderings, would
shift the balance. The masked path is still in the tree and selectable precisely so that is a
measurement rather than an argument.
