# Mario Input Autoencoder

Shannon's source coding theorem sets a hard floor on compression: a source with
entropy `H` bits per event cannot be encoded losslessly in fewer than `H` bits per
event. That floor applies to every encoder, learned ones included, so it makes a
concrete and falsifiable prediction about what a trained latent space should look
like once its capacity is pinned to exactly `H`. This project tests that
prediction on a neural autoencoder that optimizes nothing but reconstruction error
and is given no knowledge of information theory.

The test bed is Super Mario 64 TAS controller input. Real `.m64` replays are
parsed into six mutually exclusive categories (`Directional, A, B, Z/L/R, Start,
No Input`), each frame is expanded into a stream of one-hot events, and that
stream is sliced into 100-event windows. Its empirical entropy measures 2.4683
bits per event, against 2.585 for a uniform six-way source, so a 100-event window
carries 246.83 bits on average. The latent is fixed at 247 bits to match, leaving
the models exactly the theoretical minimum and no slack. That constraint is the
point of the exercise: given spare capacity, a code could reconstruct well while
leaving bits dead or redundant, and any claim to have compressed to `H` would be
empty.

Training runs on 500k windows sampled i.i.d. from the empirical category
marginals rather than on the raw replay stream, which is central to the design
rather than a convenience. Sampling i.i.d. fixes the true entropy of the training
distribution at exactly 2.4683 bits per event, giving the experiment a target that
is computed rather than estimated. The real stream carries heavy temporal
structure, with inputs held for 100+ frames and long idle waits, which pushes its
true entropy lower by an unknown margin and would leave the central measurement
without a reference point. Discarding the runs buys an exact bound.

Three models are compared at that latent size. Two are binarizing autoencoders,
identical in architecture (`600 → 512 → 247 → 512 → 600`) and parameter count
(869,199), differing only in how they binarize: `AnnealingAutoencoder` applies
`sigmoid(β · logits)` with `β` hardened toward a step over 50 epochs, while
`STEAutoencoder` applies a hard threshold with a straight-through gradient. The
third is a lookup memoriser that learns nothing, assigning every window in the
real dataset a fixed random code and encoding anything unseen to zeros. It makes
the standard objection to any autoencoder result, that the network merely
memorized a `sample ↔ bitstring` table, literal enough to measure against. All
three are evaluated on reconstruction, latent correlation, and per-bit entropy.

Everything lives in `main.ipynb`, alongside the source TAS replays (`120_stars`,
`70_stars`, `big_bob_omb_on_the_summit`).

## Results

| | Annealing | STE | Control (lookup) |
|---|---|---|---|
| Mean per-bit entropy (of max 1.0) | 0.984 | 0.987 | 0.000 |
| Bits that vary at all | 247/247 | 247/247 | 0/247 |
| Mean pairwise \|r\| between bits | 0.029 | 0.014 | n/a |
| Bit pairs with \|r\| > 0.5 | 0 of 30,381 | 0 of 30,381 | n/a |
| Reconstruction, macro F1 | 0.973 | 0.767 | 0.048 |

The central result is that an autoencoder handed exactly `H` bits of capacity
learns a code with the statistical properties theory predicts for an optimal one.
A representation compressed to its entropy bound should itself be incompressible,
its bits indistinguishable from fair, independent coin flips, since any remaining
structure would be residual redundancy a better code could squeeze out. Both
trained models land there: per-bit entropy at roughly 0.98 of the maximum, all 247
bits varying, and not one of the 30,381 bit pairs correlating above `|r| = 0.5`.
Nothing in the loss function asked for decorrelated or high-entropy bits.
Reconstruction pressure against a bound-sized bottleneck was enough to produce
them.

That signature would be worth little if the code were not encoding the sequence
itself. A representation keeping only category counts would need far fewer than
247 bits, and its bits could still show high entropy and low correlation, making
the result above an artifact of a bag-of-events summary rather than evidence of
efficient sequence coding. Reconstruction fidelity rules this out on its own,
because it is scored per position: the loss is a 6-way cross-entropy at each of
the 100 slots, and a counts-only code would have to guess the marginal at every
slot, scoring near the control's 0.048 rather than 0.973. Per-position recovery at
that level is only possible if the 247 bits carry the ordered sequence.

The lookup control addresses the other standard objection, that the network simply
memorized a table. It behaves nothing like the trained models, reconstructing
unseen windows at 0.048 macro F1 against Annealing's 0.973, with not one of its
247 bits varying across the evaluation windows. Whatever the trained encoders learned
generalizes to windows they never encountered, which a lookup table cannot do by
construction.

The comparison between binarization schemes is secondary but unusually clean,
since the two models differ only in gradient path. Annealing reaches 0.973 macro
F1 against STE's 0.767, and STE's shortfall is structured rather than uniform: it
over-predicts common categories (`Directional` recall 0.993) at the expense of
rare ones (`B` recall 0.577), roughly what a biased straight-through gradient
would be expected to produce. One caution follows from this. STE's bit statistics
look every bit as optimal as Annealing's, so healthy-looking code statistics do
not by themselves imply a good code. The entropy signature and reconstruction
fidelity measure different things, and only the pair together supports the
argument above.

## Open questions

A fixed-width code sitting at exactly the entropy bound cannot be lossless in
general. It has no headroom, and lossless coding needs
either slack or variable-length output. Per-event accuracy also compounds across a
window, since exact recovery demands all 100 events land correctly, so
whole-window fidelity falls off far faster than the per-event rate suggests. The
honest description remains a strong lossy compressor at the bound rather than a
bijection, and the useful question is how small the gap gets.

A single point on the capacity curve is likewise suggestive rather than
conclusive. The strongest version of this result would sweep `LATENT_DIM` above
and below 247 and show fidelity degrading as capacity crosses the bound, then
repeat the experiment at a different `H`, for instance by re-binning categories,
to confirm that required capacity tracks entropy as entropy moves.

Finally, the source data is narrow and real gameplay remains untested.
`120_stars.m64` and `70_stars.m64` have nearly identical entropy at 2.4596 and
2.4720 bits, so the input distribution is barely probed for robustness. The
temporal structure discarded during training also implies the real stream's true
entropy sits well below 2.4683 bits per event, meaning a sequence model should
beat 247 bits per window on real runs. That is a harder question than the one
asked here, since the target `H` would have to be estimated before it could be
answered.

## Running it

Open `main.ipynb` and run top to bottom. It works on Colab (models persist to
Google Drive) or locally (models persist to the working directory). Dependencies
install in the first cell. Training is skipped when saved weights are found:

```python
SAVE_MODELS   = True    # load saved weights if present, otherwise train and save
FORCE_RETRAIN = False   # True ignores saved weights and retrains from scratch
```

Roughly 50 epochs per model over 400k synthetic windows. A GPU is strongly
recommended.
