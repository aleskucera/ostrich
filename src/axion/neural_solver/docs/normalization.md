# Input/Output Normalization in the MSE State Pipeline

## TL;DR (direct answer to the concern)

Yes — you **are** denormalizing the network output before computing the MSE. The
loss is genuinely evaluated in **absolute physical units**, so the per-dimension
standard deviation `σ` from `RunningMeanStd` does **not** appear as a weight in
the loss *objective*. Your mental model is correct.

My earlier phrasing ("the empirical std is acting as an implicit per-coordinate
loss weight") was imprecise and is **corrected below**. Because you denormalize
*before* the loss, `σ` algebraically cancels out of the objective. What remains
is only an **optimization-conditioning** effect (initialization, gradient
magnitude, weight-decay interaction), and most of that is absorbed by AdamW. It
is not a reweighting of what the model is asked to fit.

The practically important consequence (which ties back to the physics-informed
scaling idea) is at the very end: **if you swap the empirical `σ` for a
physics-derived scale but keep computing MSE in absolute units, it will not
change the objective at all** — only the optimization path. To make a physical
scale actually change the position-vs-velocity trade-off, you must weight the
loss (or compute it in scaled space).

---

## The full pipeline, step by step

Notation, per environment: state is `[q (dof_q) , qd (dof_qd)]`. We focus on the
**state** head (lambdas are analogous but separate).

### 1. Input normalization

Dataset statistics are fit once over the training set, and inputs are whitened
on every forward pass:

```500:528:src/axion/neural_solver/algorithms/sequence_model_trainer.py
    def compute_dataset_statistics(self, dataset):
        # compute the mean and std of the input and output of the dataset
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = max(512, self.batch_size),
            collate_fn = self.collate_fn,
            shuffle = False,
            num_workers = self.num_data_workers,
            drop_last = True
        )
        dataloader_iter = iter(dataloader)
        self.dataset_rms = {}

        for _ in range(len(dataloader)):
            data = next(dataloader_iter)
            data = self.preprocess_data_batch(data)

            for key in data.keys():
                if not (key in self.dataset_rms):
                    self.dataset_rms[key] = RunningMeanStd(
                        shape = data[key].shape[2:],
                        device = self.device
                    )

                self.dataset_rms[key].update(
                    data[key],
                    batch_dim = True,
                    time_dim = True
                )
```

`normalize`/un-normalize is the standard affine transform with a per-dimension
mean and std:

```83:88:src/axion/neural_solver/utils/running_mean_std.py
    def normalize(self, arr:torch.tensor, un_norm = False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean
        return result
```

Input normalization is uncontroversial: it conditions the encoder inputs and has
no bearing on the loss-weighting question. It happens inside the model forward:

```192:208:src/axion/neural_solver/models/mse_model.py
    def forward(self, input_dict, deterministic=False, inject_noise=False):
        del deterministic
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        features = self._extract_input_features(input_dict)
        if self.is_transformer:
            features = self.transformer_model(features)

        bsz, seq_len, feature_dim = features.shape
        features_flatten = features.contiguous().view(-1, feature_dim)

        regression_flatten = self.regression_head(features_flatten)
        reg_value = regression_flatten.view(bsz, seq_len, -1)
        reg_value = self._format_output(reg_value)
        return reg_value
```

### 2. Target construction (the "relative" delta)

The regression target is built in `preprocess_data_batch` →
`convert_next_states_to_prediction`. With `state_prediction_type: relative` the
target is the per-step delta `next - current` (with angular `q` wrapped to
`(-π, π]`):

```540:554:src/axion/neural_solver/neural_model_utils_providers/transformer_neural_utils_provider_new.py
        if self.state_prediction_type == "absolute":
            pred = next_bt.clone()
        else:
            pred = (next_bt - states_bt)
            if self.prediction_quantity_type == "full_state":
                q_delta = pred[..., : self.dof_q_per_env]
                q_sel = q_delta.index_select(-1, self.angular_q_indices.to(q_delta.device))
                _wrap_to_pi_(q_sel)
                q_delta.index_copy_(-1, self.angular_q_indices.to(q_delta.device), q_sel)
            elif self.prediction_quantity_type == "velocities_only":
                pred = pred[..., self.dof_q_per_env:]
```

This delta tensor is stored under `data['target']`, and it is **this delta**
whose mean `μ` and std `σ` get accumulated into `regression_state_rms` by the
statistics pass above.

### 3. Installing the output statistics

```330:333:src/axion/neural_solver/algorithms/sequence_model_trainer.py
                self.neural_model.set_output_rms(
                    self.dataset_rms.get('target') if self.has_state_head else None,
                    self.dataset_rms.get('target_lambda') if self.has_lambda_head else None,
                )
```

So `regression_state_rms.mean = μ` and `regression_state_rms.var = σ²` are the
per-coordinate moments of the **delta target**.

### 4. Forward pass and output **de**normalization

The regression head emits raw values `z` (this is the network's actual learnable
output). `_format_output` immediately maps them back to physical delta units via
`un_norm=True`, i.e. `y = z·σ + μ`:

```161:177:src/axion/neural_solver/models/mse_model.py
    def _format_output(self, regression_output):
        if not self.normalize_output:
            return regression_output
        if self.states_only:
            # Legacy / states-only path: single RMS covering the whole output.
            rms = getattr(self, "regression_state_rms", None) or self.regression_output_rms
            if rms is not None:
                regression_output = rms.normalize(regression_output, un_norm=True)
            return regression_output
        # Combined [state | lambda] output: unnormalize each slice independently.
        state_part = regression_output[..., : self.state_output_dim]
        lambda_part = regression_output[..., self.state_output_dim :]
        if getattr(self, "regression_state_rms", None) is not None:
            state_part = self.regression_state_rms.normalize(state_part, un_norm=True)
        if getattr(self, "regression_lambda_rms", None) is not None:
            lambda_part = self.regression_lambda_rms.normalize(lambda_part, un_norm=True)
        return torch.cat([state_part, lambda_part], dim=-1)
```

**Key fact:** `σ` and `μ` are stored buffers (no gradient). So the model output
that leaves `forward()` is already a *physical-unit predicted delta*. The "z"
(normalized) representation only exists *inside* the model.

### 5. Relative → absolute, then the loss

The trainer turns the predicted delta into an absolute next-state, builds the
absolute target, and computes the loss — all in **absolute units**:

```115:127:src/axion/neural_solver/algorithms/mse_trainer.py
    def compute_loss(self, data, train):
        del train
        regression_prediction = self.neural_model(data)

        regression_prediction = self._convert_regression_to_absolute(data, regression_prediction)
        regression_target = self._build_regression_target(data, regression_prediction)

        state_loss = self._compute_state_loss(regression_prediction, regression_target)
        if getattr(self.neural_model, "states_only", False):
            lambda_loss = torch.zeros((), device=state_loss.device, dtype=state_loss.dtype)
        else:
            lambda_loss = self._compute_lambda_loss(regression_prediction, regression_target)
        total_loss = self.state_loss_weight * state_loss + self.lambda_loss_weight * lambda_loss
```

`_convert_regression_to_absolute` adds the current state back to the predicted
delta (relative mode):

```51:56:src/axion/neural_solver/algorithms/mse_trainer.py
        if self._state_prediction_type == "relative":
            next_states = self.utils_provider.convert_prediction_to_next_states(
                data["states"], state_pred
            )
        else:
            next_states = state_pred
```

And the state loss is `1 − cos` on the angular `q` coordinates plus plain MSE on
`qd`, in absolute next-state units:

```84:98:src/axion/neural_solver/algorithms/mse_trainer.py
    def _compute_state_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Angular q uses 1 − cos (wrap-safe); qd uses MSE."""
        state_dim = self.utils_provider.state_prediction_dim
        q_dim = self._dof_q_per_env
        pred_state = prediction[..., :state_dim]
        tgt_state = target[..., :state_dim]
        q_loss = angular_prediction_loss(
            pred_state[..., :q_dim],
            tgt_state[..., :q_dim],
            angular_prediction_l2_weight=self.angular_prediction_l2_weight,
        )
        qd_loss = F.mse_loss(pred_state[..., q_dim:], tgt_state[..., q_dim:])
        return q_loss + qd_loss
```

So the end-to-end state path is:

```
inputs ─(input_rms.normalize)─▶ encoder+transformer ─▶ head ─▶ z
   z ─(y = z·σ + μ)─▶ predicted Δstate (physical units)
   predicted Δstate ─(+ current state, wrap)─▶ predicted next_state
   loss = (1−cos) on q  +  MSE on qd     [absolute units]
```

---

## The core question: does `σ` reweight the objective?

Look only at the `qd` part, where the loss is a clean MSE (the `q` part uses
`1−cos`, see the note below). For a single velocity coordinate `i`:

- Network raw output: `z_i`
- Denormalized prediction: `y_i = z_i · σ_i + μ_i`
- Predicted next velocity: `qd⁺_i = qd_i + y_i`, target `t_i = qd⁺_target,i`
- Absolute error: `e_i = qd⁺_i − t_i`

The loss term is simply `e_i²`. **`σ_i` does not appear in `e_i²`.** The value of
the loss depends only on the absolute prediction error. That is exactly what
"denormalize before MSE" buys you, and it is what you intended.

### Where does `σ` show up at all, then?

Only when you express the *same* loss as a function of the network's own raw
output `z_i`. Substituting `y_i = z_i σ_i + μ_i`:

```
e_i² = (z_i σ_i + μ_i + qd_i − t_i)²
     = σ_i² · (z_i − t̃_i)² ,   where  t̃_i = (t_i − qd_i − μ_i)/σ_i
```

This is the whole story in one line. Read two ways:

1. **As an objective over physical predictions** (`y_i`): the `σ_i` is gone. All
   coordinates are weighted 1:1 by absolute squared error. ← this is your loss.
2. **As an objective over the network's raw output** (`z_i`): it is a weighted
   regression `σ_i² · (z_i − t̃_i)²` onto the *normalized* target `t̃_i`.

The crucial point: **(1) and (2) are the same function.** The `σ_i²` weight in
view (2) exactly compensates for the fact that `z_i` is in normalized units. It
does **not** introduce any net preference for one coordinate over another in the
thing being minimized. There is no free lunch where normalization secretly
reweights an absolute-unit loss — the denormalization undoes it.

(Contrast: if you had computed the MSE in **normalized** space — `Σ (z_i − t̃_i)²`
with no `σ²` — *then* every coordinate would be equally weighted in normalized
units, which is a genuinely different objective and the usual reason people
normalize targets. You don't do that here; you denormalize first.)

### So what is the residual effect of output normalization here?

Since the function class is identical (the output layer is linear, so
`y = σ ⊙ (W h + b) + μ` is just another linear map `(σ⊙W) h + (σ⊙b+μ)`), output
normalization is a **reparametrization**. It cannot change the optimum. It only
changes the optimization *dynamics*:

- **Initialization scale.** With standard init of `W`, the effective absolute-unit
  output weights `σ⊙W` start at a scale matched to each coordinate's natural
  magnitude. This is the main practical benefit and is generally helpful.
- **Gradient magnitude.** `∂loss/∂z_i = 2 e_i σ_i`: the raw-output gradient is
  scaled by `σ_i`. Under plain SGD this would bias step sizes per coordinate.
  Under **AdamW** (your optimizer) the per-parameter second-moment normalization
  divides this back out, so the direction of the update is largely
  scale-invariant and the `σ_i` factor is mostly absorbed.
- **Weight decay interaction.** AdamW's decoupled weight decay acts on `W` (the
  normalized-space weights), not on the effective `σ⊙W`. For a small-`σ`
  coordinate, `W` must be large to produce the needed output, so proportional
  decay translates into a different absolute-space regularization than it would
  without normalization. This is a real but second-order effect.

**Correction to my earlier statement:** none of these amount to "the std silently
deciding how much the optimizer cares about each coordinate" in the sense of
reweighting the objective. They are conditioning effects on the optimization
path, not a change to what counts as a good fit. The objective itself is
`σ`-free absolute-unit error. I was wrong to call it an implicit loss weight.

### Note on the angular `q` coordinates

The `q` part does not use MSE; it uses `1 − cos(pred_q − tgt_q) + w·pred_q²` in
absolute next-state units:

```19:21:src/axion/neural_solver/utils/loss_utils.py
    periodic_term = torch.mean(1.0 - torch.cos(pred_q - tgt_q))
    prediction_l2_term = torch.mean(pred_q.square())
    return periodic_term + float(angular_prediction_l2_weight) * prediction_l2_term
```

`1 − cos` is bounded and curvature-`1` near zero error, so it behaves like
`½(error)²` for small errors but saturates for large ones. Same conclusion holds:
`σ` does not appear in this term either; it only affects how the network's raw
output is conditioned.

---

## Where per-coordinate weighting actually lives

If `σ` is *not* the knob that trades off position vs. velocity vs. joint, what
is? In the current code:

1. **The `q` vs `qd` loss form**: `1 − cos` (bounded, radians) for `q` vs raw MSE
   (rad²/s²) for `qd`. These have inherently different scales and curvatures, and
   that difference *is* a real, if implicit, weighting between position and
   velocity accuracy.
2. **`state_loss_weight` / `lambda_loss_weight`** in the YAML loss block — a
   global weight between the state and lambda heads (`total_loss` line above).
3. Nothing currently differentiates *between joints* in the loss: every `qd_i`
   contributes its absolute squared error equally, and every `q_i` contributes
   its `1−cos` equally.

There is presently **no** explicit per-coordinate physical weighting of the loss.

---

## Implication for the physics-informed scaling idea

This is the load-bearing consequence for the earlier brainstorm. Suppose you
replace the empirical `regression_state_rms` `σ` with a physics-derived scale
(natural frequency, inertia, `qd_char·dt`, etc.), per joint, via `set_output_rms`.

Given the pipeline above:

- Because the loss is computed in **absolute units after denormalization**, that
  physical scale **cancels out of the objective** exactly like the empirical
  `σ` does. The model would be fitting the identical absolute-error objective;
  only the initialization/conditioning would change. You would likely see little
  to no change in what the model converges to (especially under AdamW).

To make a physics scale actually change the position-vs-velocity (or
per-joint) trade-off the optimizer pursues, you need one of:

- **(A) Weight the loss explicitly.** Introduce per-coordinate weights `1/σ_phys²`
  (or any physical weighting) inside `_compute_state_loss`, e.g. a weighted MSE
  on `qd` and a weighted angular term on `q`. This directly changes the
  objective.
- **(B) Compute the loss in scaled/normalized space.** Evaluate the MSE on the
  *normalized* prediction and target (i.e. don't denormalize before the loss, or
  divide both sides by the physical scale). Then the scale genuinely sets the
  relative emphasis — `Σ (z_i − t̃_i)²` weights every coordinate equally in
  scaled units, and a physical scale chooses what "equal" means.
- **(C) Change the anchor (the deepest lever).** A physics-residual target
  (`next − f_physics(state)`) shrinks and homogenizes the residual itself, which
  changes the objective regardless of scaling.

In short: output normalization as currently wired is a **conditioning** tool, not
a **weighting** tool. Your concern is well-founded, and the takeaway is that a
physics-informed *scale* only becomes a physics-informed *trade-off* if it enters
the loss, not merely the (invertible) output normalization.
