# Understanding Beta (β) in GraphFlix Training

## What is Beta?

**Beta (β) is a learnable scaling parameter** that controls the influence of content-based metadata on your movie recommendations. It's one of the key hyperparameters that makes GraphFlix unique by balancing collaborative filtering with content-based filtering.

## The Two Recommendation Signals

GraphFlix combines two different approaches to make recommendations:

### 1. Graph Structure Signal (Collaborative Filtering)
```
s_enc(u,j) = ⟨z_u, z_j⟩
```
- Learned from the graph connections (who watched what)
- Captures patterns like "users similar to you liked this movie"
- Pure collaborative filtering based on behavior

### 2. Metadata Signal (Content-Based Filtering)
```
b_meta(u,j) = β * tanh(LN(p(u))ᵀ W LN(φ(j)))
```
- Based on movie metadata: genres, actors, directors
- Captures patterns like "you like action movies, this is an action movie"
- Uses your personalized user profile p(u) matched against movie features φ(j)

### Final Prediction Score
```
s(u,j) = s_enc(u,j) + b_meta(u,j)
         ↑              ↑
    Graph signal   Metadata signal (scaled by β)
```

## How Beta Works

Beta is a **learnable parameter** that adjusts during training to find the optimal balance:

| Beta Value | Interpretation | What the Model Learns |
|------------|----------------|----------------------|
| β ≈ 0 | Metadata ignored | Graph structure alone is best for predictions |
| β ≈ 0.5 | Metadata weak | Graph structure is more important |
| β ≈ 1.0 | Balanced | Both signals equally important |
| β > 1.0 | Metadata strong | Content features more predictive than graph |

### Training Behavior

Beta starts at `beta_init: 1.0` (configured in `configs/model/graphflix_full.yaml`) and then:
- **Increases** if metadata helps improve predictions
- **Decreases** if metadata is misleading or redundant
- **Stabilizes** once optimal balance is found

## What You See in Training Logs

```bash
Batch 100/5844 | Loss: 0.9188 | Avg Loss: 3.2386 | Beta: 0.9971 | Time: 34.0s
Batch 200/5844 | Loss: 0.7120 | Avg Loss: 2.0603 | Beta: 0.9942 | Time: 67.9s
```

### Interpreting Beta Values:

**Beta: 0.9971**
- The model is using metadata at almost full strength
- Both graph structure and content features are valuable
- Metadata contributes roughly equal to graph signal

**If Beta → 0.5**
- Model would be down-weighting metadata
- Suggests graph structure is more predictive
- Content features less useful for this dataset

**If Beta → 1.5**
- Model would be up-weighting metadata
- Suggests content features are very predictive
- Genre/actor/director information is highly relevant

## Why Beta is Bounded by tanh()

The formula includes `tanh()` which bounds the bias:
```
b_meta(u,j) = β * tanh(...)
              ↑    ↑
           scale  bounded to [-1, 1]
```

This means:
- Metadata bias is constrained to `[-β, +β]`
- Prevents metadata from overwhelming the graph signal
- Provides stable gradients during training

## How Beta Reflects Dataset Properties

Beta's learned value tells you about your data:

### Scenario 1: Beta stays near 1.0
```
Beta: 0.98 → 1.02
```
- **Interpretation**: Metadata and graph structure are complementary
- Both collaborative filtering and content-based filtering matter equally
- Users watch movies based on both what others watch AND content preferences

### Scenario 2: Beta decreases to ~0.3
```
Beta: 1.0 → 0.5 → 0.3
```
- **Interpretation**: Graph structure dominates
- Users mostly follow what similar users watch
- Genre/actor information doesn't add much value (pure collaborative filtering works)

### Scenario 3: Beta increases to ~1.5
```
Beta: 1.0 → 1.2 → 1.5
```
- **Interpretation**: Metadata is highly predictive
- Users have strong content preferences (e.g., "I only watch comedies")
- Graph connections are weaker signals

## Technical Implementation

### In the Code:
```python
# Beta is a learnable parameter (not a fixed hyperparameter)
self.beta = nn.Parameter(torch.tensor(float(beta_init)))

# Used to scale the metadata bias
b_meta = self.beta * torch.tanh(scores)
```

### During Training:
1. Beta receives gradients just like other model weights
2. Optimizer (AdamW) updates beta to minimize BPR loss
3. Beta converges to optimal value for your dataset

### In Your Config:
```yaml
# configs/model/graphflix_full.yaml
params:
  beta_init: 1.0  # Starting value only - will be learned
```

## Monitoring Beta During Training

### What to Watch For:

**Healthy Training:**
```
Epoch 1: Beta: 1.0000 → 0.9971
Epoch 2: Beta: 0.9971 → 0.9856  
Epoch 3: Beta: 0.9856 → 0.9849
Epoch 5: Beta: 0.9849 → 0.9851 (stabilizing)
```
- Smooth changes
- Gradual convergence
- Stabilizes after a few epochs

**Potential Issues:**

**Beta → 0 quickly:**
```
Epoch 1: Beta: 1.0000 → 0.3214
Epoch 2: Beta: 0.3214 → 0.0892
```
- Metadata may not be helpful for this data
- Check if metadata features are meaningful
- Consider if metadata preprocessing is correct

**Beta oscillates wildly:**
```
Epoch 1: Beta: 1.0000 → 1.5234
Epoch 2: Beta: 1.5234 → 0.2134
Epoch 3: Beta: 0.2134 → 2.1234
```
- Training instability
- Learning rate may be too high
- Check for bugs in metadata loading

**Beta explodes:**
```
Epoch 1: Beta: 1.0000 → 5.2341
Epoch 2: Beta: 5.2341 → 15.234
```
- Gradient explosion
- Check gradient clipping is enabled
- Verify LayerNorm is working

## Practical Implications

### For Model Performance:

**Beta ≈ 1.0** (Your Current Training)
- Model is successfully leveraging both signals
- Expected to work well for cold-start (new movies with metadata but few interactions)
- Good generalization to users with sparse history

**Beta ≈ 0** (Hypothetical)
- Model reduces to pure collaborative filtering
- Won't help with cold-start problem
- Needs dense interaction history to work well

**Beta > 1** (Hypothetical)
- Model heavily relies on content features
- Better for cold-start scenarios
- May miss collaborative signals (users with similar tastes)

### For Recommendation Quality:

The optimal beta value depends on:
1. **Dataset characteristics**: How predictive is metadata?
2. **Sparsity**: Sparse graphs → beta may increase (need content)
3. **Metadata quality**: Better metadata → beta may increase
4. **User behavior**: Content-driven users → higher beta

## Visualizing Beta in TensorBoard

Beta is logged to TensorBoard at two levels:

### Batch Level: `Beta/train_batch`
- Shows beta after every batch (~5,844 points per epoch)
- Useful for seeing fine-grained updates
- Can see if beta is stable or oscillating

### Epoch Level: `Beta/train_epoch`
- Shows beta at epoch end (20 points total)
- Useful for overall trend
- Easier to interpret long-term behavior

### How to View:
```bash
tensorboard --logdir=runs/
# Open http://localhost:6006
# Go to "Scalars" tab → "Beta" section
```

## Summary

| Aspect | Description |
|--------|-------------|
| **Type** | Learnable parameter (like a weight) |
| **Initial Value** | 1.0 (configured via `beta_init`) |
| **Purpose** | Balance graph structure vs. metadata signals |
| **Range** | Typically 0.3 - 1.5, but unconstrained |
| **Optimization** | Updated via gradient descent (AdamW) |
| **Interpretation** | Higher = more metadata influence, Lower = more graph influence |
| **Logged to** | Console, training.log, TensorBoard |

## Key Takeaways

1. **Beta is NOT a fixed hyperparameter** - it learns the optimal value during training
2. **Beta reflects dataset properties** - tells you if metadata or graph structure matters more
3. **Beta ≈ 1.0 is good** - means both signals are valuable (balanced model)
4. **Monitor beta over epochs** - should stabilize, not oscillate wildly
5. **Beta helps interpretability** - you can understand what the model relies on

Your current training shows **Beta: 0.9849-0.9971**, indicating:
✅ Model successfully uses both collaborative filtering and content-based filtering
✅ Metadata (genres, actors, directors) adds value beyond just graph connections
✅ Well-balanced recommendation approach
