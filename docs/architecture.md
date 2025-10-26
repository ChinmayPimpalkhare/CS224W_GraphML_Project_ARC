# GraphFlix Architecture (Notes)

- Heterogeneous graph with users, movies, directors, actors, genres
- Temporal LOO split
- BPR training
- Metadata bias: b_meta(u,i) = beta * tanh(LN(p)^T W LN(phi))
- Half-life τ + rating-weight in p(u)