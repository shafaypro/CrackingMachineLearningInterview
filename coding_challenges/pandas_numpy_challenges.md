# Pandas and NumPy Challenges

Data-manipulation screens are the most common coding round for Data Scientists and ML Engineers, and the most commonly under-prepared. The tasks are rarely algorithmically hard: they test whether you can reshape real data fluently, avoid the silent-correctness traps, and know why the vectorized version is 100× faster.

Each challenge below gives the problem, an idiomatic solution, and the follow-up interviewers ask.

---

## Table of Contents
1. [How These Rounds Are Scored](#how-these-rounds-are-scored)
2. [NumPy Fundamentals](#numpy-fundamentals)
3. [Broadcasting](#broadcasting)
4. [Vectorization Over Loops](#vectorization-over-loops)
5. [GroupBy Patterns](#groupby-patterns)
6. [Window and Rolling Operations](#window-and-rolling-operations)
7. [Joins and the Duplicate Trap](#joins-and-the-duplicate-trap)
8. [Reshaping](#reshaping)
9. [Time Series Operations](#time-series-operations)
10. [Missing Data](#missing-data)
11. [Performance and Memory](#performance-and-memory)
12. [ML-Specific Patterns](#ml-specific-patterns)
13. [Complexity and Cost Reference](#complexity-and-cost-reference)
14. [Interview Q&A](#interview-qa)
15. [Common Pitfalls](#common-pitfalls)
16. [Related Topics](#related-topics)

---

## How These Rounds Are Scored

1. **Clarify the schema and the grain.** "One row per what?" prevents most wrong answers. Ask about duplicates, nulls, and whether timestamps are sorted and timezone-aware.
2. **State the approach before typing.** "Group by user, take the last event per group, then join back" earns credit even if the syntax needs a lookup.
3. **Vectorize, but correctness first.** A working `apply` beats a broken one-liner. Then say "this is `O(n)` Python-level calls; here's the vectorized version" and rewrite it.
4. **Name the traps unprompted.** Chained assignment, silent join fan-out, `groupby` dropping NaN keys, off-by-one in rolling windows. Mentioning these signals real experience.
5. **Check your output.** `df.shape` before and after a merge is the single highest-value habit, it catches fan-out immediately.

---

## NumPy Fundamentals

```python
import numpy as np

a = np.arange(12).reshape(3, 4)     # (3, 4)

a.shape, a.dtype, a.ndim, a.nbytes  # always know these

# Views vs copies: the correctness trap
b = a[:, 1:3]      # VIEW: writing to b modifies a
c = a[:, [1, 2]]   # COPY: fancy indexing always copies
b[0, 0] = 999      # a is now modified
```

**Basic slicing returns a view; fancy (integer/boolean) indexing returns a copy.** This is the source of "why did my original array change?" and its mirror "why didn't my change stick?". Use `np.shares_memory(a, b)` to check when unsure.

```python
# Boolean masking
x = np.array([1, -2, 3, -4, 5])
x[x < 0] = 0                        # in-place clamp
np.where(x > 2, x, 0)               # vectorized conditional

# Aggregation along axes: axis is the one that DISAPPEARS
m = np.arange(12).reshape(3, 4)
m.sum(axis=0)     # (4,): collapses rows, one value per column
m.sum(axis=1)     # (3,): collapses columns, one value per row
m.sum(axis=1, keepdims=True)   # (3, 1): keeps dims for broadcasting
```

**The `axis` mnemonic worth memorizing**: `axis=k` is the axis that gets *removed* by the reduction. That resolves nearly all axis confusion.

---

## Broadcasting

Rules, applied right-to-left: dimensions are compatible if they're equal, or one of them is 1, or one is missing.

```python
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])          # (4,)   → (1,4) → (3,4)   ✓
A + b

c = np.array([1, 2, 3])             # (3,)   → (1,3) vs (3,4)  ✗ error
A + c[:, None]                      # (3,1) vs (3,4)           ✓ explicit reshape
```

**Challenge: pairwise distances without loops:**

```python
def pairwise_sq_dists(X, Y):
    """(n, d), (m, d) -> (n, m) squared Euclidean distances.
    Uses ||x-y||² = ||x||² - 2x·y + ||y||², avoiding an (n, m, d) tensor."""
    return (
        (X ** 2).sum(1)[:, None]     # (n, 1)
        - 2 * X @ Y.T                # (n, m)
        + (Y ** 2).sum(1)[None, :]   # (1, m)
    )
```

**Follow-up**: *why not `((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)`?* It materializes an `(n, m, d)` intermediate. For n=m=10,000 and d=128 that's 10⁸ × 128 × 8 bytes ≈ 100 GB. The expansion above peaks at `(n, m)` and delegates the heavy work to an optimized BLAS matmul.

---

## Vectorization Over Loops

```python
import pandas as pd

df = pd.DataFrame({"price": np.random.rand(1_000_000) * 100,
                   "qty": np.random.randint(1, 10, 1_000_000)})

# Slowest → fastest
# df.apply(lambda r: r.price * r.qty, axis=1)   # ~10 s: row-wise Python
# df["price"].combine(df["qty"], lambda a, b: a*b)  # slow
df["total"] = df["price"] * df["qty"]           # ~5 ms: vectorized
```

Roughly **1000× difference**. `apply(axis=1)` constructs a Series per row; the vectorized version is a single C-level operation over contiguous memory.

**Conditional logic without `apply`:**

```python
# np.select for multi-branch conditions
conditions = [df.price > 80, df.price > 50, df.price > 20]
choices = ["premium", "high", "medium"]
df["tier"] = np.select(conditions, choices, default="low")

# np.where for two branches
df["flag"] = np.where(df.qty > 5, "bulk", "single")

# pd.cut for binning
df["band"] = pd.cut(df.price, bins=[0, 20, 50, 80, np.inf],
                    labels=["low", "medium", "high", "premium"])
```

When you need per-row Python (calling an external API, complex branching), `apply` is acceptable, but say so explicitly rather than reaching for it by default.

---

## GroupBy Patterns

```python
sales = pd.DataFrame({
    "region": ["E", "W", "E", "W", "E"],
    "rep": ["a", "b", "a", "c", "b"],
    "amount": [100, 200, 150, 300, 250],
    "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01",
                            "2026-02-03", "2026-03-01"]),
})

# Named aggregation: clean, flat column names
sales.groupby("region").agg(
    total=("amount", "sum"),
    avg=("amount", "mean"),
    n_reps=("rep", "nunique"),
)

# transform: aggregate broadcast back to original shape (no join needed)
sales["region_total"] = sales.groupby("region")["amount"].transform("sum")
sales["pct_of_region"] = sales["amount"] / sales["region_total"]

# filter: keep whole groups meeting a condition
sales.groupby("region").filter(lambda g: g["amount"].sum() > 400)
```

**`transform` vs `agg` vs `apply`** is a standard question:

| Method | Returns | Use for |
|---|---|---|
| `agg` | One row per group | Summaries |
| `transform` | **Same shape as input** | Adding group statistics as a column |
| `filter` | Subset of original rows | Dropping whole groups |
| `apply` | Anything | Last resort: slowest, most flexible |

**Challenge: top-N per group:**

```python
# Idiomatic and fast
top2 = sales.sort_values("amount", ascending=False).groupby("region").head(2)

# Rank-based, keeps ties explicit
sales["rk"] = sales.groupby("region")["amount"].rank(method="dense", ascending=False)
top2 = sales[sales.rk <= 2]
```

**Trap**: `groupby` **drops NaN keys by default**. If a grouping column has nulls those rows vanish silently and totals won't reconcile. Use `dropna=False` when nulls are meaningful.

---

## Window and Rolling Operations

```python
df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=100),
                   "value": np.random.randn(100).cumsum()})
df = df.set_index("date")

df["ma7"] = df["value"].rolling(7).mean()                  # trailing 7 rows
df["ma7_min3"] = df["value"].rolling(7, min_periods=3).mean()   # tolerate warm-up
df["ma_time"] = df["value"].rolling("7D").mean()           # time-based, gap-safe
df["expanding"] = df["value"].expanding().mean()           # all history to date
df["ewm"] = df["value"].ewm(span=7).mean()                 # exponential weighting
df["pct_change"] = df["value"].pct_change()
df["lag1"] = df["value"].shift(1)
```

**`rolling(7)` vs `rolling("7D")`** is a favourite follow-up: the first takes 7 *rows* regardless of dates, the second takes 7 *days* regardless of row count. With missing days they give different answers, and the row-based version silently reaches further back than intended.

**The leakage trap in ML feature engineering:**

```python
# WRONG: includes the current row, leaking the label's own value
df["feat"] = df.groupby("user")["target"].transform(lambda s: s.rolling(7).mean())

# RIGHT: shift so only strictly prior values are used
df["feat"] = df.groupby("user")["target"].transform(
    lambda s: s.shift(1).rolling(7).mean()
)
```

Forgetting the `shift(1)` on a target-derived rolling feature is the single most common cause of "amazing offline, useless in production".

---

## Joins and the Duplicate Trap

```python
orders = pd.DataFrame({"order_id": [1, 2, 3], "cust_id": [10, 20, 10]})
custs = pd.DataFrame({"cust_id": [10, 20, 30], "name": ["A", "B", "C"]})

m = orders.merge(custs, on="cust_id", how="left", validate="many_to_one")
```

**`validate=` is the highest-value habit in this section.** It raises immediately if the join isn't the cardinality you assumed:

| Value | Asserts |
|---|---|
| `one_to_one` | Keys unique on both sides |
| `one_to_many` | Left keys unique |
| `many_to_one` | **Right keys unique**: the usual dimension lookup |
| `many_to_many` | No check (the default behavior) |

Silent fan-out is the classic disaster: duplicate keys on the right side multiply rows, every downstream sum doubles, and nothing errors. Always compare `df.shape` before and after.

```python
# Indicator reveals match quality
m = orders.merge(custs, on="cust_id", how="outer", indicator=True)
m["_merge"].value_counts()    # both / left_only / right_only

# as-of join: match the most recent prior record (time series bread and butter)
pd.merge_asof(trades.sort_values("time"), quotes.sort_values("time"),
              on="time", by="symbol", direction="backward")
```

`merge_asof` is worth knowing cold: it's the correct tool for point-in-time correct feature joins, and reaching for it signals experience with temporal data.

---

## Reshaping

```python
wide = pd.DataFrame({"id": [1, 2], "jan": [10, 20], "feb": [30, 40]})

# Wide → long
long = wide.melt(id_vars="id", var_name="month", value_name="value")

# Long → wide
back = long.pivot(index="id", columns="month", values="value").reset_index()

# pivot_table aggregates duplicates (pivot raises on them)
pd.pivot_table(long, index="id", columns="month", values="value",
               aggfunc="sum", fill_value=0)

# Flatten a MultiIndex after aggregation
agg = sales.groupby("region").agg({"amount": ["sum", "mean"]})
agg.columns = ["_".join(c) for c in agg.columns]
```

**`pivot` vs `pivot_table`**: `pivot` raises on duplicate index/column pairs; `pivot_table` aggregates them. If `pivot` errors, your data isn't at the grain you thought: investigate rather than swapping to `pivot_table` reflexively.

```python
# Explode list-valued cells into rows
df = pd.DataFrame({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
df.explode("tags")
```

---

## Time Series Operations

```python
df = df.set_index(pd.to_datetime(df["date"]))

df.resample("D").sum()          # downsample to daily
df.resample("W-MON").mean()     # weekly, weeks starting Monday
df.resample("M").agg({"value": "sum", "id": "nunique"})

# Fill gaps explicitly: resample creates rows for missing periods
daily = df.resample("D").sum().fillna(0)

# Timezones
s = df.index.tz_localize("UTC").tz_convert("Europe/Berlin")

# Business days
pd.date_range("2026-01-01", periods=10, freq="B")
```

**Timezone handling** is a frequent source of silent bugs: mixing naive and aware timestamps raises in comparisons, and localizing data that's already in local time shifts everything. Store UTC, convert at the edges.

---

## Missing Data

```python
df.isna().sum()                       # count per column
df.isna().mean().sort_values()        # proportion: more useful

df.dropna(subset=["important_col"])   # drop rows missing a specific column
df.fillna({"a": 0, "b": df.b.median()})   # per-column strategies

df["v"].ffill()                       # forward fill: legitimate for time series
df["v"].interpolate(method="time")    # time-aware interpolation
```

**`None` vs `NaN` vs `NaT` vs `pd.NA`** trips people up: `NaN` is a float, so an integer column with a missing value upcasts to float; `NaT` is the datetime equivalent; `pd.NA` is the newer, dtype-agnostic sentinel used by nullable dtypes (`Int64`, `boolean`, `string`). Using `Int64` (capital I) preserves integer semantics with nulls.

**The critical ML point**: imputation must be **fit on training data only**. Computing the median over the full dataset before splitting leaks test information. Do it inside a `Pipeline`.

---

## Performance and Memory

```python
# Memory profile
df.memory_usage(deep=True).sort_values(ascending=False)

# Downcast numerics
df["small_int"] = pd.to_numeric(df["small_int"], downcast="integer")
df["f32"] = df["f64"].astype("float32")

# Category dtype: enormous win for low-cardinality strings
df["country"] = df["country"].astype("category")   # often 10-50× smaller
```

**Category dtype is the single biggest memory lever** for typical dataframes: a string column with 200 distinct values across 10M rows drops from ~600 MB to ~10 MB, and groupby on it gets faster too.

```python
# Read only what you need
pd.read_csv("big.csv", usecols=["a", "b"], dtype={"a": "int32"})

# Chunked processing for files larger than memory
totals = sum(chunk.groupby("k")["v"].sum() for chunk in
             pd.read_csv("huge.csv", chunksize=100_000))

# Parquet over CSV: columnar, typed, compressed, ~5-10× faster to read
df.to_parquet("data.parquet")
```

**`query` and `eval`** avoid intermediate allocations on large frames:

```python
df.query("price > 100 and qty < 5")     # no boolean intermediates
```

When pandas isn't enough: **Polars** (multi-threaded, lazy, much faster), **DuckDB** (SQL over dataframes and Parquet, excellent for joins and aggregations), or **Dask**/**Spark** for distributed. Naming DuckDB as the pragmatic middle step is a strong answer.

---

## ML-Specific Patterns

```python
# Train/test split respecting time
cutoff = df["date"].quantile(0.8)
train, test = df[df.date <= cutoff], df[df.date > cutoff]

# One-hot with consistent columns between train and serve
X_train = pd.get_dummies(train[cats], columns=cats)
X_test = pd.get_dummies(test[cats], columns=cats).reindex(
    columns=X_train.columns, fill_value=0        # CRITICAL: align columns
)

# Target encoding computed out-of-fold to avoid leakage
from sklearn.model_selection import KFold
df["target_enc"] = np.nan
for tr, va in KFold(5, shuffle=True, random_state=42).split(df):
    means = df.iloc[tr].groupby("cat")["target"].mean()
    df.iloc[va, df.columns.get_loc("target_enc")] = df.iloc[va]["cat"].map(means)
```

The `reindex` on one-hot encoding is a real production bug source: an unseen category at serving time produces a different column set, and the model receives a misaligned feature vector, often without erroring.

---

## Complexity and Cost Reference

| Operation | Cost | Note |
|---|---|---|
| Vectorized arithmetic | `O(n)`, C speed | Baseline |
| `apply(axis=1)` | `O(n)` Python calls | ~100–1000× slower |
| `groupby.agg` | `O(n)` | Hash-based, fast |
| `merge` on unsorted keys | `O(n + m)` | Hash join |
| `sort_values` | `O(n log n)` | Consider `nlargest` for top-k |
| `nlargest(k)` | `O(n log k)` | Beats full sort |
| `rolling` | `O(n · w)` naive, `O(n)` for mean/sum | Optimized for common aggregations |
| `pd.concat` in a loop | **`O(n²)`** | Build a list, concat once |
| `.iterrows()` | Very slow | Almost never correct |

---

## Interview Q&A

#### Why is `apply(axis=1)` slow, and what do you use instead?

It calls a Python function once per row, and each call constructs a Series object for that row. So you pay Python interpreter overhead plus object allocation `n` times, with no opportunity for the underlying C loops or SIMD to help. On a million rows that's typically 100–1000× slower than the vectorized equivalent.

Instead: plain vectorized arithmetic for math, `np.where` for two-branch conditionals, `np.select` for multi-branch, `pd.cut` for binning, `map` for dictionary lookups, and `groupby().transform()` for group-relative computations. If the logic can't vectorize (calling an external service, say) `apply` is fine, but I'd name that explicitly rather than defaulting to it.

#### Explain views versus copies in NumPy.

Basic slicing (`a[1:3, :]`) returns a **view** (a new array object pointing at the same memory), so writing through it modifies the original. Fancy indexing (integer arrays or boolean masks) returns a **copy**, so writes don't propagate back.

That asymmetry causes both classic bugs: "why did my source array change?" when someone modifies a slice, and "why didn't my change stick?" when they write through a boolean mask expecting a view. `np.shares_memory(a, b)` settles it when unsure, and `.copy()` makes the intent explicit.

The pandas analogue is `SettingWithCopyWarning`, which fires when chained indexing makes it ambiguous whether you're writing to a view or a temporary. The fix is a single `.loc` call: `df.loc[mask, "col"] = value` rather than `df[mask]["col"] = value`.

#### Your merge produced more rows than the left dataframe. What happened?

Duplicate keys on the right side, so each left row matched multiple right rows and fanned out. It's the most damaging silent bug in data work: nothing errors, but every downstream sum is inflated.

The prevention is `validate="many_to_one"` on the merge, which raises immediately if the right keys aren't unique. The diagnosis is comparing `df.shape` before and after (a habit worth having on every merge), and `custs["cust_id"].duplicated().sum()` to confirm.

Then decide what the duplicates mean: if they're genuine data errors, deduplicate; if the right side is legitimately at a finer grain, aggregate it to the join grain first, or accept the fan-out deliberately and adjust the downstream aggregation.

#### What's the difference between `agg`, `transform`, and `apply` on a groupby?

`agg` reduces each group to a single row: use it for summaries. `transform` returns something the **same shape as the input**, broadcasting the group result back to every row, which is what you want when adding a group statistic as a new column without a join. `filter` keeps or drops whole groups based on a group-level predicate. `apply` can return anything and is the most flexible, but it's also the slowest and its return-shape behaviour is inconsistent enough to be surprising.

The practical rule: if you're about to compute a group aggregate and merge it back onto the original frame, use `transform` instead: it's one operation, faster, and can't fan out.

#### How do you compute a rolling feature without leaking the label?

Shift before rolling. A rolling window that includes the current row incorporates the very value you're predicting, which produces spectacular offline metrics and a useless production model.

```python
df["feat"] = df.groupby("user")["target"].transform(lambda s: s.shift(1).rolling(7).mean())
```

Two more temporal cautions: sort by time within each group first, or the window is meaningless; and prefer `rolling("7D")` over `rolling(7)` when rows aren't evenly spaced, because the row-based window silently reaches back further than intended when data is missing.

The general habit is to write down, for each feature, "what would I actually have known at prediction time?", and if the answer involves the future, the feature is wrong regardless of how well it scores.

#### Your dataframe uses 8 GB and you have 4. Options?

In order of effort:

**Dtypes first**: usually the biggest win for the least work. `category` for low-cardinality strings often gives 10–50× on those columns; downcast `int64` → `int32`/`int16` and `float64` → `float32` where precision allows. Check with `df.memory_usage(deep=True)`.

**Load less**: `usecols` to read only needed columns, `dtype=` on read so you never materialize the wide version, and Parquet instead of CSV since it's columnar (read only the columns you need) and compressed.

**Chunk**: process in pieces with `chunksize` and aggregate incrementally, if the operation allows it.

**Change tool**: DuckDB runs SQL directly over Parquet files with out-of-core execution and is often the pragmatic answer for joins and aggregations; Polars is much more memory-efficient and multi-threaded for dataframe work; Dask or Spark when it's distributed-scale.

I'd try dtypes and Parquet first, since they're minutes of work and frequently sufficient.

#### Why is `pd.concat` inside a loop a problem?

Each `concat` allocates a new dataframe and copies everything accumulated so far, so appending `n` times copies `1 + 2 + ... + n` rows: quadratic. For a few thousand iterations it's the difference between milliseconds and minutes.

The fix is to accumulate into a Python list and call `pd.concat(parts)` once at the end, which allocates a single result. The same reasoning applies to repeatedly appending to a Series or growing a NumPy array with `np.append`: preallocate or collect and combine once.

#### How do you avoid one-hot encoding mismatches between training and serving?

Never call `get_dummies` independently on the two sets: the column set depends on which categories happen to appear, so an unseen category at serving time, or a missing one, produces a different shape or a misaligned feature vector.

Two robust options. **`reindex` against the training columns** with `fill_value=0`, which forces the serving frame to exactly the training schema. Better, use **`sklearn.OneHotEncoder(handle_unknown="ignore")` inside a `Pipeline`**: it learns the categories at fit time, produces a consistent output width, and handles unseen values gracefully. Putting it in the pipeline also means the same object is serialized with the model, so training and serving can't diverge.

The failure mode this prevents is nasty because it often doesn't raise: the model receives numbers in the wrong columns and returns confident nonsense.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| `apply(axis=1)` by default | 100–1000× slower than vectorized | `np.where`, `np.select`, arithmetic, `transform` |
| Chained assignment `df[m]["c"] = v` | May write to a temporary; silently no-ops | Single `.loc[m, "c"] = v` |
| `merge` without `validate=` | Silent fan-out inflates every downstream sum | `validate="many_to_one"`; check `.shape` |
| `groupby` with NaN keys | Rows silently dropped; totals don't reconcile | `dropna=False` |
| Rolling feature without `shift(1)` | Leaks the target into its own feature | Shift before rolling |
| `rolling(7)` on irregular timestamps | Window spans an unintended time range | `rolling("7D")` |
| `pd.concat` in a loop | Quadratic copying | Collect in a list, concat once |
| `get_dummies` fit separately per split | Column mismatch at serving | `reindex` or `OneHotEncoder` in a pipeline |
| Imputing before the train/test split | Leaks test statistics into training | Fit imputers inside a `Pipeline` |
| Object dtype for repeated strings | Huge memory cost | `astype("category")` |
| Mixing tz-naive and tz-aware timestamps | Comparison errors or silent offsets | Store UTC, convert at the edges |
| `int` columns silently becoming float | NaN forces float upcast | Nullable `Int64` dtype |
| Full sort to get top-k | `O(n log n)` when `O(n log k)` suffices | `nlargest(k)` |

---

## Related Topics

- [Python Coding Challenges](./python_coding_challenges.md)
- [SQL Coding Challenges](./sql_coding_challenges.md)
- [ML Coding Challenges](./ml_coding_challenges.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [DuckDB](../data_engineering/intro_duckdb.md)
- [Apache Spark](../data_engineering/intro_apache_spark.md)
- [Take-Home Projects](../docs/take-home-projects.md)
- [Coding Challenges Overview](./README.md)
