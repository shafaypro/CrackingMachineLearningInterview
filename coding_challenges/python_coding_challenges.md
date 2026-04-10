# Python Coding Challenges for Interviews

This guide focuses on the Python coding rounds that appear in ML engineer, AI engineer, data engineer, and backend-leaning interviews.

The goal is not competitive programming for its own sake. The goal is to get fast and reliable at the patterns that show up in real interview loops.

---

## What Interviewers Usually Test

- choosing the right data structure quickly
- writing correct code without overengineering
- handling edge cases
- explaining time and space complexity
- improving a naive solution into a better one

---

## 1. Arrays, Strings, and Hash Maps

### Concepts to know

- arrays and list traversal
- dictionaries and frequency counting
- sets for fast membership checks
- prefix sums for repeated range questions

### Typical prompts

- find duplicates
- group anagrams
- top-k frequent elements
- longest substring without repeats
- count items by category

### Example: top-k frequent elements

```python
from collections import Counter
import heapq

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    return [value for value, _ in heapq.nlargest(k, counts.items(), key=lambda x: x[1])]
```

### Why it matters

This pattern appears in:

- token frequency analysis
- event counting
- label distribution checks
- feature summarization

---

## 2. Two Pointers and Sliding Window

### Concepts to know

- left and right pointers
- fixed-size windows
- variable-size windows
- shrinking windows when constraints break

### Typical prompts

- longest substring with at most `k` distinct values
- maximum sum subarray
- minimum window meeting a condition
- deduplicating sorted arrays in place

### Example: longest substring without repeating characters

```python
def longest_unique_substring(s: str) -> int:
    seen = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)

    return best
```

### Why it matters

This is useful for:

- stream processing
- log analysis
- sequence feature extraction
- windowed anomaly logic

---

## 3. Sorting, Intervals, and Heaps

### Concepts to know

- sorting by a key
- greedy interval merging
- min-heaps and max-heaps
- top-k maintenance

### Typical prompts

- merge overlapping intervals
- meeting rooms
- kth largest element
- schedule optimization

### Example: merge intervals

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return merged
```

### Why it matters

The same reasoning appears in:

- sessionization
- batching jobs into windows
- merging date ranges
- summarizing overlapping events

---

## 4. Stack and Queue Patterns

### Concepts to know

- stack for nested structure and monotonic behavior
- queue and deque for breadth-first or rolling problems
- monotonic stack for next greater and histogram patterns

### Typical prompts

- valid parentheses
- minimum stack
- sliding window maximum
- process tasks in FIFO order

### Example: valid parentheses

```python
def is_valid_parentheses(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []

    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False

    return not stack
```

### Why it matters

These patterns show up in:

- parser logic
- expression validation
- workflow state handling
- streaming windows with deques

---

## 5. Trees and Graphs

### Concepts to know

- DFS and BFS
- recursion versus explicit stack
- adjacency lists
- cycle detection
- topological ordering

### Typical prompts

- number of islands
- clone graph
- detect cycle in prerequisites
- shortest path in an unweighted graph

### Example: topological sort for dependencies

```python
from collections import defaultdict, deque

def topo_sort(edges: list[tuple[str, str]]) -> list[str]:
    graph = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()

    for src, dst in edges:
        graph[src].append(dst)
        indegree[dst] += 1
        nodes.add(src)
        nodes.add(dst)

    q = deque([node for node in nodes if indegree[node] == 0])
    order = []

    while q:
        node = q.popleft()
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    return order if len(order) == len(nodes) else []
```

### Why it matters

Graph problems map directly to:

- pipeline dependencies
- task orchestration
- lineage traversal
- service dependency analysis

Related repo reading:

- [Apache Airflow](../data_engineering/intro_apache_airflow.md)
- [Backend System Design Interview Guide](../system_design/backend_system_design_interview_guide.md)

---

## 6. Binary Search

### Concepts to know

- searching sorted arrays
- first true / last false template
- binary search on answer space

### Typical prompts

- first bad version
- search insert position
- minimum feasible capacity
- threshold tuning

### Example: first index greater than or equal to target

```python
def lower_bound(nums: list[int], target: int) -> int:
    left, right = 0, len(nums)

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

### Why it matters

This pattern appears in:

- threshold search
- hyperparameter search intuition
- lookup tables
- ranking cutoffs

---

## 7. Dynamic Programming Basics

### Concepts to know

- overlapping subproblems
- memoization
- bottom-up table building
- state definition and transitions

### Typical prompts

- climbing stairs
- coin change
- longest common subsequence
- house robber

### Example: coin change

```python
def coin_change(coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0

    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                dp[value] = min(dp[value], dp[value - coin] + 1)

    return dp[amount] if dp[amount] <= amount else -1
```

### Why it matters

Even if the exact DP problem never appears in production code, it tests whether you can define state cleanly and reason about optimization.

---

## 8. Matrix and Tabular Operations

### Concepts to know

- row and column traversal
- prefix sums in 2D
- grid BFS and DFS
- batching and chunking records

### Typical prompts

- count islands in a matrix
- rotate a matrix
- find connected components
- compute rolling or grouped summaries

### Example: chunk records for batch inference

```python
def batch_items(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
```

### Why it matters

This appears constantly in:

- feature pipelines
- inference batching
- ETL jobs
- evaluation runs

Related repo reading:

- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Model Serving](../mlops/intro_model_serving.md)

---

## Practice Checklist

### Easy

- frequency maps
- set membership
- two-sum variants
- basic stack validation
- simple binary search

### Medium

- sliding window
- interval merge
- heap-based top-k
- BFS and DFS
- prefix sums

### Harder but high value

- topological sort
- DP with 1D or 2D state
- graph cycle detection
- matrix traversal with state

---

## What to Say in the Interview

For each Python problem:

1. State the brute-force idea first if it helps anchor the problem.
2. State the target complexity.
3. Choose the core structure: hash map, heap, deque, recursion, or graph.
4. Walk through one sample input.
5. Call out edge cases:
   - empty input
   - duplicates
   - negative values
   - single element
   - disconnected graph

---

## Common Mistakes

- using lists when a set or dictionary is the right tool
- forgetting to update the left pointer in sliding-window problems
- mutating interval references incorrectly
- missing visited-state tracking in graph traversal
- writing recursive code without clear base cases
- giving complexity only for the best case and not the real case

---

## Repo Concepts to Pair With Practice

- [Statistics & Probability Guide](../classical_ml/intro_statistics_probability.md)
- [Classical ML Overview](../classical_ml/README.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [Backend System Design Interview Guide](../system_design/backend_system_design_interview_guide.md)

If you can solve medium Python problems and explain the linked concepts cleanly, you will cover a large share of ML and AI coding rounds.
