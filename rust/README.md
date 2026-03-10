# GCS Fast List (Rust)

This directory contains a high-performance Rust implementation of the DataFlux Fast Listing algorithm. The algorithm is designed to bypass the sequential limitations of the Google Cloud Storage (GCS) `ListObjects` API by dynamically distributing the listing workload across a large pool of concurrent workers.

## Algorithm Overview: Dynamic Work Stealing

Standard GCS listing is strictly sequential: you ask for a page of results, you get a `nextPageToken`, and you use that token to ask for the subsequent page. This architecture makes it impossible to quickly list massively populated buckets (e.g., buckets with millions of objects).

The **Fast List** algorithm solves this by utilizing lexicographical string math to proactively divide the namespace:

1. **Initialization:** The coordinator spawns `N` asynchronous workers. The entire possible string namespace `("", "")` is pushed to a central work queue.
2. **Execution:** An available worker pulls a lexicographical range (e.g., `("apple", "banana")`) from the queue and begins a standard GCS `ListObjects` paginated request using `lexicographic_start` and `lexicographic_end`.
3. **Work Stealing:** When a worker is paging through its assigned range, it constantly monitors the global pool of workers. If it detects that other workers have gone idle (because the work queue is empty), the active worker **splits** its remaining lexicographical range mathematically (e.g., it calculates the midpoint string between its current position and its end bound).
4. **Distribution:** The active worker keeps the lower half of the split for itself to continue paging seamlessly, and pushes the upper half of the split back onto the global work queue for an idle worker to immediately pick up.
5. **Termination:** The algorithm completes when all `N` workers are entirely idle and the global work queue is empty.
6. **Deduplication:** Because manual `lexicographic_start` bounds are inclusive, overlapping page queries will inherently retrieve duplicate objects on the boundaries. The final step is a global `.dedup_by()` pass to ensure exact accuracy.

This architecture ensures that regardless of how heavily skewed or alphabetized the objects in the bucket are, the workload is dynamically and continuously load-balanced across all available compute threads until the very last object is found.

## Rust Performance Enhancements

This Rust port implements several aggressive performance optimizations directly targeting the GCS HTTP API and Tokio concurrency model. On a 1.2 million object bucket, these optimizations have pushed the execution time down to ~8.3 seconds running from an 8 core machine with parallelism set to 20.

### 1. JSON Field Masking (Projection)
By default, the GCS REST/gRPC API returns a massive JSON payload for every single object, containing exhaustive metadata like ACLs, creation dates, KMS keys, and metageneration hashes. 
We aggressively utilize the `google-cloud-wkt` protobuf `FieldMask` to inject a read mask into the `ListObjects` request: `.set_paths(["name", "size", "storage_class"])`. 
By forcing GCS to drop all omitted fields server-side before transmission, we drastically reduce the network payload by over 80%, eliminating the `serde_json` parsing bottleneck.

### 2. Aggressive Multi-Stealing
In standard work-stealing implementations, an active worker typically splits its remaining work in half and gives away one piece to a single idle worker.
In our optimized model, the active worker checks how many idle workers currently exist. It then dynamically calculates `N` mathematically equal split points across its remaining range, handing out perfectly proportioned chunks of work to every idle worker simultaneously. This entirely eliminates "straggler" scenarios at the end of a run where only a subset of workers are active while others sit idle.

### 3. Asynchronous MPSC Architecture
While Python relies on heavy OS-level `multiprocessing` processes communicating via serialized Pipe/Queue mechanisms, this Rust implementation utilizes `tokio` green threads (tasks) communicating via highly-optimized `async_channel` channels. This provides lock-free, zero-copy pointer passing for the object vectors, allowing maximum CPU utilization strictly for handling network I/O rather than inter-process communication overhead.

## Usage

```bash
# Run the fast list benchmark against a predefined bucket with 20 parallel workers
cargo run --bin rust-gcs-sample fast_list 20
```
