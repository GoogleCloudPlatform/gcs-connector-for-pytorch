import os
import sys
import time

# Ensure we can find the dataflux core logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataflux_client_python')))

# Ensure we can find the compiled PyO3 rust library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'rust', 'target', 'release')))

try:
    import dataflux_rust
except ImportError as e:
    print(f"Failed to import dataflux_rust bindings: {e}")
    sys.exit(1)

from dataflux_core.fast_list import ListingController
from dataflux_core.download import dataflux_download_parallel, DataFluxDownloadOptimizationParams


def benchmark_listing(bucket: str, prefix: str, max_parallelism: int):
    print(f"--- Benchmarking Fast List ---")
    print(f"Bucket: {bucket}, Prefix: '{prefix}', Parallelism: {max_parallelism}")

    # Benchmark Native Python
    start_time = time.time()
    lister = ListingController(
        max_parallelism=max_parallelism,
        project=None, # Allow storage client to infer project from environment
        bucket=bucket,
        prefix=prefix,
        sort_results=True,
    )
    python_results = lister.run()
    python_duration = time.time() - start_time
    print(f"[Python] Discovered {len(python_results)} objects in {python_duration:.4f} seconds")

    # Benchmark Rust FFI
    start_time = time.time()
    rust_results = dataflux_rust.fast_list_wrapper(
        bucket, 
        prefix, 
        max_parallelism,
        True # skip_compose = True
    )
    # Rust results are un-sorted natively, so we sort them to match Python
    rust_results.sort(key=lambda x: x[0]) 
    rust_duration = time.time() - start_time
    print(f"[Rust] Discovered {len(rust_results)} objects in {rust_duration:.4f} seconds")

    assert len(python_results) == len(rust_results), "Result counts do not match!"
    print(f"Speedup: {python_duration / rust_duration:.2f}x\n")


def benchmark_download(bucket: str, prefix: str, num_objects: int):
    print(f"--- Benchmarking Compose Download ---")
    print(f"Bucket: {bucket}, Prefix: '{prefix}', Objects: {num_objects}")

    # Get a listing of objects to download
    lister = ListingController(
        max_parallelism=10,
        project=None, # Allow storage client to infer project from environment
        bucket=bucket,
        prefix=prefix,
        sort_results=True,
    )
    all_objects = lister.run()
    if len(all_objects) < num_objects:
        print(f"Warning: Only found {len(all_objects)} objects, benchmarking with that instead.")
        objects_to_download = all_objects
    else:
        objects_to_download = all_objects[:num_objects]

    # Benchmark Native Python
    opt_params = DataFluxDownloadOptimizationParams(max_composite_object_size=32 * 1024 * 1024)
    start_time = time.time()
    python_results = dataflux_download_parallel(
        project_name=None, # Allow storage client to infer project from environment
        bucket_name=bucket,
        objects=objects_to_download,
        dataflux_download_optimization_params=opt_params,
        parallelization=10,
    )
    python_duration = time.time() - start_time
    print(f"[Python] Downloaded {len(python_results)} objects in {python_duration:.4f} seconds")

    # Benchmark Rust FFI
    # In the rust benchmark, we just invoke the hierarchical compose and download directly 
    # instead of passing tuples. The Rust expects just names.
    names = [obj[0] for obj in objects_to_download]
    
    start_time = time.time()
    dataflux_rust.download_wrapper(
        bucket,
        names,
        2 # depth 2
    )
    rust_duration = time.time() - start_time
    print(f"[Rust] Downloaded {len(names)} objects in {rust_duration:.4f} seconds")

    print(f"Speedup: {python_duration / rust_duration:.2f}x\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Dataflux Rust vs Python")
    parser.add_argument("--bucket", type=str, required=True, help="GCS Bucket Name")
    parser.add_argument("--prefix", type=str, default="", help="GCS Prefix")
    parser.add_argument("--list-parallelism", type=int, default=10, help="Number of workers for fast list")
    parser.add_argument("--download-objects", type=int, default=100, help="Number of objects to download")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download benchmark due to disk space constraints")
    args = parser.parse_args()

    benchmark_listing(args.bucket, args.prefix, args.list_parallelism)
    if not args.skip_download:
        benchmark_download(args.bucket, args.prefix, args.download_objects)
