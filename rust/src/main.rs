mod download;
mod range_splitter;
pub mod fast_list;

use google_cloud_storage::client::StorageControl;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create clients
    let client = StorageControl::builder().build().await?;
    let storage = google_cloud_storage::client::Storage::builder().build().await?;

    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("download");

    if command == "fast_list" {
        let max_parallelism = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
        println!("Running Fast List against bucket: tess-listing with max_parallelism: {}", max_parallelism);
        let start_time = std::time::Instant::now();
        
        let config = fast_list::FastListConfig {
            bucket: "tess-listing".to_string(),
            prefix: "".to_string(),
            max_parallelism,
            skip_compose: false,
        };

        match fast_list::fast_list(client, config).await {
            Ok(results) => {
                let duration = start_time.elapsed();
                println!("Fast List finished successfully.");
                println!("Discovered {} objects in {:?}", results.len(), duration);
            }
            Err(e) => eprintln!("Fast List failed: {:?}", e),
        }
    } else {
        // Run download
        let bucket_name = "jd-compose-rust";
        println!("Running compose sample against bucket: {}", bucket_name);

        let depth = args.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(2);
        println!("Using depth: {}", depth);

        let mut objects: Vec<String> = (0..100).map(|i| format!("obj_{}", i)).collect();
        objects.sort();

        match download::download(&client, &storage, bucket_name, objects, depth).await {
            Ok(_) => println!("Download finished successfully."),
            Err(e) => eprintln!("Download failed: {:?}", e),
        }
    }

    Ok(())
}
