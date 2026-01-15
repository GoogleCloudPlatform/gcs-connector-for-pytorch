mod sample;

use google_cloud_storage::client::StorageControl;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create clients
    let client = StorageControl::builder().build().await?;
    let storage = google_cloud_storage::client::Storage::builder().build().await?;

    // Run sample
    let bucket_name = "jd-compose-rust";
    println!("Running sample against bucket: {}", bucket_name);

    // List objects to compose
    // Ideally we would list them from the bucket, but for this test we know they are obj_0..obj_99
    let mut objects: Vec<String> = (0..100).map(|i| format!("obj_{}", i)).collect();
    // Sort to ensure deterministic order if needed, though 0..100 is sorted.
    objects.sort();

    match sample::sample(&client, &storage, bucket_name, objects).await {
        Ok(_) => println!("Sample finished successfully."),
        Err(e) => eprintln!("Sample failed: {:?}", e),
    }

    Ok(())
}
