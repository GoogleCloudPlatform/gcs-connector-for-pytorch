// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [START storage_compose_file]
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model::{Object, compose_object_request::SourceObject};

use futures::stream::{self, StreamExt};

pub async fn sample(client: &StorageControl, storage: &Storage, bucket_id: &str, objects: Vec<String>) -> anyhow::Result<()> {
    // GCS allows composing up to 32 objects at a time
    const MAX_COMPOSE_BATCH: usize = 32;
    // Limit concurrency to avoid hitting rate limits or overwhelming the client
    const MAX_CONCURRENCY: usize = 10;

    let chunks: Vec<_> = objects.chunks(MAX_COMPOSE_BATCH).enumerate().collect();
    
    let bodies = stream::iter(chunks)
        .map(|(i, chunk)| {
            let client = client.clone();
            let storage = storage.clone();
            let bucket_id = bucket_id.to_string();
            let chunk = chunk.to_vec();
            async move {
                let destination_name = format!("composed_batch_{}", i);
                println!("Processing chunk {}: Composing into {}", i, destination_name);

                let source_objects: Vec<SourceObject> = chunk
                    .iter()
                    .map(|name| SourceObject::new().set_name(name))
                    .collect();

                // 1. Compose
                let compose_result = client
                    .compose_object()
                    .set_destination(
                        Object::new()
                            .set_bucket(format!("projects/_/buckets/{bucket_id}"))
                            .set_name(&destination_name),
                    )
                    .set_source_objects(source_objects)
                    .send()
                    .await;

                let object = match compose_result {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("Failed to compose chunk {}: {:?}", i, e);
                        return Err(anyhow::Error::from(e));
                    }
                };

                println!("Chunk {}: Composed. Size: {}", i, object.size);

                // 2. Fetch Source Sizes (Concurrent & Ordered)
                println!("Chunk {}: Fetching component sizes...", i);
                let size_futures = chunk.iter().map(|name| {
                    let client = client.clone();
                    let bucket_id = bucket_id.clone();
                    // clone name for async block
                    let name = name.clone();
                    async move {
                         client
                            .get_object()
                            .set_bucket(format!("projects/_/buckets/{bucket_id}"))
                            .set_object(&name)
                            .send()
                            .await
                            .map(|o| o.size)
                            .map_err(anyhow::Error::from)
                    }
                });
                
                let sizes: Vec<i64> = stream::iter(size_futures)
                    .buffered(MAX_CONCURRENCY)
                    .collect::<Vec<anyhow::Result<i64>>>()
                    .await
                    .into_iter()
                    .collect::<anyhow::Result<Vec<i64>>>()
                    .map_err(|e| {
                         eprintln!("Failed to fetch sizes for chunk {}: {:?}", i, e);
                         e
                    })?;
                
                println!("Chunk {}: Sizes fetched: {:?}", i, sizes);

                // 3. Download & Demux
                println!("Chunk {}: Downloading & Decomposing {}", i, destination_name);
                let read_result = storage
                    .read_object(&format!("projects/_/buckets/{bucket_id}"), &destination_name)
                    .send()
                    .await;
                
                match read_result {
                    Ok(mut reader) => {
                        let mut current_file_idx = 0;
                        let mut current_file_written = 0;
                        let mut current_file: Option<tokio::fs::File> = None;

                        while let Some(chunk_res) = reader.next().await {
                             let mut data_chunk = match chunk_res {
                                Ok(b) => b,
                                Err(e) => {
                                    eprintln!("Failed to read chunk from {}: {:?}", destination_name, e);
                                    return Err(anyhow::Error::from(e));
                                }
                            };

                            while !data_chunk.is_empty() {
                                if current_file_idx >= sizes.len() {
                                    eprintln!("Chunk {}: Received more data than expected based on component sizes", i);
                                    break;
                                }

                                let expected_size = sizes[current_file_idx];
                                let remaining_for_file = expected_size - current_file_written;

                                // Open file if not open
                                if current_file.is_none() {
                                    let filename = &chunk[current_file_idx]; // chunk names are in 'chunk' variable
                                    let path = format!("restored/{}", filename);
                                    let file = tokio::fs::File::create(&path).await.map_err(anyhow::Error::from)?;
                                    current_file = Some(file);
                                }

                                let write_len = std::cmp::min(data_chunk.len() as i64, remaining_for_file) as usize;
                                let write_slice = data_chunk.split_to(write_len);

                                if let Some(file) = current_file.as_mut() {
                                    use tokio::io::AsyncWriteExt;
                                    file.write_all(&write_slice).await.map_err(anyhow::Error::from)?;
                                }

                                current_file_written += write_len as i64;

                                if current_file_written == expected_size {
                                    // Finished this file
                                    current_file = None;
                                    current_file_idx += 1;
                                    current_file_written = 0;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to start download for {}: {:?}", destination_name, e);
                        return Err(anyhow::Error::from(e));
                    }
                }
                println!("Chunk {}: Decomposed into {} files", i, sizes.len());

                // 3. Delete
                println!("Chunk {}: Deleting {}", i, destination_name);
                let delete_result = client
                    .delete_object()
                    .set_bucket(format!("projects/_/buckets/{bucket_id}"))
                    .set_object(&destination_name)
                    .send()
                    .await;

                match delete_result {
                    Ok(_) => {
                         println!("Chunk {}: Deleted", i);
                         Ok(())
                    }
                    Err(e) => {
                        eprintln!("Failed to delete {}: {:?}", destination_name, e);
                        Err(anyhow::Error::from(e))
                    }
                }
            }
        })
        .buffer_unordered(MAX_CONCURRENCY);

    // Drive the stream to completion and check for errors
    let results: Vec<anyhow::Result<()>> = bodies.collect().await;
    
    // Verify all succeeded
    for res in results {
        res?;
    }
    
    Ok(())
}
// [END storage_compose_file]
