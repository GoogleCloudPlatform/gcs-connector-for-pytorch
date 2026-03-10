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

#[derive(Clone, Debug)]
struct ComposedBatch {
    name: String,
    start_idx: usize, // inclusive
    end_idx: usize,   // exclusive
}

pub async fn sample(client: &StorageControl, storage: &Storage, bucket_id: &str, objects: Vec<String>, depth: u32) -> anyhow::Result<()> {
    tokio::fs::create_dir_all("restored").await?;
    const MAX_COMPOSE_BATCH: usize = 32;
    const MAX_CONCURRENCY: usize = 10;

    if depth == 0 || objects.is_empty() {
        return Ok(());
    }

    // 1. Fetch Source Sizes (Concurrent & Ordered)
    println!("Fetching base component sizes...");
    let size_futures = objects.iter().map(|name| {
        let client = client.clone();
        let bucket_id = bucket_id.to_string();
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

    let base_sizes: Vec<i64> = stream::iter(size_futures)
        .buffered(MAX_CONCURRENCY)
        .collect::<Vec<anyhow::Result<i64>>>()
        .await
        .into_iter()
        .collect::<anyhow::Result<Vec<i64>>>()
        .map_err(|e| {
            eprintln!("Failed to fetch base sizes: {:?}", e);
            e
        })?;

    println!("Fetched {} base sizes.", base_sizes.len());

    // 2. Hierarchical Compose
    let mut current_level_batches: Vec<ComposedBatch> = objects.into_iter().enumerate().map(|(i, name)| {
        ComposedBatch {
            name,
            start_idx: i,
            end_idx: i + 1,
        }
    }).collect();

    let original_objects: Vec<String> = current_level_batches.iter().map(|b| b.name.clone()).collect();
    let mut generated_objects_to_delete: Vec<String> = Vec::new();

    for current_depth in 1..=depth {
        if current_level_batches.len() == 1 {
            println!("Reached single object at depth {}, stopping composition.", current_depth - 1);
            break;
        }

        println!("--- Starting Composition Depth {} ---", current_depth);
        let chunks: Vec<_> = current_level_batches.chunks(MAX_COMPOSE_BATCH).enumerate().collect();

        let compose_futures = chunks.iter().map(|(i, chunk)| {
            let client = client.clone();
            let bucket_id = bucket_id.to_string();
            let chunk = chunk.to_vec();
            let destination_name = format!("composed_d{}_b{}", current_depth, i);
            let start_idx = chunk.first().unwrap().start_idx;
            let end_idx = chunk.last().unwrap().end_idx;

            async move {
                let source_objects: Vec<SourceObject> = chunk
                    .iter()
                    .map(|b| SourceObject::new().set_name(&b.name))
                    .collect();

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

                match compose_result {
                    Ok(object) => {
                        println!("Composed {} with size: {}", destination_name, object.size);
                        Ok(ComposedBatch {
                            name: destination_name,
                            start_idx,
                            end_idx,
                        })
                    }
                    Err(e) => {
                        eprintln!("Failed to compose {}: {:?}", destination_name, e);
                        Err(anyhow::Error::from(e))
                    }
                }
            }
        });

        let results: Vec<anyhow::Result<ComposedBatch>> = stream::iter(compose_futures)
            .buffer_unordered(MAX_CONCURRENCY)
            .collect().await;

        let mut level_outputs = Vec::new();
        for res in results {
            let batch = res?;
            generated_objects_to_delete.push(batch.name.clone());
            level_outputs.push(batch);
        }

        // Sort to maintain original object ordering
        level_outputs.sort_by_key(|b| b.start_idx);
        current_level_batches = level_outputs;
    }

    // 3. Download & Demux
    println!("--- Downloading and Decomposing ---");
    let download_futures = current_level_batches.into_iter().map(|batch| {
         let storage = storage.clone();
         let bucket_id = bucket_id.to_string();
         let original_objects = original_objects.clone();
         let base_sizes = base_sizes.clone();

         async move {
             println!("Downloading & Decomposing {}", batch.name);
             let read_result = storage
                 .read_object(&format!("projects/_/buckets/{bucket_id}"), &batch.name)
                 .send()
                 .await;
                 
             match read_result {
                 Ok(mut reader) => {
                     let mut current_file_idx = batch.start_idx;
                     let mut current_file_written = 0;
                     let mut current_file: Option<tokio::fs::File> = None;

                     while let Some(chunk_res) = reader.next().await {
                         let mut data_chunk = match chunk_res {
                             Ok(b) => b,
                             Err(e) => {
                                 eprintln!("Failed to read chunk from {}: {:?}", batch.name, e);
                                 return Err(anyhow::Error::from(e));
                             }
                         };

                         while !data_chunk.is_empty() {
                             if current_file_idx >= batch.end_idx {
                                 eprintln!("Received more data than expected for batch {}", batch.name);
                                 break;
                             }

                             let expected_size = base_sizes[current_file_idx];
                             let remaining_for_file = expected_size - current_file_written;

                             if current_file.is_none() {
                                 let filename = &original_objects[current_file_idx];
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
                                 current_file = None;
                                 current_file_idx += 1;
                                 current_file_written = 0;
                             }
                         }
                     }
                     println!("Decomposed {} into {} files.", batch.name, batch.end_idx - batch.start_idx);
                     Ok(())
                 }
                 Err(e) => {
                     eprintln!("Failed to start download for {}: {:?}", batch.name, e);
                     Err(anyhow::Error::from(e))
                 }
             }
         }
    });

    let download_results: Vec<anyhow::Result<()>> = stream::iter(download_futures)
        .buffer_unordered(MAX_CONCURRENCY)
        .collect().await;

    for res in download_results {
        res?;
    }

    // 4. Delete
    println!("--- Deleting Generated Objects ---");
    let delete_futures = generated_objects_to_delete.into_iter().map(|name| {
        let client = client.clone();
        let bucket_id = bucket_id.to_string();
        async move {
            println!("Deleting {}", name);
            let delete_result = client
                .delete_object()
                .set_bucket(format!("projects/_/buckets/{bucket_id}"))
                .set_object(&name)
                .send()
                .await;
            
            match delete_result {
                Ok(_) => Ok(()),
                Err(e) => {
                    eprintln!("Failed to delete {}: {:?}", name, e);
                    // Just log the error, don't fail the whole process if one deletion fails?
                    // Usually we want to bubble it up.
                    Err(anyhow::Error::from(e))
                }
            }
        }
    });

    let delete_results: Vec<anyhow::Result<()>> = stream::iter(delete_futures)
        .buffer_unordered(MAX_CONCURRENCY)
        .collect().await;

    for res in delete_results {
        res?;
    }

    Ok(())
}
// [END storage_compose_file]
