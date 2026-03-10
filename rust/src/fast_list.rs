use crate::range_splitter::RangeSplitter;
use async_channel::{Receiver, Sender};
use google_cloud_storage::client::StorageControl;
use google_cloud_storage::model::Object;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::task::JoinHandle;

pub struct FastListConfig {
    pub bucket: String,
    pub prefix: String,
    pub max_parallelism: usize,
    pub skip_compose: bool,
}

pub async fn fast_list(
    client: StorageControl, 
    config: FastListConfig,
) -> anyhow::Result<Vec<Object>> {
    let (work_tx, work_rx) = async_channel::unbounded::<(String, String)>();
    let (res_tx, res_rx) = async_channel::unbounded::<Vec<Object>>();
    let idle_workers = Arc::new(AtomicUsize::new(0));

    // Seed the initial work block (the entire namespace)
    work_tx.send((String::new(), String::new())).await?;

    let mut handles: Vec<JoinHandle<anyhow::Result<()>>> = Vec::new();

    for _worker_id in 0..config.max_parallelism {
        let client = client.clone();
        let work_tx = work_tx.clone();
        let work_rx = work_rx.clone();
        let res_tx = res_tx.clone();
        let idle_workers = Arc::clone(&idle_workers);
        let bucket = config.bucket.clone();
        let prefix = config.prefix.clone();
        let skip_compose = config.skip_compose;
        let num_workers = config.max_parallelism;

        let handle = tokio::spawn(async move {
            // Give workers a local range_splitter initialized with a basic alphabet
            let mut splitter = RangeSplitter::new("ab");

            loop {
                // If the channel is empty, check if we should terminate
                if work_rx.is_empty() {
                    let currently_idle = idle_workers.fetch_add(1, Ordering::SeqCst) + 1;
                    
                    if currently_idle == num_workers {
                        // All workers are idle waiting for work, and the queue is empty.
                        // We are done! Broadcast termination by closing the channel.
                        work_tx.close();
                    }
                    
                    // Wait for work or channel closed
                    match work_rx.recv().await {
                        Ok((start_range, end_range)) => {
                            idle_workers.fetch_sub(1, Ordering::SeqCst);
                            
                            // Process this range
                            process_range(
                                &client,
                                &bucket,
                                &prefix,
                                start_range,
                                end_range,
                                skip_compose,
                                &mut splitter,
                                &work_rx,
                                &work_tx,
                                &res_tx,
                                &idle_workers,
                            ).await?;
                        }
                        Err(_) => {
                            // Channel closed, terminate worker
                            break;
                        }
                    }
                } else {
                    // Work is available immediately
                    if let Ok((start_range, end_range)) = work_rx.recv().await {
                        process_range(
                            &client,
                            &bucket,
                            &prefix,
                            start_range,
                            end_range,
                            skip_compose,
                            &mut splitter,
                            &work_rx,
                            &work_tx,
                            &res_tx,
                            &idle_workers,
                        ).await?;
                    }
                }
            }
            Ok(())
        });
        handles.push(handle);
    }

    // Drop the sender halves held by the main thread so the receiver can close
    drop(work_tx);
    drop(res_tx);

    let mut final_results = Vec::new();
    while let Ok(mut batch) = res_rx.recv().await {
        final_results.append(&mut batch);
    }

    // Wait for all workers to cleanly shut down and check for any GCS errors
    for handle in handles {
        match handle.await {
            Ok(result) => result?, // unwrap interior Result
            Err(e) => anyhow::bail!("Worker task panicked: {}", e),
        }
    }

    // Deduplicate results globally. 
    // Manual pagination relies on overlapping `lexicographical_start` ranges which produces overlaps.
    // Python natively relies on a global `set()` to drop these page-continuation overlaps.
    final_results.sort_by(|a, b| a.name.cmp(&b.name));
    final_results.dedup_by(|a, b| a.name == b.name);

    Ok(final_results)
}

async fn process_range(
    client: &StorageControl,
    bucket: &str,
    prefix: &str,
    mut start_range: String,
    mut end_range: String,
    skip_compose: bool,
    splitter: &mut RangeSplitter,
    work_rx: &Receiver<(String, String)>,
    work_tx: &Sender<(String, String)>,
    res_tx: &Sender<Vec<Object>>,
    idle_workers: &AtomicUsize,
) -> anyhow::Result<()> {
    const MAX_RESULTS: i32 = 5000;
    let parent = format!("projects/_/buckets/{}", bucket);

    loop {
        // Field Masking optimization: only ask GCS for exactly the JSON fields we strictly need!
        let mut req = client.list_objects()
            .set_parent(&parent)
            .set_page_size(MAX_RESULTS)
            .set_read_mask(google_cloud_wkt::FieldMask::default().set_paths(
                ["name", "size", "storage_class"]
            ));

        if !prefix.is_empty() {
            req = req.set_prefix(prefix.to_string())
                     .set_lexicographic_start(format!("{}{}", prefix, start_range));
            if !end_range.is_empty() {
                req = req.set_lexicographic_end(format!("{}{}", prefix, end_range));
            }
        } else {
            req = req.set_lexicographic_start(start_range.clone());
            if !end_range.is_empty() {
                req = req.set_lexicographic_end(end_range.clone());
            }
        }

        let result = req.send().await?;
        
        let mut items = result.objects;
        let num_returned = items.len() as i32;

        if num_returned > 0 {
            // Extract the last name to advance the start_offset for the next page
            let last_name = items.last().unwrap().name.clone();
            start_range = if !prefix.is_empty() && last_name.starts_with(prefix) {
                last_name[prefix.len()..].to_string()
            } else {
                last_name
            };

            // Filter skipped files, directory markers, and allow only STANDARD class
            items.retain(|blob| {
                let is_dir = blob.name.ends_with('/');
                let is_compose = skip_compose && blob.name.starts_with("composed_");
                let is_standard = blob.storage_class == "STANDARD";
                !is_dir && !is_compose && is_standard
            });
            res_tx.send(items).await?;
        }

        // If we received fewer items than requested, we've exhausted this specific range.
        if num_returned < MAX_RESULTS || result.next_page_token.is_empty() {
            break;
        }

        // --- Work Stealing Check ---
        // If there are idle workers AND the queue is empty, we must split our remaining work.
        if work_rx.is_empty() && idle_workers.load(Ordering::SeqCst) > 0 {
            let num_stealers = 1; // Basic stealing: give away half. (Could scale with idle count)
            let mut split_points = splitter.split_range(&start_range, &end_range, num_stealers);
            
            if !split_points.is_empty() {
                let steal_range_start = split_points.remove(0);
                
                // Give the upper half to someone else
                let steal_range = (steal_range_start.clone(), end_range.clone());
                let _ = work_tx.send(steal_range).await;
                
                // Keep the lower half for ourselves
                end_range = steal_range_start;
            }
        }
    }

    Ok(())
}
