use std::time::Instant;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

use incdbscan_rs::engine::IncrementalDbscan;

/// Generate L2-normalized random vectors resembling real text embeddings
/// (OpenAI, Gemini style): each component ~ N(0,1), then normalized to unit length.
///
/// Points are grouped into clusters to simulate real article embeddings where
/// articles about similar topics have nearby embeddings.
fn generate_clustered_embeddings(
    rng: &mut StdRng,
    count: usize,
    dims: usize,
    num_clusters: usize,
    noise_scale: f64,
) -> Vec<Vec<f64>> {
    // Generate cluster centers (random unit vectors)
    let centers: Vec<Vec<f64>> = (0..num_clusters)
        .map(|_| {
            let v: Vec<f64> = (0..dims).map(|_| rng.sample(StandardNormal)).collect();
            normalize(&v)
        })
        .collect();

    // Generate points around cluster centers
    (0..count)
        .map(|_| {
            let center = &centers[rng.gen_range(0..num_clusters)];
            let noisy: Vec<f64> = center
                .iter()
                .map(|&c| c + noise_scale * rng.sample::<f64, _>(StandardNormal))
                .collect();
            normalize(&noisy)
        })
        .collect()
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn main() {
    let dims = 996;
    let eps = 1.2;
    let min_pts: u32 = 5;
    let p = 2.0;
    let num_batches = 10;
    let batch_size = 1500;
    let num_clusters = 50;
    // noise_scale calibrated so intra-cluster distances are ~0.5-1.0 (within eps=1.2)
    let noise_scale = 0.02;

    let mut rng = StdRng::seed_from_u64(42);

    // Pre-generate all batches
    let batches: Vec<Vec<Vec<f64>>> = (0..num_batches)
        .map(|_| generate_clustered_embeddings(&mut rng, batch_size, dims, num_clusters, noise_scale))
        .collect();

    println!();
    println!("IncrementalDBSCAN Batch Insertion Benchmark");
    println!("============================================");
    println!(
        "dims={}, eps={}, min_pts={}, p={}, clusters={}",
        dims, eps, min_pts, p, num_clusters
    );
    println!(
        "batches={}, points_per_batch={}",
        num_batches, batch_size
    );
    println!("{:-<60}", "");
    println!(
        "{:<8} {:>14} {:>14} {:>14}",
        "Batch", "Total Points", "Batch (s)", "Cumul (s)"
    );
    println!("{:-<60}", "");

    let mut db = IncrementalDbscan::new(eps, min_pts, p);
    let overall_start = Instant::now();

    for (i, batch) in batches.iter().enumerate() {
        let batch_start = Instant::now();
        for point in batch {
            db.insert(point);
        }
        let batch_elapsed = batch_start.elapsed().as_secs_f64();
        let cumul_elapsed = overall_start.elapsed().as_secs_f64();

        println!(
            "{:<8} {:>14} {:>14.3} {:>14.3}",
            i + 1,
            (i + 1) * batch_size,
            batch_elapsed,
            cumul_elapsed
        );
    }

    let total = overall_start.elapsed();
    println!("{:-<60}", "");
    println!("Total: {:.3}s", total.as_secs_f64());
    println!();
}
