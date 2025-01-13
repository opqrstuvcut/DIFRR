use std::cmp::min;
use std::collections::HashSet;

use ndarray::s;

#[derive(Debug)]
pub struct SimilarImages {
    threshold: f32,
    duplicated_images_idx_set: HashSet<usize>,
}

impl SimilarImages {
    pub fn new(threshold: f32) -> SimilarImages {
        SimilarImages {
            threshold,
            duplicated_images_idx_set: HashSet::new(),
        }
    }

    pub fn comp(
        &mut self,
        features1: &ndarray::Array2<f32>,
        features2: &ndarray::Array2<f32>,
    ) -> &HashSet<usize> {
        let chunk_size = 10000;
        if features1 == features2 {
            for feature1_chunk_idx in 0..(features1.shape()[0] + chunk_size - 1) / chunk_size {
                let feature1_chunk = features1.slice(s![
                    feature1_chunk_idx * chunk_size
                        ..min((feature1_chunk_idx + 1) * chunk_size, features1.shape()[0]),
                    ..
                ]);
                for feature2_chunk_idx in
                    feature1_chunk_idx..(features2.shape()[0] + chunk_size - 1) / chunk_size
                {
                    let feature2_start = feature2_chunk_idx * chunk_size;
                    let feature2_chunk = features2.slice(s![
                        feature2_start..min(feature2_start + chunk_size, features2.shape()[0]),
                        ..
                    ]);
                    let similarities = feature1_chunk.dot(&feature2_chunk.t());
                    let indices = if feature1_chunk_idx == feature2_chunk_idx {
                        similarities
                            .indexed_iter()
                            .filter(|((i, j), &sim)| i < j && sim > self.threshold)
                            .map(|((_, index), _)| index)
                            .collect::<Vec<_>>()
                    } else {
                        similarities
                            .indexed_iter()
                            .filter(|(_, &sim)| sim > self.threshold)
                            .map(|((_, index), _)| index)
                            .collect::<Vec<_>>()
                    };

                    for i in indices.iter() {
                        self.duplicated_images_idx_set.insert(*i + feature2_start);
                    }
                }
            }
        } else {
            for feature1_chunk_idx in 0..(features1.shape()[0] + chunk_size - 1) / chunk_size {
                let feature1_chunk = features1.slice(s![
                    feature1_chunk_idx * chunk_size..(feature1_chunk_idx + 1) * chunk_size,
                    ..
                ]);
                for feature2_chunk_idx in 0..(features2.shape()[0] + chunk_size - 1) / chunk_size {
                    let feature2_start = feature2_chunk_idx * chunk_size;
                    let feature2_chunk =
                        features2.slice(s![feature2_start..feature2_start + chunk_size, ..]);

                    let similarities = feature1_chunk.dot(&feature2_chunk.t());
                    let indices: Vec<_> = similarities
                        .indexed_iter()
                        .filter(|(_, &sim)| sim > self.threshold)
                        .map(|((_, index), _)| index)
                        .collect();
                    for i in indices.iter() {
                        self.duplicated_images_idx_set.insert(*i + feature2_start);
                    }
                }
            }
        }

        &self.duplicated_images_idx_set
    }
}
