use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use image::{GenericImageView, ImageError};
use indicatif::ProgressBar;
use ndarray::{s, Array, Array2, Array4, Axis};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    Error,
};

pub struct FeatureExtractor {
    model: Session,
    cache_dir: Option<PathBuf>,
}

impl FeatureExtractor {
    const RGB_MEAN: [f32; 3] = [0.4850, 0.4560, 0.4060];
    const RGB_STD: [f32; 3] = [0.2290, 0.2240, 0.2250];
    const IMAGE_SIZE: [usize; 2] = [256, 256];
    const FEATURE_SIZE: usize = 1280;

    pub fn new(cache_dir: &Path) -> Result<Self, Error> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file("./assets/mobilenetv4_conv_medium.e500_r256_in1k_features.onnx")?;

        Ok(FeatureExtractor {
            model,
            cache_dir: Some(cache_dir.to_path_buf()),
        })
    }

    fn load_images(&self, image_paths: &[PathBuf]) -> Result<Array4<f32>, ImageError> {
        let mut inputs = Array::zeros((
            image_paths.len(),
            3,
            FeatureExtractor::IMAGE_SIZE[0],
            FeatureExtractor::IMAGE_SIZE[1],
        ));

        let results: Result<Vec<_>, ImageError> = image_paths
            .par_iter()
            .enumerate()
            .map(|(i, image_path)| -> Result<_, ImageError> {
                let image = image::open(image_path)?.resize_exact(
                    FeatureExtractor::IMAGE_SIZE[0].try_into().unwrap(),
                    FeatureExtractor::IMAGE_SIZE[1].try_into().unwrap(),
                    image::imageops::FilterType::Nearest,
                );
                let mut image_array = Array::zeros((
                    3,
                    FeatureExtractor::IMAGE_SIZE[0],
                    FeatureExtractor::IMAGE_SIZE[1],
                ));
                for pixel in image.pixels() {
                    let x = pixel.0 as usize;
                    let y = pixel.1 as usize;
                    let [r, g, b, _] = pixel.2 .0;
                    image_array[[0, y, x]] = ((r as f32) / 255.0 - FeatureExtractor::RGB_MEAN[0])
                        / FeatureExtractor::RGB_STD[0];
                    image_array[[1, y, x]] = ((g as f32) / 255.0 - FeatureExtractor::RGB_MEAN[1])
                        / FeatureExtractor::RGB_STD[1];
                    image_array[[2, y, x]] = ((b as f32) / 255.0 - FeatureExtractor::RGB_MEAN[2])
                        / FeatureExtractor::RGB_STD[2];
                }
                Ok((i, image_array))
            })
            .collect();

        for (i, local_array) in results? {
            inputs.slice_mut(s![i, .., .., ..]).assign(&local_array);
        }

        Ok(inputs)
    }

    pub fn inference(
        &self,
        image_paths: &Vec<PathBuf>,
        batch_size: usize,
    ) -> anyhow::Result<Array2<f32>> {
        let mut features =
            ndarray::Array2::<f32>::zeros((image_paths.len(), FeatureExtractor::FEATURE_SIZE));

        let mut cached_features: ndarray::Array2<f32> = ndarray::Array2::zeros((0, 0));
        let mut cached_files: Vec<String> = vec![];
        if let Some(path) = &self.cache_dir {
            if path.exists() {
                let mut buffer = Vec::new();
                File::open(path.join("features.bin"))?.read_to_end(&mut buffer)?;
                cached_features = bincode::deserialize::<Array2<f32>>(&buffer)?;

                buffer.clear();

                File::open(path.join("image_paths.bin"))?.read_to_end(&mut buffer)?;
                cached_files = bincode::deserialize::<Vec<String>>(&buffer)?;

                println!(
                    "{} features are loaded from cache.",
                    cached_features.shape()[0]
                );
            }
        }

        let mut feature_idxs: Vec<usize> = vec![];
        let calc_image_paths: &Vec<PathBuf>;
        let mut target_image_paths: Vec<PathBuf> = vec![];
        if !cached_files.is_empty() {
            let cached_files_idx_map: HashMap<_, usize> = cached_files
                .iter()
                .enumerate()
                .map(|(idx, path)| (Path::new(path.as_str()).file_name().unwrap(), idx))
                .collect();

            for (i, image_path) in image_paths.iter().enumerate() {
                match cached_files_idx_map.get(Path::new(image_path).file_name().unwrap()) {
                    Some(idx) => {
                        features
                            .slice_mut(s![i, ..])
                            .assign(&cached_features.slice(s![*idx, ..]));
                    }
                    None => {
                        target_image_paths.push(image_path.to_path_buf());
                        feature_idxs.push(i);
                    }
                }
            }

            calc_image_paths = &target_image_paths;
        } else {
            feature_idxs = (0..image_paths.len()).collect();
            calc_image_paths = image_paths;
        }

        let bar = ProgressBar::new(((calc_image_paths.len() + batch_size - 1) / batch_size) as u64);
        for (batch_image_paths, batch_feature_idxs) in calc_image_paths
            .chunks(batch_size)
            .zip(feature_idxs.chunks(batch_size))
        {
            let input = self.load_images(batch_image_paths)?;
            let output = self.model.run(ort::inputs!["input" => input]?)?;
            let feature = output["output"].try_extract_tensor::<f32>()?.to_owned();

            let normalized = feature
                .mapv(|x| x * x)
                .sum_axis(ndarray::Axis(1))
                .mapv(|x| x.sqrt())
                .insert_axis(Axis(1));

            let result: Array2<f32> = (&feature / &normalized)
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()?;

            for (i, &feature_idx) in batch_feature_idxs.iter().enumerate() {
                features
                    .slice_mut(s![feature_idx, ..])
                    .assign(&result.slice(s![i, ..]));
            }
            bar.inc(1);
        }
        bar.finish();

        if let Some(path) = &self.cache_dir {
            if !path.exists() {
                fs::create_dir(path)?;
            }
            let mut feature_file = File::create(path.join("features.bin"))?;
            let encoded: Vec<u8> = bincode::serialize(&features).unwrap();
            feature_file.write_all(&encoded)?;

            let mut image_paths_file = File::create(path.join("image_paths.bin"))?;
            let encoded: Vec<u8> = bincode::serialize(&image_paths).unwrap();
            image_paths_file.write_all(&encoded)?;
        }

        Ok(features)
    }
}
