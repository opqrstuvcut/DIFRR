use clap::{ArgGroup, Parser};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

mod feature_extract;
use feature_extract::FeatureExtractor;

mod similar_images;
use similar_images::SimilarImages;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(group(ArgGroup::new("compare_options").args(&["comp_dir_path", "comp_self"]).required(true)))]
struct Args {
    #[arg(short = 't', long)]
    target_dir_path: PathBuf,

    #[arg(short = 'c', long)]
    comp_dir_path: Option<Vec<PathBuf>>,

    #[arg(short = 's', long)]
    comp_self: bool,

    #[arg(short, long, default_value_t = 16)]
    batch_size: usize,

    #[arg(short = 'r', long, default_value_t = 0.95)]
    sim_threshold: f32,

    #[arg(short = 'a', long)]
    cache_dir_path: PathBuf,

    #[arg(short = 'o', long)]
    output_dir_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let cache_dir = args.cache_dir_path.as_path();
    let target_feature_extractor = FeatureExtractor::new(cache_dir)?;

    let image_paths = fs::read_dir(args.target_dir_path)?
        .map(|p| p.unwrap().path())
        .collect();
    let target_features = target_feature_extractor.inference(&image_paths, args.batch_size)?;

    let comp_feature_extractor = FeatureExtractor::new(cache_dir)?;
    let mut comp_image_paths: Vec<PathBuf> = Vec::new();

    if let Some(compare_dirs) = args.comp_dir_path {
        for compare_dir in compare_dirs {
            let compare_dir_image_paths: Vec<PathBuf> = fs::read_dir(compare_dir)?
                .map(|p| p.unwrap().path())
                .collect();
            comp_image_paths.extend(compare_dir_image_paths);
        }
    } else {
        comp_image_paths = image_paths.clone();
    };
    let comp_features = comp_feature_extractor.inference(&comp_image_paths, args.batch_size)?;

    let mut comp = SimilarImages::new(args.sim_threshold);
    let res = comp.comp(&target_features, &comp_features);

    let mut sim_idx_set = HashSet::new();
    for i in res {
        sim_idx_set.insert(i);
    }

    if !args.output_dir_path.exists() {
        fs::create_dir(&args.output_dir_path)?;
        println!("{:?} is created.", args.output_dir_path);
    }

    for (i, image_path) in image_paths.iter().enumerate() {
        if !sim_idx_set.contains(&i) {
            fs::copy(
                image_path,
                args.output_dir_path
                    .join(Path::new(image_path).file_name().unwrap()),
            )
            .unwrap();
        }
    }

    Ok(())
}
