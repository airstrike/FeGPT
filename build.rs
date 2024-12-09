use glob;

fn main() {
    // Get the cargo manifest dir (project root)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir");

    // For release builds, construct path to the release libtorch
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let torch_path = format!(
        "{manifest_dir}/target/{profile}/build/torch-sys-*/out/libtorch/libtorch/lib",
        manifest_dir = manifest_dir,
        profile = profile
    );

    // Handle glob expansion to get the actual path
    let torch_paths = glob::glob(&torch_path)
        .expect("Failed to read glob pattern")
        .next()
        .expect("No torch path found")
        .expect("Invalid torch path");

    // Convert path to string
    let torch_path = torch_paths.to_str().expect("Invalid path string");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", torch_path);
    // println!("cargo:warning=Setting RPATH to: {}", torch_path);
}
