fn main() {
    if std::env::var("CARGO_FEATURE_INCOMPLETE_REXL").is_ok() {
        return;
    }

    println!("cargo:rustc-link-lib=dylib=hexl_wrapper");
    println!("cargo:rustc-link-search=native=.");
    println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib");
}
