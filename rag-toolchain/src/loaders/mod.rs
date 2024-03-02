/// # Loaders
/// This modules aims to provide some easy methods of loading in
/// input data to you Gen AI workflow.
mod single_file_loader;
mod traits;

pub use single_file_loader::SingleFileSource;
pub use traits::AsyncLoadSource;
pub use traits::LoadSource;
