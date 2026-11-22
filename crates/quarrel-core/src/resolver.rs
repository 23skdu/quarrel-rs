//! Filesystem discovery and name resolution for GGUF model files.
//!
//! [`ModelResolver`] scans a set of roots recursively using `std::fs` only. Model ids are the
//! file stems of `*.gguf` files, which keeps them stable across symlinks and directory layouts.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{CoreError, ModelId};

const GGUF_EXTENSION: &str = "gguf";
const MAX_SCAN_DEPTH: usize = 64;

/// Searches one or more roots for GGUF models by file stem.
#[derive(Clone, Debug, Default)]
pub struct ModelResolver {
    roots: Vec<PathBuf>,
}

impl ModelResolver {
    /// Creates a resolver over `roots`, preserving their order.
    #[must_use]
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// Appends another root to the search path.
    pub fn add_root(&mut self, root: PathBuf) {
        self.roots.push(root);
    }

    /// Returns all discovered model ids, sorted and de-duplicated.
    ///
    /// Unreadable directories and non-existent roots are skipped only when the root does not
    /// exist; an existing directory that cannot be read is reported as a resolver error.
    #[must_use = "returns an error when a root cannot be read"]
    pub fn discover(&self) -> Result<Vec<ModelId>, CoreError> {
        let mut ids = Vec::new();
        for path in self.model_files()? {
            if let Some(stem) = stem_of(&path)
                && !stem.is_empty()
                && !ids.iter().any(|id: &ModelId| id.as_str() == stem)
            {
                ids.push(ModelId::new(stem)?);
            }
        }
        ids.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        Ok(ids)
    }

    /// Resolves a model id to its GGUF file path.
    ///
    /// Exact, case-sensitive stem matches are tried first. If none exist, a single
    /// case-insensitive substring match is accepted. Zero matches or more than one match at
    /// either stage is an error.
    #[must_use = "returns an error for missing or ambiguous model names"]
    pub fn resolve(&self, query: &str) -> Result<PathBuf, CoreError> {
        let query = ModelId::new(query)?;
        let files = self.model_files()?;

        let exact: Vec<PathBuf> = files
            .iter()
            .filter(|path| stem_of(path) == Some(query.as_str()))
            .cloned()
            .collect();
        if !exact.is_empty() {
            return select_unique(exact, query.as_str());
        }

        let needle = query.as_str().to_lowercase();
        let partial: Vec<PathBuf> = files
            .iter()
            .filter(|path| stem_of(path).is_some_and(|stem| stem.to_lowercase().contains(&needle)))
            .cloned()
            .collect();
        select_unique(partial, query.as_str())
    }

    fn model_files(&self) -> Result<Vec<PathBuf>, CoreError> {
        let mut files = Vec::new();
        for root in &self.roots {
            if !root.exists() {
                continue;
            }
            if root.is_file() {
                if is_gguf(root) {
                    files.push(root.clone());
                }
                continue;
            }
            scan_directory(root, 0, &mut files)?;
        }
        files.sort();
        files.dedup();
        Ok(files)
    }
}

fn scan_directory(
    directory: &Path,
    depth: usize,
    files: &mut Vec<PathBuf>,
) -> Result<(), CoreError> {
    if depth > MAX_SCAN_DEPTH {
        return Ok(());
    }
    let entries = fs::read_dir(directory).map_err(|error| {
        resolver_error(format!(
            "cannot read directory {}: {error}",
            directory.display()
        ))
    })?;

    let mut sorted = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            resolver_error(format!(
                "cannot read entry in {}: {error}",
                directory.display()
            ))
        })?;
        sorted.push(entry);
    }
    sorted.sort_by_key(|entry| entry.path());

    for entry in sorted {
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| {
            resolver_error(format!("cannot inspect {}: {error}", path.display()))
        })?;
        if file_type.is_dir() {
            scan_directory(&path, depth + 1, files)?;
        } else if file_type.is_file() && is_gguf(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_gguf(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()) == Some(GGUF_EXTENSION)
}

fn stem_of(path: &Path) -> Option<&str> {
    path.file_stem().and_then(|stem| stem.to_str())
}

fn select_unique(candidates: Vec<PathBuf>, query: &str) -> Result<PathBuf, CoreError> {
    let mut candidates = candidates.into_iter();
    let first = candidates
        .next()
        .ok_or_else(|| resolver_error(format!("no GGUF model matches `{query}`")))?;
    if let Some(second) = candidates.next() {
        let mut paths = vec![first, second];
        paths.extend(candidates);
        return Err(resolver_error(format!(
            "ambiguous model `{query}`: {}",
            describe_paths(&paths)
        )));
    }
    Ok(first)
}

fn describe_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn resolver_error(message: impl Into<String>) -> CoreError {
    CoreError::Resolver(message.into())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "quarrel-core-resolver-{tag}-{}-{nanos}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temporary directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn write(&self, relative: &str) -> PathBuf {
            let target = self.path.join(relative);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).expect("create parent directory");
            }
            fs::write(&target, b"fixture").expect("write fixture");
            target
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn ids(resolver: &ModelResolver) -> Vec<String> {
        resolver
            .discover()
            .expect("discover models")
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect()
    }

    #[test]
    fn discover_is_recursive_sorted_and_ignores_other_files() {
        let temp = TempDir::new("discover");
        temp.write("zeta.gguf");
        temp.write("nested/alpha.gguf");
        temp.write("notes.txt");
        let resolver = ModelResolver::new(vec![temp.path().to_path_buf()]);
        assert_eq!(ids(&resolver), vec!["alpha", "zeta"]);
    }

    #[test]
    fn discover_deduplicates_stems_across_roots() {
        let first = TempDir::new("dup-a");
        let second = TempDir::new("dup-b");
        first.write("same.gguf");
        second.write("same.gguf");
        let resolver = ModelResolver::new(vec![
            first.path().to_path_buf(),
            second.path().to_path_buf(),
        ]);
        assert_eq!(ids(&resolver), vec!["same"]);
        assert!(matches!(
            resolver.resolve("same"),
            Err(CoreError::Resolver(_))
        ));
    }

    #[test]
    fn resolve_prefers_exact_case_sensitive_stem() {
        let upper = TempDir::new("exact-a");
        let lower = TempDir::new("exact-b");
        let upper_path = upper.write("Llama-3.gguf");
        let lower_path = lower.write("llama-3.gguf");
        let resolver =
            ModelResolver::new(vec![upper.path().to_path_buf(), lower.path().to_path_buf()]);

        assert_eq!(
            resolver.resolve("Llama-3").expect("exact upper"),
            upper_path
        );
        assert_eq!(
            resolver.resolve("llama-3").expect("exact lower"),
            lower_path
        );
        assert!(matches!(
            resolver.resolve("LLAMA"),
            Err(CoreError::Resolver(_))
        ));
    }

    #[test]
    fn resolve_accepts_unique_case_insensitive_substring() {
        let temp = TempDir::new("substring");
        let model = temp.write("mistral-7b-instruct.gguf");
        let resolver = ModelResolver::new(vec![temp.path().to_path_buf()]);
        assert_eq!(resolver.resolve("MISTRAL").expect("substring"), model);
        assert!(matches!(
            resolver.resolve("does-not-exist"),
            Err(CoreError::Resolver(_))
        ));
        assert!(matches!(
            resolver.resolve(""),
            Err(CoreError::InvalidModelId(_))
        ));
    }

    #[test]
    fn add_root_extends_the_search_path() {
        let first = TempDir::new("add-a");
        let second = TempDir::new("add-b");
        first.write("a-model.gguf");
        second.write("b-model.gguf");
        let mut resolver = ModelResolver::new(vec![first.path().to_path_buf()]);
        assert_eq!(ids(&resolver), vec!["a-model"]);
        resolver.add_root(second.path().to_path_buf());
        assert_eq!(ids(&resolver), vec!["a-model", "b-model"]);
    }

    #[test]
    fn missing_roots_are_skipped() {
        let temp = TempDir::new("missing");
        let missing = temp.path().join("not-created");
        let resolver = ModelResolver::new(vec![missing]);
        assert!(resolver.discover().expect("empty discovery").is_empty());
        assert!(matches!(
            resolver.resolve("anything"),
            Err(CoreError::Resolver(_))
        ));
    }

    #[test]
    fn resolver_default_is_empty() {
        let resolver = ModelResolver::default();
        assert!(resolver.discover().expect("empty resolver").is_empty());
    }
}
