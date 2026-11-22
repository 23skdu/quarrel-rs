//! Filesystem discovery and name resolution for GGUF model files.
//!
//! [`ModelResolver`] scans a set of roots recursively using `std::fs` only. Model ids are the
//! file stems of `*.gguf` files, which keeps them stable across symlinks and directory layouts.
//!
//! # Sources
//!
//! [`ModelSource`] describes where models normally live: llmfit's cache, llama.cpp, the
//! Hugging Face hub cache, Ollama's model store, or a caller-supplied path. Each source expands
//! into one or more roots via [`ModelSource::candidate_roots`], which takes the home directory and
//! an environment map from the caller rather than reading process-global state.
//!
//! # Name ranking
//!
//! Exact, case-sensitive stems always win. Otherwise every candidate is ranked (best first) by:
//!
//! 1. case-insensitive exact match after normalising separators (`-`, `_`, `.`, `:`, `/`,
//!    whitespace) and, separately, after ignoring separators entirely;
//! 2. token prefix: the query matches complete leading tokens (for example `llama` in
//!    `llama-3-8b`);
//! 3. string prefix: the normalised or separator-free candidate starts with the query;
//! 4. token subsequence: query tokens match distinct candidate tokens in order, preferring
//!    fewer skipped tokens and then an earlier first token;
//! 5. character subsequence: query characters appear in order in the separator-free candidate,
//!    preferring fewer gaps, an earlier first character, then a shorter name.
//!
//! Prefix tiers prefer shorter candidates. Two candidates with the same rank are ambiguous and
//! are rejected instead of being chosen arbitrarily.

use std::{
    collections::{BTreeMap, HashMap},
    ffi::{OsStr, OsString},
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use crate::{CoreError, ModelId};

const GGUF_EXTENSION: &str = "gguf";
const MAX_SCAN_DEPTH: usize = 64;
const QUARREL_MODELS_DIR_ENV: &str = "QUARREL_MODELS_DIR";

/// Read-only environment values used when expanding [`ModelSource`] roots.
///
/// This trait is implemented for `HashMap` and `BTreeMap` whose keys and values are string-like,
/// so a captured process environment can be supplied directly, including in tests. Implement it
/// for a custom map when a different lookup is needed; no environment is read implicitly.
pub trait Environment {
    /// Returns the value of `key`, if it is present.
    fn value(&self, key: &str) -> Option<OsString>;
}

impl<K, V> Environment for HashMap<K, V>
where
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    fn value(&self, key: &str) -> Option<OsString> {
        map_value(self.iter(), key)
    }
}

impl<K, V> Environment for BTreeMap<K, V>
where
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    fn value(&self, key: &str) -> Option<OsString> {
        map_value(self.iter(), key)
    }
}

fn map_value<'a, K, V>(
    mut entries: impl Iterator<Item = (&'a K, &'a V)>,
    key: &str,
) -> Option<OsString>
where
    K: AsRef<OsStr> + 'a,
    V: AsRef<OsStr> + 'a,
{
    entries
        .find(|(candidate, _)| candidate.as_ref() == OsStr::new(key))
        .map(|(_, value)| value.as_ref().to_os_string())
}

/// A place that may contain GGUF models.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSource {
    /// llmfit's model cache at `~/.cache/llmfit`.
    Llmfit,
    /// llama.cpp models at `~/.cache/llama.cpp`, or `LLAMA_CPP_MODELS_DIR` when set.
    LlamaCpp,
    /// Hugging Face hub cache at `~/.cache/huggingface/hub`, or `HF_HOME/hub` when set.
    HuggingFace,
    /// Ollama's model store at `~/.ollama/models`, or `OLLAMA_MODELS` when set.
    Ollama,
    /// A caller-supplied file or directory, used verbatim.
    Explicit(PathBuf),
}

impl ModelSource {
    /// Returns the built-in sources in discovery order.
    #[must_use]
    pub const fn defaults() -> [Self; 4] {
        [
            Self::Llmfit,
            Self::LlamaCpp,
            Self::HuggingFace,
            Self::Ollama,
        ]
    }

    /// Returns candidate roots for this source.
    ///
    /// Both `home` and `env` are supplied by the caller, so expansion is deterministic and does
    /// not consult process-global state. `HF_HOME` names the Hugging Face cache base directory;
    /// its hub data lives in the `hub` child. An explicit source always yields exactly its path.
    #[must_use]
    pub fn candidate_roots<E: Environment>(&self, home: &Path, env: &E) -> Vec<PathBuf> {
        match self {
            Self::Llmfit => vec![home.join(".cache").join("llmfit")],
            Self::LlamaCpp => vec![
                env_root(env, "LLAMA_CPP_MODELS_DIR")
                    .unwrap_or_else(|| home.join(".cache").join("llama.cpp")),
            ],
            Self::HuggingFace => vec![
                env_root(env, "HF_HOME")
                    .map(|base| base.join("hub"))
                    .unwrap_or_else(|| home.join(".cache").join("huggingface").join("hub")),
            ],
            Self::Ollama => vec![
                env_root(env, "OLLAMA_MODELS")
                    .unwrap_or_else(|| home.join(".ollama").join("models")),
            ],
            Self::Explicit(path) => vec![path.clone()],
        }
    }
}

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

    /// Creates a resolver from `sources`, using the current process home and environment.
    ///
    /// Sources are expanded in the order given. An empty `sources` iterator selects all
    /// [`ModelSource::defaults`]; [`ModelSource::Explicit`] paths are used verbatim. The
    /// `QUARREL_MODELS_DIR` root is appended when that variable is set. Use
    /// [`ModelResolver::default`] for a resolver with no roots.
    #[must_use]
    pub fn with_sources(sources: impl IntoIterator<Item = ModelSource>) -> Self {
        let env = process_environment();
        Self::with_sources_from(sources, &home_from(&env), &env)
    }

    /// Creates a resolver over all built-in sources plus `QUARREL_MODELS_DIR`, if set.
    #[must_use]
    pub fn with_default_sources() -> Self {
        Self::with_sources(ModelSource::defaults())
    }

    /// Appends another root to the search path.
    pub fn add_root(&mut self, root: PathBuf) {
        self.roots.push(root);
    }

    /// Adds the roots of `source`, using the current process home and environment.
    ///
    /// This does not add `QUARREL_MODELS_DIR`; [`ModelResolver::with_sources`] handles that
    /// environment-driven default.
    pub fn add_source(&mut self, source: ModelSource) {
        let env = process_environment();
        self.add_source_from(source, &home_from(&env), &env);
    }

    fn with_sources_from<E: Environment>(
        sources: impl IntoIterator<Item = ModelSource>,
        home: &Path,
        env: &E,
    ) -> Self {
        let mut resolver = Self::default();
        let mut sources: Vec<_> = sources.into_iter().collect();
        if sources.is_empty() {
            sources.extend(ModelSource::defaults());
        }
        for source in sources {
            resolver.add_source_from(source, home, env);
        }
        if let Some(dir) = env_root(env, QUARREL_MODELS_DIR_ENV) {
            resolver.add_root(dir);
        }
        resolver
    }

    fn add_source_from<E: Environment>(&mut self, source: ModelSource, home: &Path, env: &E) {
        for root in source.candidate_roots(home, env) {
            self.add_root(root);
        }
    }

    /// Returns all discovered model ids, sorted and de-duplicated.
    ///
    /// Missing roots and entries that disappear during a scan are skipped; existing roots that
    /// cannot be read are reported as a resolver error.
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
    /// Exact, case-sensitive stem matches are tried first. Otherwise candidates are scored as
    /// documented in the [module description](crate::resolver#name-ranking); the best-scoring
    /// candidate is returned. Zero matches or several candidates sharing the best score are
    /// errors.
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

        let mut ranked: Vec<(PathBuf, MatchRank)> = files
            .iter()
            .filter_map(|path| {
                let rank = match_rank(query.as_str(), stem_of(path)?)?;
                Some((path.clone(), rank))
            })
            .collect();
        ranked.sort_by(|(left_path, left_rank), (right_path, right_rank)| {
            left_rank
                .cmp(right_rank)
                .then_with(|| left_path.cmp(right_path))
        });

        let Some((_, best)) = ranked.first() else {
            return Err(resolver_error(format!(
                "no GGUF model matches `{}`",
                query.as_str()
            )));
        };
        let tied: Vec<PathBuf> = ranked
            .iter()
            .take_while(|(_, rank)| rank == best)
            .map(|(path, _)| path.clone())
            .collect();
        select_unique(tied, query.as_str())
    }

    fn model_files(&self) -> Result<Vec<PathBuf>, CoreError> {
        let mut files = Vec::new();
        for root in &self.roots {
            match fs::metadata(root) {
                Ok(metadata) if metadata.is_file() => {
                    if is_gguf(root) {
                        files.push(root.clone());
                    }
                }
                Ok(metadata) if metadata.is_dir() => scan_directory(root, 0, &mut files)?,
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(resolver_error(format!(
                        "cannot access root {}: {error}",
                        root.display()
                    )));
                }
            }
        }
        files.sort();
        files.dedup();
        Ok(files)
    }
}

fn env_root<E: Environment>(env: &E, key: &str) -> Option<PathBuf> {
    env.value(key)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn process_environment() -> HashMap<OsString, OsString> {
    std::env::vars_os().collect()
}

fn home_from<E: Environment>(env: &E) -> PathBuf {
    env_root(env, "HOME")
        .or_else(|| env_root(env, "USERPROFILE"))
        .unwrap_or_default()
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
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(resolver_error(format!(
                    "cannot inspect {}: {error}",
                    path.display()
                )));
            }
        };
        if metadata.is_dir() {
            scan_directory(&path, depth + 1, files)?;
        } else if metadata.is_file() && is_gguf(&path) {
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

/// A deterministic score; smaller tuples match better. Fields are tier-specific.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MatchRank {
    tier: u8,
    skipped: usize,
    start: usize,
    len: usize,
}

impl MatchRank {
    const fn new(tier: u8, skipped: usize, start: usize, len: usize) -> Self {
        Self {
            tier,
            skipped,
            start,
            len,
        }
    }
}

fn match_rank(query: &str, candidate: &str) -> Option<MatchRank> {
    let query_canonical = canonical(query);
    let candidate_canonical = canonical(candidate);
    let query_compact = compact(&query_canonical);
    let candidate_compact = compact(&candidate_canonical);

    if query_canonical == candidate_canonical
        || (!query_compact.is_empty() && query_compact == candidate_compact)
    {
        return Some(MatchRank::new(0, 0, 0, 0));
    }

    if let Some(is_token_prefix) = prefix_boundary(&query_canonical, &candidate_canonical) {
        let tier = if is_token_prefix { 1 } else { 2 };
        return Some(MatchRank::new(
            tier,
            0,
            0,
            candidate_canonical.chars().count(),
        ));
    }
    if !query_compact.is_empty() && candidate_compact.starts_with(&query_compact) {
        return Some(MatchRank::new(2, 0, 0, candidate_compact.chars().count()));
    }

    let query_tokens = tokens(&query_canonical);
    let candidate_tokens = tokens(&candidate_canonical);
    if let Some((skipped, start)) = token_subsequence(&query_tokens, &candidate_tokens) {
        return Some(MatchRank::new(
            3,
            skipped,
            start,
            candidate_canonical.chars().count(),
        ));
    }

    if !query_compact.is_empty()
        && let Some((gaps, start)) = char_subsequence(&query_compact, &candidate_compact)
    {
        return Some(MatchRank::new(
            4,
            gaps,
            start,
            candidate_compact.chars().count(),
        ));
    }
    None
}

fn canonical(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut separator = false;
    for character in value.to_lowercase().chars() {
        if matches!(character, '-' | '_' | '.' | ':' | '/') || character.is_whitespace() {
            if !separator {
                result.push('-');
                separator = true;
            }
        } else {
            result.push(character);
            separator = false;
        }
    }
    result.trim_matches('-').to_owned()
}

fn compact(canonical: &str) -> String {
    canonical
        .chars()
        .filter(|character| *character != '-')
        .collect()
}

fn prefix_boundary(query: &str, candidate: &str) -> Option<bool> {
    if query.is_empty() || !candidate.starts_with(query) {
        return None;
    }
    Some(candidate.chars().nth(query.chars().count()) == Some('-'))
}

fn tokens(canonical: &str) -> Vec<&str> {
    canonical
        .split('-')
        .filter(|token| !token.is_empty())
        .collect()
}

fn token_matches(query: &str, candidate: &str) -> bool {
    candidate.starts_with(query) || query.starts_with(candidate)
}

/// Returns `(skipped_between_matches, first_match_index)` for the best token alignment.
fn token_subsequence(query: &[&str], candidate: &[&str]) -> Option<(usize, usize)> {
    if query.is_empty() || query.len() > candidate.len() {
        return None;
    }
    let mut gaps = vec![vec![None; candidate.len()]; query.len()];
    let mut starts = vec![vec![None; candidate.len()]; query.len()];
    for (index, token) in candidate.iter().enumerate() {
        if token_matches(query[0], token) {
            gaps[0][index] = Some(0);
            starts[0][index] = Some(index);
        }
    }
    for step in 1..query.len() {
        for (index, token) in candidate.iter().enumerate() {
            if !token_matches(query[step], token) {
                continue;
            }
            let mut best_gap = usize::MAX;
            let mut best_start = usize::MAX;
            for previous in 0..index {
                if let (Some(gap), Some(start)) =
                    (gaps[step - 1][previous], starts[step - 1][previous])
                {
                    let candidate_gap = gap + index - previous - 1;
                    if candidate_gap < best_gap || (candidate_gap == best_gap && start < best_start)
                    {
                        best_gap = candidate_gap;
                        best_start = start;
                    }
                }
            }
            if best_gap != usize::MAX {
                gaps[step][index] = Some(best_gap);
                starts[step][index] = Some(best_start);
            }
        }
    }

    let last = query.len() - 1;
    let mut best: Option<(usize, usize)> = None;
    for index in 0..candidate.len() {
        if let (Some(gap), Some(start)) = (gaps[last][index], starts[last][index]) {
            let score = (gap, start);
            if best.is_none_or(|previous| score < previous) {
                best = Some(score);
            }
        }
    }
    best
}

/// Returns `(gaps_between_matches, first_match_index)` using a leftmost greedy alignment.
fn char_subsequence(query: &str, candidate: &str) -> Option<(usize, usize)> {
    let query_chars: Vec<char> = query.chars().collect();
    if query_chars.is_empty() {
        return None;
    }
    let mut matched = 0;
    let mut first = 0;
    let mut gaps = 0;
    let mut previous = None;
    for (index, character) in candidate.chars().enumerate() {
        if character == query_chars[matched] {
            if matched == 0 {
                first = index;
            }
            if let Some(previous_index) = previous {
                gaps += index - previous_index - 1;
            }
            previous = Some(index);
            matched += 1;
            if matched == query_chars.len() {
                break;
            }
        }
    }
    (matched == query_chars.len()).then_some((gaps, first))
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
        collections::HashMap,
        fs,
        path::{Path, PathBuf},
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

    fn env_map(entries: &[(&str, &Path)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.display().to_string()))
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

    #[test]
    fn llmfit_source_discovers_default_root() {
        let home = TempDir::new("llmfit-home");
        let env = HashMap::<String, String>::new();
        let roots = ModelSource::Llmfit.candidate_roots(home.path(), &env);
        assert_eq!(roots, vec![home.path().join(".cache/llmfit")]);

        let model = home.write(".cache/llmfit/qwen-2.5.gguf");
        let resolver = ModelResolver::with_sources_from([ModelSource::Llmfit], home.path(), &env);
        assert_eq!(ids(&resolver), vec!["qwen-2.5"]);
        assert_eq!(resolver.resolve("qwen_2_5").expect("llmfit resolve"), model);
    }

    #[test]
    fn llama_cpp_source_uses_env_override_and_discovers_models() {
        let home = TempDir::new("llama-home");
        let custom = TempDir::new("llama-custom");
        let env = env_map(&[("LLAMA_CPP_MODELS_DIR", custom.path())]);
        let roots = ModelSource::LlamaCpp.candidate_roots(home.path(), &env);
        assert_eq!(roots, vec![custom.path().to_path_buf()]);

        let model = custom.write("ggml/mistral-7b.gguf");
        let resolver = ModelResolver::with_sources_from([ModelSource::LlamaCpp], home.path(), &env);
        assert_eq!(ids(&resolver), vec!["mistral-7b"]);
        assert_eq!(
            resolver.resolve("mistral").expect("llama.cpp resolve"),
            model
        );

        let default_env = HashMap::<String, String>::new();
        assert_eq!(
            ModelSource::LlamaCpp.candidate_roots(home.path(), &default_env),
            vec![home.path().join(".cache/llama.cpp")]
        );
    }

    #[test]
    fn huggingface_source_uses_hf_home_hub_and_discovers_snapshots() {
        let home = TempDir::new("hf-home");
        let cache = TempDir::new("hf-cache");
        let env = env_map(&[("HF_HOME", cache.path())]);
        let roots = ModelSource::HuggingFace.candidate_roots(home.path(), &env);
        assert_eq!(roots, vec![cache.path().join("hub")]);

        let model = cache.write("hub/models--org--model/snapshots/rev/model.gguf");
        let resolver =
            ModelResolver::with_sources_from([ModelSource::HuggingFace], home.path(), &env);
        assert_eq!(ids(&resolver), vec!["model"]);
        assert_eq!(resolver.resolve("MODEL").expect("hf resolve"), model);

        let default_env = HashMap::<String, String>::new();
        assert_eq!(
            ModelSource::HuggingFace.candidate_roots(home.path(), &default_env),
            vec![home.path().join(".cache/huggingface/hub")]
        );
    }

    #[test]
    fn ollama_source_uses_env_override_and_discovers_models() {
        let home = TempDir::new("ollama-home");
        let custom = TempDir::new("ollama-custom");
        let env = env_map(&[("OLLAMA_MODELS", custom.path())]);
        let roots = ModelSource::Ollama.candidate_roots(home.path(), &env);
        assert_eq!(roots, vec![custom.path().to_path_buf()]);

        let model = custom.write("manifests/library/llama3/8b.gguf");
        let resolver = ModelResolver::with_sources_from([ModelSource::Ollama], home.path(), &env);
        assert_eq!(ids(&resolver), vec!["8b"]);
        assert_eq!(resolver.resolve("8B").expect("ollama resolve"), model);

        let default_env = HashMap::<String, String>::new();
        assert_eq!(
            ModelSource::Ollama.candidate_roots(home.path(), &default_env),
            vec![home.path().join(".ollama/models")]
        );
    }

    #[test]
    fn explicit_source_is_used_verbatim_and_add_source_extends_search() {
        let home = TempDir::new("explicit-home");
        let explicit = TempDir::new("explicit-models");
        let env = HashMap::<String, String>::new();
        let roots =
            ModelSource::Explicit(explicit.path().to_path_buf()).candidate_roots(home.path(), &env);
        assert_eq!(roots, vec![explicit.path().to_path_buf()]);

        let model = explicit.write("nested/deepseek.gguf");
        let mut resolver = ModelResolver::new(Vec::new());
        resolver.add_source_from(
            ModelSource::Explicit(explicit.path().to_path_buf()),
            home.path(),
            &env,
        );
        assert_eq!(ids(&resolver), vec!["deepseek"]);
        assert_eq!(
            resolver.resolve("deepseek").expect("explicit resolve"),
            model
        );
    }

    #[test]
    fn with_sources_appends_quarrel_models_dir_and_empty_uses_defaults() {
        let home = TempDir::new("sources-home");
        let extra = TempDir::new("quarrel-models");
        let explicit = TempDir::new("sources-explicit");
        home.write(".cache/llmfit/default-a.gguf");
        home.write(".cache/llama.cpp/default-b.gguf");
        home.write(".cache/huggingface/hub/models--o--m/snapshots/rev/default-c.gguf");
        home.write(".ollama/models/default-d.gguf");
        extra.write("extra.gguf");
        explicit.write("explicit.gguf");

        let env = env_map(&[("QUARREL_MODELS_DIR", extra.path())]);
        let default_resolver = ModelResolver::with_sources_from(
            Vec::new(),
            home.path(),
            &HashMap::<String, String>::new(),
        );
        assert_eq!(
            ids(&default_resolver),
            vec!["default-a", "default-b", "default-c", "default-d"]
        );

        let resolver = ModelResolver::with_sources_from(
            [ModelSource::Explicit(explicit.path().to_path_buf())],
            home.path(),
            &env,
        );
        assert_eq!(ids(&resolver), vec!["explicit", "extra"]);
    }

    #[test]
    fn fuzzy_matching_ranks_prefix_token_and_subsequence_candidates() {
        let temp = TempDir::new("fuzzy");
        let short = temp.write("llama-3.gguf");
        let long = temp.write("llama-3-8b.gguf");
        let close = temp.write("mistral-7b-instruct.gguf");
        let far = temp.write("mistral-8b-alpha-instruct.gguf");
        let separated = temp.write("Phi_3.2-mini.gguf");
        let resolver = ModelResolver::new(vec![temp.path().to_path_buf()]);

        assert_eq!(resolver.resolve("llama").expect("shortest prefix"), short);
        assert_eq!(resolver.resolve("LLAMA3").expect("compact exact"), short);
        assert_eq!(resolver.resolve("llama38b").expect("compact prefix"), long);
        assert_eq!(
            resolver.resolve("mis-ins").expect("token subsequence"),
            close
        );
        assert_eq!(
            resolver.resolve("phi-3-2").expect("normalised exact"),
            separated
        );
        assert_ne!(long, short);
        assert_ne!(far, close);
    }

    #[test]
    fn fuzzy_ties_are_rejected_when_the_best_rank_is_shared() {
        let temp = TempDir::new("fuzzy-tie");
        temp.write("mistral-7b-instruct.gguf");
        temp.write("mistral-8b-instruct.gguf");
        let resolver = ModelResolver::new(vec![temp.path().to_path_buf()]);
        let error = resolver
            .resolve("mis-ins")
            .expect_err("equally ranked candidates are ambiguous");
        assert!(matches!(error, CoreError::Resolver(_)));
        assert!(error.to_string().contains("ambiguous model `mis-ins`"));
    }

    #[test]
    fn missing_model_reports_a_resolver_error() {
        let temp = TempDir::new("missing-model");
        temp.write("mistral-7b-instruct.gguf");
        let resolver = ModelResolver::new(vec![temp.path().to_path_buf()]);
        let error = resolver
            .resolve("not-a-model")
            .expect_err("no candidate matches");
        assert_eq!(
            error,
            CoreError::Resolver("no GGUF model matches `not-a-model`".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_existing_root_is_a_resolver_error() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new("unreadable");
        let root = temp
            .write("models.gguf")
            .parent()
            .expect("root")
            .to_path_buf();
        let original = fs::metadata(&root).expect("root metadata").permissions();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o000))
            .expect("remove read permission");
        let readable = fs::read_dir(&root).is_ok();
        let resolver = ModelResolver::new(vec![root.clone()]);
        let result = resolver.discover();
        fs::set_permissions(&root, original).expect("restore permission");
        if !readable {
            assert!(matches!(result, Err(CoreError::Resolver(_))));
        }
    }
}
