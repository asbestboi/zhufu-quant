//! The one error type of the core.
//!
//! Every way the core can fail on bad input is an `EngineError`. The Python bindings in
//! `lib.rs` turn it into a `ValueError` with the same message the Python core raises.

use std::fmt;

/// Bad input: an invalid order, a malformed table, an impossible split ratio.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError {
    pub message: String,
}

impl EngineError {
    /// An error with `message`. `impl Into<String>` accepts a `&str` or a `String`.
    pub fn invalid(message: impl Into<String>) -> EngineError {
        EngineError {
            message: message.into(),
        }
    }
}

// Implementing the standard `Display` trait is what lets an error be printed with `{}`;
// implementing `std::error::Error` makes it a "proper" error other code can work with.
impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EngineError {}
