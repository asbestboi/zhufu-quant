//! Small numeric helpers that copy Python's behaviour exactly.
//!
//! The Rust core must give the *same* floating-point results as the Python core, bit for
//! bit. Most arithmetic (`+`, `*`, `/`, `floor`, ...) is IEEE 754 in both languages and
//! needs no help. Python's built-in `max()` and `min()` are different from Rust's
//! `f64::max` and `f64::min` when a NaN is involved, so they get their own functions here.

/// Python's `max(a, b)`: `b` if `b > a`, otherwise `a` (so a NaN in `b` loses).
///
/// Rust's `f64::max` ignores NaN on either side, which would give a different answer.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: `b` if `b < a`, otherwise `a`.
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python's built-in `sum()` of floats, bit for bit.
///
/// Since Python 3.12, `sum()` does not simply add left to right: it uses Neumaier's
/// compensated summation, which carries the rounding error of each addition in a second
/// variable `c` and adds it back at the end. The account's totals (market value, gross
/// exposure, unrealized P&L) are `sum()`s in Python, so the Rust core must sum the same
/// way, in the same order, to give identical equity.
///
/// `impl IntoIterator<Item = f64>` accepts anything that can be looped over and yields
/// floats: a `Vec`, an array, or a lazy iterator such as `positions.iter().map(...)`.
pub fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter();
    let Some(first) = values.next() else {
        return 0.0; // Python returns the integer 0 here; as a float that is 0.0
    };
    // Python starts from the integer 0, so the first float is added to 0: this turns a
    // `-0.0` into `0.0`, exactly as `0 + -0.0` does in Python.
    let mut total = 0.0 + first;
    let mut compensation = 0.0;
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            compensation += (total - t) + x;
        } else {
            compensation += (x - t) + total;
        }
        total = t;
    }
    // Python skips the correction when it is zero, and when it is not finite (so an
    // infinite or overflowed sum does not turn into NaN).
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// A float formatted the way Python's `repr()` prints the special values.
///
/// Used in error messages so both cores raise the same text: Rust prints `NaN`, Python `nan`.
pub fn py_repr(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{value:?}")
    }
}

// `#[cfg(test)]` means "only compile this module for `cargo test`". Unit tests usually
// live next to the code they test, in a child module like this one.
#[cfg(test)]
mod tests {
    // `super::*` brings everything from the parent module (this file) into scope.
    use super::*;

    #[test]
    fn max_and_min_follow_python_with_nan() {
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_min(1.0, 2.0), 1.0);
        // Python: max(1.0, nan) == 1.0, but max(nan, 1.0) is nan.
        assert_eq!(py_max(1.0, f64::NAN), 1.0);
        assert!(py_max(f64::NAN, 1.0).is_nan());
        assert_eq!(py_min(1.0, f64::NAN), 1.0);
        assert!(py_min(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn sum_compensates_rounding_like_python() {
        // Python 3.12+: sum([0.1] * 10) == 1.0, while adding left to right gives 0.9999999999999999.
        let tenths = [0.1; 10];
        assert_eq!(py_sum(tenths), 1.0);
        assert_eq!(tenths.iter().fold(0.0, |a, b| a + b), 0.9999999999999999);
        // sum([1e100, 1.0, -1e100]) == 1.0 in Python; plain addition loses the 1.0.
        assert_eq!(py_sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(py_sum([]), 0.0);
        assert_eq!(py_sum([-0.0]).to_bits(), 0.0_f64.to_bits()); // +0.0, as in Python
        assert_eq!(py_sum([1e308, 1e308, -1e308]), f64::INFINITY);
    }

    #[test]
    fn repr_matches_python_for_special_values() {
        assert_eq!(py_repr(f64::NAN), "nan");
        assert_eq!(py_repr(f64::INFINITY), "inf");
        assert_eq!(py_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(py_repr(2.5), "2.5");
    }
}
