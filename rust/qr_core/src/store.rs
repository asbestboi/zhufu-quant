//! The bar store: every bar of every symbol in flat arrays, owned by Rust.
//!
//! Each field (open, high, low, close, volume, and the adjusted prices) is one `Vec<f64>`
//! holding a `(dates x symbols)` table row by row: the value for date row `t` and symbol
//! column `j` is at index `t * n_symbols + j`. NaN means "no bar". Mirrors the arrays of
//! `qr.backtest.event_engine.PythonCore`.
//!
//! The store never changes after it is built. That is what makes it safe for Python to
//! look at it through NumPy views without copying (see `lib.rs`).

use crate::error::EngineError;

/// A bar field. The discriminants (`= 0`, ...) are the numbers Python passes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Open = 0,
    High = 1,
    Low = 2,
    Close = 3,
    Volume = 4,
}

impl Field {
    /// All fields, in the order of their numbers.
    pub const ALL: [Field; 5] = [
        Field::Open,
        Field::High,
        Field::Low,
        Field::Close,
        Field::Volume,
    ];

    /// The field with number `index`, if there is one.
    pub fn from_index(index: usize) -> Option<Field> {
        Field::ALL.get(index).copied()
    }
}

/// The rows of one symbol's history window.
///
/// The `'a` is a *lifetime*: it says `rows` borrows from something that lives at least
/// as long as `'a` (here, the store). The compiler uses it to make sure a `Window` is
/// never used after the store it points into is gone.
#[derive(Debug, PartialEq)]
pub struct Window<'a> {
    /// The date rows of the symbol's bars in the window, oldest first.
    pub rows: &'a [usize],
    /// True when those rows are consecutive, so the window is a plain slice of the table.
    pub contiguous: bool,
}

/// Bars of a fixed list of symbols over a fixed list of dates.
#[derive(Debug)]
pub struct BarStore {
    n_dates: usize,
    n_symbols: usize,
    raw: [Vec<f64>; 5],
    adjusted: [Vec<f64>; 4],
    adv: Vec<f64>,
    has_bar: Vec<bool>,
    /// For each symbol, the rows where it has a bar.
    rows: Vec<Vec<usize>>,
    /// For each symbol, whether those rows have no gaps.
    gapless: Vec<bool>,
}

impl BarStore {
    /// Build a store from row-major tables.
    ///
    /// `raw` is open, high, low, close, volume as traded; `adjusted` is open, high, low,
    /// close total-return adjusted (anchored at the last date); `adv` is the average daily
    /// dollar volume known before each date. A symbol has a bar on a date when its four
    /// raw prices are all finite.
    ///
    /// Returns `Result<BarStore, EngineError>`: either `Ok(store)` or `Err(error)`. Rust
    /// has no exceptions; a function that can fail says so in its return type, and the
    /// caller has to deal with both cases (the `?` operator passes an error on upward).
    pub fn new(
        n_dates: usize,
        n_symbols: usize,
        raw: [Vec<f64>; 5],
        adjusted: [Vec<f64>; 4],
        adv: Vec<f64>,
    ) -> Result<BarStore, EngineError> {
        let size = n_dates * n_symbols;
        let all_sized = raw
            .iter()
            .chain(adjusted.iter())
            .chain([&adv])
            .all(|t| t.len() == size);
        if !all_sized {
            return Err(EngineError::invalid(format!(
                "every bar table must have {n_dates} x {n_symbols} values"
            )));
        }
        let has_bar: Vec<bool> = (0..size)
            .map(|i| raw[..4].iter().all(|table| table[i].is_finite()))
            .collect();
        let rows: Vec<Vec<usize>> = (0..n_symbols)
            .map(|j| {
                (0..n_dates)
                    .filter(|t| has_bar[t * n_symbols + j])
                    .collect()
            })
            .collect();
        let gapless = rows
            .iter()
            .map(|r| match (r.first(), r.last()) {
                (Some(first), Some(last)) => last - first + 1 == r.len(),
                _ => true, // no bars at all: an empty window is trivially gapless
            })
            .collect();
        Ok(BarStore {
            n_dates,
            n_symbols,
            raw,
            adjusted,
            adv,
            has_bar,
            rows,
            gapless,
        })
    }

    pub fn n_dates(&self) -> usize {
        self.n_dates
    }

    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// Whether `symbol` has a bar on date row `t`.
    pub fn has_bar(&self, t: usize, symbol: usize) -> bool {
        self.has_bar[t * self.n_symbols + symbol]
    }

    /// The whole has-a-bar table, row by row.
    pub fn has_bar_table(&self) -> &[bool] {
        &self.has_bar
    }

    /// An unadjusted value.
    pub fn raw(&self, field: Field, t: usize, symbol: usize) -> f64 {
        self.raw[field as usize][t * self.n_symbols + symbol]
    }

    /// The whole table of an unadjusted field, row by row.
    pub fn raw_table(&self, field: Field) -> &[f64] {
        &self.raw[field as usize]
    }

    /// Average daily dollar volume of `symbol` before date row `t` (NaN if unknown).
    pub fn adv(&self, t: usize, symbol: usize) -> f64 {
        self.adv[t * self.n_symbols + symbol]
    }

    /// The last `n` bars of `symbol` up to and including date row `upto`, never beyond.
    ///
    /// `upto` is `None` before the first date of the store.
    pub fn window(&self, symbol: usize, n: usize, upto: Option<usize>) -> Window<'_> {
        let rows = &self.rows[symbol];
        // `partition_point` is a binary search: the number of rows at or before `upto`.
        let seen = upto.map_or(0, |t| rows.partition_point(|&row| row <= t));
        let own = &rows[seen.saturating_sub(n)..seen];
        Window {
            rows: own,
            contiguous: !own.is_empty() && self.gapless[symbol],
        }
    }

    /// The close of `symbol`'s last bar at or before `upto`, if it has had one.
    pub fn last_close(&self, symbol: usize, upto: Option<usize>) -> Option<f64> {
        let window = self.window(symbol, 1, upto);
        window
            .rows
            .first()
            .map(|&row| self.raw(Field::Close, row, symbol))
    }

    /// The values of a window, adjusted point-in-time when `adjusted` is true.
    ///
    /// The stored adjusted prices are anchored at the end of the data. Rescaling them so
    /// that the window's last close equals the traded close removes every corporate
    /// action after it: the series as it was known on that day. The arithmetic is the
    /// same as Python's (`adjusted * (raw_close / adjusted_close)`), so the values are
    /// identical. Volume is never adjusted.
    pub fn values(&self, symbol: usize, window: &Window, field: Field, adjusted: bool) -> Vec<f64> {
        if field == Field::Volume || !adjusted {
            let table = self.raw_table(field);
            return window
                .rows
                .iter()
                .map(|&t| table[t * self.n_symbols + symbol])
                .collect();
        }
        let Some(&last) = window.rows.last() else {
            return Vec::new(); // `let ... else`: bail out early when there is nothing
        };
        let at = |table: &[f64], t: usize| table[t * self.n_symbols + symbol];
        let anchor = at(self.raw_table(Field::Close), last)
            / at(&self.adjusted[Field::Close as usize], last);
        let table = &self.adjusted[field as usize];
        window.rows.iter().map(|&t| at(table, t) * anchor).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    /// Two symbols over four dates. Symbol 1 has no bar on date 1 (a gap). Symbol 0 had a
    /// 2:1 split on date 2, so its adjusted prices before then are halved.
    pub(crate) fn store() -> BarStore {
        let close = vec![100.0, 10.0, 102.0, NAN, 51.0, 11.0, 52.0, 12.0];
        let open = close.clone();
        let raw = [open.clone(), open.clone(), open, close, vec![1000.0; 8]];
        let adjusted_close = vec![50.0, 10.0, 51.0, NAN, 51.0, 11.0, 52.0, 12.0];
        let adjusted = [
            adjusted_close.clone(),
            adjusted_close.clone(),
            adjusted_close.clone(),
            adjusted_close,
        ];
        BarStore::new(4, 2, raw, adjusted, vec![NAN; 8]).expect("a valid store")
    }

    #[test]
    fn rejects_tables_of_the_wrong_size() {
        let short = || vec![1.0; 3];
        let result = BarStore::new(
            2,
            2,
            [short(), short(), short(), short(), short()],
            [short(), short(), short(), short()],
            short(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn finds_bars_and_gaps() {
        let store = store();
        assert!(store.has_bar(0, 1));
        assert!(!store.has_bar(1, 1));
        assert_eq!(store.window(1, 10, Some(3)).rows, &[0, 2, 3]);
        assert!(!store.window(1, 10, Some(3)).contiguous);
        assert!(store.window(0, 10, Some(3)).contiguous);
    }

    #[test]
    fn windows_never_reach_past_the_current_row() {
        let store = store();
        assert_eq!(store.window(0, 10, Some(1)).rows, &[0, 1]);
        assert_eq!(store.window(0, 2, Some(3)).rows, &[2, 3]);
        assert_eq!(store.window(0, 0, Some(3)).rows, &[] as &[usize]);
        assert_eq!(store.window(0, 5, None).rows, &[] as &[usize]);
    }

    #[test]
    fn adjusted_history_only_knows_past_splits() {
        let store = store();
        // On date 1 the split has not happened: prices as traded.
        let before = store.window(0, 10, Some(1));
        assert_eq!(
            store.values(0, &before, Field::Close, true),
            vec![100.0, 102.0]
        );
        // On date 3 it has: earlier prices halved, the latest one as traded.
        let after = store.window(0, 10, Some(3));
        assert_eq!(
            store.values(0, &after, Field::Close, true),
            vec![50.0, 51.0, 51.0, 52.0]
        );
        assert_eq!(
            store.values(0, &after, Field::Close, false),
            vec![100.0, 102.0, 51.0, 52.0]
        );
        assert_eq!(store.last_close(0, Some(3)), Some(52.0));
        assert_eq!(store.last_close(1, None), None);
    }
}
