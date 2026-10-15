//! Trading costs for single orders: commission, spread, slippage, borrow, volume cap.
//!
//! Mirrors the single-order methods of `qr.backtest.costs.CostModel`. Python resolves the
//! config into one set of numbers per symbol (presets, asset-class and symbol overrides,
//! the "free" before-costs run) and hands them over; this module only does the
//! arithmetic, in the same order as Python so the results are identical.

use crate::numeric::{py_max, py_min};

/// Borrow fees are charged per calendar day on a 360-day year.
pub const BORROW_DAY_COUNT: f64 = 360.0;

/// The cost numbers of one symbol. Rates are fractions, not basis points.
///
/// `#[derive(...)]` asks the compiler to write some standard trait implementations for
/// us. A *trait* is like a Java interface: `Clone` and `Copy` let a value be duplicated
/// (`Copy` means "just copy the bytes", fine for a struct of plain numbers), and `Debug`
/// lets it be printed with `{:?}`.
#[derive(Clone, Copy, Debug)]
pub struct CostParams {
    pub per_share: f64,
    pub min_per_order: f64,
    /// Commission cap as a fraction of the order's value (1e30 when there is none).
    pub max_pct: f64,
    pub pct: f64,
    pub spread: f64,
    pub slippage: f64,
    pub impact: f64,
    pub impact_exponent: f64,
    pub borrow_rate: f64,
    /// Largest order as a fraction of average daily dollar volume (infinite: no cap).
    pub max_participation: f64,
    /// Refuse an order over the cap instead of cutting it down.
    pub reject_over_cap: bool,
}

/// What one order costs, in dollars.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TradeCosts {
    pub commission: f64,
    pub spread: f64,
    pub slippage: f64,
}

impl TradeCosts {
    /// Commission, spread, and slippage together (added in that order, like Python).
    pub fn total(&self) -> f64 {
        self.commission + self.spread + self.slippage
    }
}

// An `impl` block adds methods to a type. `&self` means the method borrows the value
// read-only: it can look at it but not change it, and the caller keeps ownership.
impl CostParams {
    /// Whether a rule of this symbol depends on its trading volume.
    pub fn needs_volume(&self) -> bool {
        self.max_participation.is_finite() || self.impact > 0.0
    }

    /// Costs of one order of `shares` (sign ignored) at `price`.
    ///
    /// `adv_value` is the average daily dollar volume; it only matters with an impact
    /// cost. NaN means "unknown".
    pub fn order_costs(&self, shares: f64, price: f64, adv_value: f64) -> TradeCosts {
        let shares = shares.abs(); // a new variable that shadows the argument
        let value = shares * price;
        // Written as `!(value > 0.0)` rather than `value <= 0.0` so that a NaN value also
        // costs nothing, exactly like Python's `if not value > 0`.
        if !(value > 0.0) {
            return TradeCosts::default();
        }
        let commission = self.per_share * shares + self.pct * value;
        let commission = py_max(py_min(commission, self.max_pct * value), self.min_per_order);
        let mut slippage = value * self.slippage;
        if self.impact > 0.0 && adv_value > 0.0 {
            // Python's `x ** y` and Rust's `powf` both call the C library's `pow`.
            slippage += value * (self.impact * (value / adv_value).powf(self.impact_exponent));
        }
        TradeCosts {
            commission,
            spread: value * self.spread,
            slippage,
        }
    }

    /// How many of `shares` (positive) the volume participation cap lets through.
    pub fn fillable_shares(&self, shares: f64, price: f64, adv_value: f64) -> f64 {
        if !self.needs_volume() {
            return shares;
        }
        if !(adv_value > 0.0) {
            return 0.0; // no volume history yet: a volume rule cannot be checked
        }
        let limit = self.max_participation * adv_value;
        if shares * price <= limit {
            shares
        } else if self.reject_over_cap {
            0.0
        } else {
            limit / price
        }
    }

    /// Fee for holding a short worth `short_value` dollars for `days` calendar days.
    pub fn borrow_fee(&self, short_value: f64, days: f64) -> f64 {
        short_value.abs() * self.borrow_rate * days / BORROW_DAY_COUNT
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// IBKR-tiered-like numbers: 0.0035 a share, 0.35 minimum, 1% cap, 1 bp spread,
    /// 2 bp slippage, 1% borrow.
    pub(crate) fn ibkr() -> CostParams {
        CostParams {
            per_share: 0.0035,
            min_per_order: 0.35,
            max_pct: 0.01,
            pct: 0.0,
            spread: 0.0001,
            slippage: 0.0002,
            impact: 0.0,
            impact_exponent: 1.0,
            borrow_rate: 0.01,
            max_participation: f64::INFINITY,
            reject_over_cap: false,
        }
    }

    #[test]
    fn commission_spread_and_slippage() {
        let costs = ibkr().order_costs(1000.0, 50.0, f64::NAN);
        assert_eq!(costs.commission, 3.5); // 1000 shares * 0.0035
        assert!((costs.spread - 5.0).abs() < 1e-12); // 50,000 * 1 bp
        assert!((costs.slippage - 10.0).abs() < 1e-12); // 50,000 * 2 bp
        assert_eq!(
            costs.total(),
            costs.commission + costs.spread + costs.slippage
        );
    }

    #[test]
    fn minimum_commission_wins_over_the_cap() {
        // One share at 10: 0.0035 per share, capped at 1% (0.10), raised to the 0.35 minimum.
        assert_eq!(ibkr().order_costs(1.0, 10.0, f64::NAN).commission, 0.35);
    }

    #[test]
    fn nothing_costs_nothing() {
        assert_eq!(
            ibkr().order_costs(0.0, 10.0, f64::NAN),
            TradeCosts::default()
        );
        assert_eq!(
            ibkr().order_costs(5.0, f64::NAN, f64::NAN),
            TradeCosts::default()
        );
    }

    #[test]
    fn impact_grows_with_participation() {
        let params = CostParams {
            impact: 0.002,
            impact_exponent: 0.5,
            ..ibkr()
        };
        // Order worth 25% of daily volume: extra slippage 20 bp * sqrt(0.25) = 10 bp.
        let costs = params.order_costs(100.0, 100.0, 40_000.0);
        let expected = 10_000.0 * 0.0002 + 10_000.0 * (0.002 * 0.25_f64.sqrt());
        assert!((costs.slippage - expected).abs() < 1e-12);
    }

    #[test]
    fn participation_cap_cuts_or_rejects() {
        let capped = CostParams {
            max_participation: 0.1,
            ..ibkr()
        };
        assert_eq!(capped.fillable_shares(100.0, 10.0, 5_000.0), 50.0); // 10% of 5,000 = 500
        assert_eq!(capped.fillable_shares(10.0, 10.0, 5_000.0), 10.0); // under the cap
        assert_eq!(capped.fillable_shares(10.0, 10.0, f64::NAN), 0.0); // no volume history
        let rejecting = CostParams {
            reject_over_cap: true,
            ..capped
        };
        assert_eq!(rejecting.fillable_shares(100.0, 10.0, 5_000.0), 0.0);
        assert_eq!(ibkr().fillable_shares(1e9, 10.0, f64::NAN), 1e9); // no cap at all
    }

    #[test]
    fn borrow_accrues_per_calendar_day() {
        // 10,000 short at 1% a year for a weekend (3 days): 10,000 * 0.01 * 3 / 360.
        let fee = ibkr().borrow_fee(-10_000.0, 3.0);
        assert!((fee - 10_000.0 * 0.01 * 3.0 / 360.0).abs() < 1e-12);
    }
}
