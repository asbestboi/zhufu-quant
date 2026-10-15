//! The account: cash, positions, realized P&L, and fees, updated fill by fill.
//!
//! Mirrors `qr.backtest.account.Account`, and keeps the same identity true at all times:
//!
//! ```text
//! equity = initial cash + realized_pnl + unrealized_pnl - fees_paid
//! ```
//!
//! **Order matters for exact results.** Python keeps positions in a `dict`, which
//! remembers insertion order, and totals them with `sum()` in that order. Floating-point
//! addition is not associative (`(a + b) + c` can differ from `a + (b + c)` in the last
//! bit), so the Rust account keeps positions in a `Vec` in the same order (a new
//! position goes at the end, a closed one is removed, an updated one stays where it is)
//! and totals them with [`py_sum`], which adds exactly like Python's `sum()`.

use crate::numeric::{py_min, py_sum};
use crate::orders::{Fill, Side};

/// A holding in one symbol. `quantity` is negative for a short.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Position {
    pub quantity: f64,
    pub avg_cost: f64,
    pub market_value: f64,
}

/// The record of one fill: its sequence number (Python's `T0000001`) and the profit it
/// locked in by closing shares, before fees. The rest of the trade is the fill itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trade {
    pub number: usize,
    pub realized_pnl: f64,
}

/// A cash account with positions.
#[derive(Clone, Debug, Default)]
pub struct Account {
    pub cash: f64,
    /// `(symbol, position)` pairs in the order the positions were opened.
    positions: Vec<(usize, Position)>,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub trade_count: usize,
}

impl Account {
    /// A new account holding only `cash`.
    pub fn new(cash: f64) -> Account {
        // `..Default::default()` fills the remaining fields with their defaults (0, empty).
        Account {
            cash,
            ..Default::default()
        }
    }

    /// The positions, in the order they were opened.
    ///
    /// Returning `&[...]` (a *slice*: a borrowed view of part of a `Vec`) lets the caller
    /// read the positions without copying them and without being able to change them.
    pub fn positions(&self) -> &[(usize, Position)] {
        &self.positions
    }

    fn slot(&self, symbol: usize) -> Option<usize> {
        self.positions.iter().position(|(s, _)| *s == symbol)
    }

    /// The position in `symbol`, if there is one.
    pub fn position(&self, symbol: usize) -> Option<&Position> {
        self.slot(symbol).map(|i| &self.positions[i].1)
    }

    /// Signed number of shares held in `symbol` (0.0 when flat).
    pub fn quantity(&self, symbol: usize) -> f64 {
        self.position(symbol).map_or(0.0, |p| p.quantity)
    }

    /// Signed market value of all positions.
    pub fn market_value(&self) -> f64 {
        // `iter().map(...)` is a lazy loop: it yields each value only when `py_sum` asks.
        py_sum(self.positions.iter().map(|(_, p)| p.market_value))
    }

    /// Long plus short market value, both counted positive.
    pub fn gross_exposure(&self) -> f64 {
        py_sum(self.positions.iter().map(|(_, p)| p.market_value.abs()))
    }

    /// Market value minus cost of the open positions.
    pub fn unrealized_pnl(&self) -> f64 {
        py_sum(
            self.positions
                .iter()
                .map(|(_, p)| p.market_value - p.quantity * p.avg_cost),
        )
    }

    /// Cash plus the market value of positions.
    pub fn equity(&self) -> f64 {
        self.cash + self.market_value()
    }

    /// Book a fill: move cash, update the position, and return the trade record.
    ///
    /// `&mut self` borrows the account *mutably*: while this method runs, nothing else
    /// can read or change it. The compiler checks this at compile time, which is how Rust
    /// rules out data races without a garbage collector or locks.
    pub fn apply_fill(&mut self, fill: &Fill) -> Trade {
        let signed = match fill.side {
            Side::Buy => fill.quantity,
            Side::Sell => -fill.quantity,
        };
        let slot = self.slot(fill.symbol);
        // `Position` is `Copy`, so this takes a copy we can change freely and store back.
        let mut position = slot.map_or(Position::default(), |i| self.positions[i].1);
        let held = position.quantity;
        let mut realized = 0.0;
        if held == 0.0 || (held > 0.0) == (signed > 0.0) {
            // Opening or adding: blend the new shares into the average cost.
            let cost = held.abs() * position.avg_cost + fill.quantity * fill.price;
            position.avg_cost = cost / (held.abs() + fill.quantity);
        } else {
            let closed = py_min(fill.quantity, held.abs());
            let direction = if held > 0.0 { 1.0 } else { -1.0 };
            realized = closed * (fill.price - position.avg_cost) * direction;
            if fill.quantity > held.abs() {
                position.avg_cost = fill.price; // went through zero: the rest is new
            }
        }
        position.quantity = held + signed;
        position.market_value = position.quantity * fill.price;
        match (slot, position.quantity == 0.0) {
            (Some(i), true) => {
                // `Vec::remove` shifts the later positions down, keeping their order.
                self.positions.remove(i);
            }
            (Some(i), false) => self.positions[i].1 = position,
            (None, false) => self.positions.push((fill.symbol, position)),
            (None, true) => {}
        }

        self.cash -= signed * fill.price + fill.commission;
        self.fees_paid += fill.commission;
        self.realized_pnl += realized;
        self.trade_count += 1;
        Trade {
            number: self.trade_count,
            realized_pnl: realized,
        }
    }

    /// A `ratio`-for-one split: more shares at a lower cost, same value.
    ///
    /// Fractional shares are kept as they are (a real broker pays cash in lieu). The
    /// caller has checked that `ratio` is positive.
    pub fn apply_split(&mut self, symbol: usize, ratio: f64) {
        if let Some(i) = self.slot(symbol) {
            let position = &mut self.positions[i].1;
            position.quantity *= ratio;
            position.avg_cost /= ratio;
        }
    }

    /// Pay a cash dividend on the shares held; shorts pay it. Returns the cash moved.
    pub fn apply_dividend(&mut self, symbol: usize, per_share: f64) -> f64 {
        let amount = self.quantity(symbol) * per_share;
        self.cash += amount;
        self.realized_pnl += amount;
        amount
    }

    /// Take a fee (for example short borrow) out of cash.
    pub fn charge_fee(&mut self, amount: f64) {
        self.cash -= amount;
        self.fees_paid += amount;
    }

    /// Revalue positions at `price(symbol)`; a symbol without a price keeps its value.
    ///
    /// `impl Fn(usize) -> Option<f64>` accepts any function or closure with that
    /// signature, so the caller decides where prices come from.
    pub fn mark(&mut self, price: impl Fn(usize) -> Option<f64>) {
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(price) = price(*symbol) {
                position.market_value = position.quantity * price;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(symbol: usize, side: Side, quantity: f64, price: f64, commission: f64) -> Fill {
        Fill {
            order_number: 1,
            symbol,
            side,
            quantity,
            price,
            commission,
            spread_cost: 0.0,
            slippage_cost: 0.0,
        }
    }

    /// The account identity, which must hold after anything happens.
    fn identity_holds(account: &Account, initial: f64) -> bool {
        let rhs = initial + account.realized_pnl + account.unrealized_pnl() - account.fees_paid;
        (account.equity() - rhs).abs() < 1e-9
    }

    #[test]
    fn buy_then_sell_with_average_cost() {
        let mut account = Account::new(10_000.0);
        account.apply_fill(&fill(0, Side::Buy, 10.0, 100.0, 1.0));
        account.apply_fill(&fill(0, Side::Buy, 10.0, 110.0, 1.0));
        let position = *account.position(0).expect("a position after buying");
        assert_eq!(position.quantity, 20.0);
        assert_eq!(position.avg_cost, 105.0);
        assert_eq!(account.cash, 10_000.0 - 1000.0 - 1.0 - 1100.0 - 1.0);

        let trade = account.apply_fill(&fill(0, Side::Sell, 5.0, 120.0, 1.0));
        assert_eq!(trade.number, 3);
        assert_eq!(trade.realized_pnl, 5.0 * (120.0 - 105.0));
        assert_eq!(account.quantity(0), 15.0);
        assert_eq!(account.fees_paid, 3.0);
        assert!(identity_holds(&account, 10_000.0));
    }

    #[test]
    fn going_through_zero_starts_a_new_short() {
        let mut account = Account::new(1_000.0);
        account.apply_fill(&fill(0, Side::Buy, 10.0, 50.0, 0.0));
        let trade = account.apply_fill(&fill(0, Side::Sell, 15.0, 60.0, 0.0));
        assert_eq!(trade.realized_pnl, 10.0 * 10.0);
        let position = *account
            .position(0)
            .expect("a short after selling through zero");
        assert_eq!(position.quantity, -5.0);
        assert_eq!(position.avg_cost, 60.0);
        assert!(identity_holds(&account, 1_000.0));
    }

    #[test]
    fn closing_removes_the_position_and_keeps_the_others_in_order() {
        let mut account = Account::new(10_000.0);
        for symbol in [2, 0, 1] {
            account.apply_fill(&fill(symbol, Side::Buy, 1.0, 10.0, 0.0));
        }
        account.apply_fill(&fill(0, Side::Sell, 1.0, 10.0, 0.0));
        let order: Vec<usize> = account.positions().iter().map(|(s, _)| *s).collect();
        assert_eq!(order, vec![2, 1]);
        account.apply_fill(&fill(0, Side::Buy, 1.0, 10.0, 0.0)); // reopened: goes last
        let order: Vec<usize> = account.positions().iter().map(|(s, _)| *s).collect();
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn two_for_one_split_keeps_value() {
        let mut account = Account::new(0.0);
        account.apply_fill(&fill(0, Side::Buy, 10.0, 100.0, 0.0));
        account.apply_split(0, 2.0);
        let position = *account.position(0).expect("still held after the split");
        assert_eq!(position.quantity, 20.0);
        assert_eq!(position.avg_cost, 50.0);
        account.mark(|_| Some(50.0));
        assert_eq!(account.market_value(), 1000.0);
        assert!(identity_holds(&account, 0.0));
    }

    #[test]
    fn dividends_are_paid_to_longs_and_by_shorts() {
        let mut account = Account::new(1_000.0);
        account.apply_fill(&fill(0, Side::Buy, 10.0, 10.0, 0.0));
        account.apply_fill(&fill(1, Side::Sell, 4.0, 10.0, 0.0));
        assert_eq!(account.apply_dividend(0, 0.5), 5.0);
        assert_eq!(account.apply_dividend(1, 0.5), -2.0);
        assert_eq!(account.apply_dividend(2, 0.5), 0.0); // not held
        assert_eq!(account.realized_pnl, 3.0);
        assert!(identity_holds(&account, 1_000.0));
    }

    #[test]
    fn fees_reduce_cash_and_count_as_fees() {
        let mut account = Account::new(100.0);
        account.charge_fee(2.5);
        assert_eq!(account.cash, 97.5);
        assert_eq!(account.fees_paid, 2.5);
        assert!(identity_holds(&account, 100.0));
    }

    #[test]
    fn marking_without_a_price_keeps_the_last_value() {
        let mut account = Account::new(0.0);
        account.apply_fill(&fill(0, Side::Buy, 2.0, 10.0, 0.0));
        account.apply_fill(&fill(1, Side::Buy, 3.0, 10.0, 0.0));
        account.mark(|symbol| if symbol == 0 { Some(12.0) } else { None });
        assert_eq!(account.position(0).map(|p| p.market_value), Some(24.0));
        assert_eq!(account.position(1).map(|p| p.market_value), Some(30.0));
    }
}
