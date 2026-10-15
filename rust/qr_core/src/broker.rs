//! The simulated broker: keeps the order book and turns orders into fills.
//!
//! Mirrors `qr.backtest.broker_sim.SimBroker`, step by step and in the same order, so the
//! two cores fill the same orders at the same prices. One session is matched against one
//! daily bar:
//!
//! 1. **Market orders** at the open, sells before buys so that sales pay for purchases.
//! 2. **Limit orders** (see [`crate::orders::limit_fill`]).
//! 3. **Market-on-close orders** at the close, sells before buys.
//!
//! Before filling, an order is cut down by, in turn: the long-only rule, the volume
//! participation cap, and the leverage limit. Whatever is cut is cancelled; an order
//! fills once or not at all.

use crate::account::{Account, Trade};
use crate::costs::{CostParams, TradeCosts};
use crate::error::EngineError;
use crate::numeric::{py_max, py_min, py_repr};
use crate::orders::{
    Fill, LimitFill, Order, OrderStatus, OrderType, Side, TimeInForce, limit_fill,
};
use crate::store::{BarStore, Field};

/// Steps of the bisection that finds the largest affordable fractional order.
const BISECTION_STEPS: usize = 60;

/// Portfolio limits (mirrors `qr.backtest.config.Constraints`).
#[derive(Clone, Copy, Debug)]
pub struct Constraints {
    pub long_only: bool,
    pub max_gross_leverage: f64,
    pub max_weight: Option<f64>,
    pub min_trade_weight: f64,
}

// A hand-written `Default`, because the default leverage limit is 1.0, not 0.0.
impl Default for Constraints {
    fn default() -> Constraints {
        Constraints {
            long_only: false,
            max_gross_leverage: 1.0,
            max_weight: None,
            min_trade_weight: 0.0,
        }
    }
}

/// Order book and matching for one account.
#[derive(Debug)]
pub struct Broker {
    pub constraints: Constraints,
    pub whole_shares: bool,
    /// Every order ever submitted; `orders[i]` has number `i + 1`.
    orders: Vec<Order>,
    /// Indices of the open orders, in the order they were submitted.
    open: Vec<usize>,
    /// Indices of orders changed since the last call to [`Broker::take_touched`].
    touched: Vec<usize>,
}

/// Where one session's prices come from: the store and a date row.
///
/// A struct that only *borrows* the store (`&'a BarStore`) costs nothing to create.
struct Session<'a> {
    store: &'a BarStore,
    t: usize,
}

impl Session<'_> {
    fn has_bar(&self, symbol: usize) -> bool {
        self.store.has_bar(self.t, symbol)
    }

    fn price(&self, field: Field, symbol: usize) -> f64 {
        self.store.raw(field, self.t, symbol)
    }
}

impl Broker {
    pub fn new(constraints: Constraints, whole_shares: bool) -> Broker {
        Broker {
            constraints,
            whole_shares,
            orders: Vec::new(),
            open: Vec::new(),
            touched: Vec::new(),
        }
    }

    /// The order with sequence number `number` (1-based).
    ///
    /// Panics if there is no such order: numbers only come from [`Broker::submit`], so a
    /// bad number is a bug in the caller, not bad input. Python-facing code checks first.
    pub fn order(&self, number: usize) -> &Order {
        &self.orders[number - 1]
    }

    /// Whether an order with this number exists.
    pub fn has_order(&self, number: usize) -> bool {
        number >= 1 && number <= self.orders.len()
    }

    /// Every order submitted so far.
    pub fn orders(&self) -> &[Order] {
        &self.orders
    }

    /// Numbers of the open orders, in submission order, optionally for one symbol.
    pub fn open_orders(&self, symbol: Option<usize>) -> Vec<usize> {
        self.open
            .iter()
            .map(|&i| &self.orders[i])
            .filter(|o| symbol.is_none_or(|s| o.symbol == s))
            .map(|o| o.number)
            .collect()
    }

    /// Numbers of the orders changed since the last call, each once, in the order they
    /// were first changed. Python uses this to update its `Order` objects.
    pub fn take_touched(&mut self) -> Vec<usize> {
        let mut seen = vec![false; self.orders.len()];
        // `std::mem::take` moves the Vec out and leaves an empty one in its place: we
        // get ownership of the list without copying it.
        std::mem::take(&mut self.touched)
            .into_iter()
            .filter(|&i| !std::mem::replace(&mut seen[i], true))
            .map(|i| self.orders[i].number)
            .collect()
    }

    fn round(&self, quantity: f64) -> f64 {
        // A hair of tolerance so 99.99999999999999 shares still counts as 100.
        if self.whole_shares {
            (quantity + 1e-9).floor()
        } else {
            quantity
        }
    }

    /// Queue an order for the next session. `Ok(None)` when it rounds to zero shares.
    pub fn submit(
        &mut self,
        symbol: usize,
        side: Side,
        quantity: f64,
        order_type: OrderType,
        limit_price: Option<f64>,
        tif: TimeInForce,
    ) -> Result<Option<usize>, EngineError> {
        match (order_type, limit_price) {
            (OrderType::Limit, Some(limit)) if limit > 0.0 => {}
            (OrderType::Limit, _) => {
                return Err(EngineError::invalid(
                    "a limit order needs a positive limit_price",
                ));
            }
            (_, Some(_)) => {
                return Err(EngineError::invalid(format!(
                    "a {} order takes no limit_price",
                    order_type.name()
                )));
            }
            (_, None) => {}
        }
        if !quantity.is_finite() {
            return Err(EngineError::invalid(format!(
                "order quantity must be finite, got {}",
                py_repr(quantity)
            )));
        }
        let quantity = self.round(quantity);
        if quantity <= 0.0 {
            return Ok(None);
        }
        let number = self.orders.len() + 1;
        self.orders.push(Order {
            number,
            symbol,
            side,
            quantity,
            order_type,
            limit_price,
            tif,
            status: OrderStatus::Open,
            filled_quantity: 0.0,
            reason: "",
        });
        self.open.push(number - 1);
        Ok(Some(number))
    }

    fn close(&mut self, index: usize, status: OrderStatus, reason: &'static str) {
        let order = &mut self.orders[index];
        order.status = status;
        order.reason = reason;
        self.open.retain(|&i| i != index); // keeps the others in order
        self.touched.push(index);
    }

    /// Cancel an open order; anything else is ignored.
    pub fn cancel(&mut self, number: usize) {
        if self.has_order(number) && self.open.contains(&(number - 1)) {
            self.close(number - 1, OrderStatus::Cancelled, "");
        }
    }

    /// Restate open orders for a split, as a real broker does.
    pub fn apply_split(&mut self, symbol: usize, ratio: f64) {
        for &i in &self.open {
            let order = &mut self.orders[i];
            if order.symbol == symbol {
                order.quantity *= ratio;
                if let Some(limit) = order.limit_price.as_mut() {
                    *limit /= ratio; // `*` writes through the reference
                }
                self.touched.push(i);
            }
        }
    }

    /// Match the open orders against the bars of date row `t` and book the fills.
    ///
    /// Leaves the account marked at the session's close.
    pub fn run_session(
        &mut self,
        account: &mut Account,
        costs: &[CostParams],
        store: &BarStore,
        t: usize,
    ) -> Vec<(Fill, Trade)> {
        let session = Session { store, t };
        let mut done = Vec::new();
        account.mark(|s| session.has_bar(s).then(|| session.price(Field::Open, s)));
        let waiting: Vec<usize> = self
            .open
            .iter()
            .copied()
            .filter(|&i| session.has_bar(self.orders[i].symbol))
            .collect();

        for i in self.sells_first(&waiting, OrderType::Market) {
            let price = session.price(Field::Open, self.orders[i].symbol);
            self.execute(i, price, false, account, costs, &session, &mut done);
        }
        for &i in &waiting {
            let order = &self.orders[i];
            let Some(limit) = order
                .limit_price
                .filter(|_| order.order_type == OrderType::Limit)
            else {
                continue;
            };
            let s = order.symbol;
            let (open, high, low) = (
                session.price(Field::Open, s),
                session.price(Field::High, s),
                session.price(Field::Low, s),
            );
            match limit_fill(order.side, limit, open, high, low) {
                LimitFill::AtOpen => {
                    self.execute(i, open, false, account, costs, &session, &mut done)
                }
                LimitFill::AtLimit => {
                    self.execute(i, limit, true, account, costs, &session, &mut done)
                }
                LimitFill::No if order.tif == TimeInForce::Day => {
                    self.close(i, OrderStatus::Expired, "");
                }
                LimitFill::No => {}
            }
        }
        account.mark(|s| session.has_bar(s).then(|| session.price(Field::Close, s)));
        for i in self.sells_first(&waiting, OrderType::Moc) {
            let price = session.price(Field::Close, self.orders[i].symbol);
            self.execute(i, price, false, account, costs, &session, &mut done);
        }
        done
    }

    /// The waiting orders of one type, sells first. `sort_by_key` is a *stable* sort, like
    /// Python's `sorted`, so orders on the same side keep their submission order.
    fn sells_first(&self, waiting: &[usize], order_type: OrderType) -> Vec<usize> {
        let mut picked: Vec<usize> = waiting
            .iter()
            .copied()
            .filter(|&i| self.orders[i].order_type == order_type)
            .collect();
        picked.sort_by_key(|&i| self.orders[i].side != Side::Sell);
        picked
    }

    fn order_costs(
        &self,
        index: usize,
        quantity: f64,
        price: f64,
        adv: f64,
        passive: bool,
        costs: &[CostParams],
    ) -> TradeCosts {
        let all = costs[self.orders[index].symbol].order_costs(quantity, price, adv);
        // A limit order resting at its price pays commission but no spread or slippage.
        if passive {
            TradeCosts {
                commission: all.commission,
                ..TradeCosts::default()
            }
        } else {
            all
        }
    }

    /// Fill order `index` at reference `price` as far as the limits allow.
    #[allow(clippy::too_many_arguments)] // mirrors the Python signature on purpose
    fn execute(
        &mut self,
        index: usize,
        price: f64,
        passive: bool,
        account: &mut Account,
        costs: &[CostParams],
        session: &Session,
        done: &mut Vec<(Fill, Trade)>,
    ) {
        let (symbol, side, wanted) = {
            let order = &self.orders[index];
            (order.symbol, order.side, order.quantity)
        };
        let adv = session.store.adv(session.t, symbol);
        let held = account.quantity(symbol);
        let mut quantity = wanted;
        let mut reason = "";

        if self.constraints.long_only && side == Side::Sell && quantity > py_max(held, 0.0) {
            quantity = py_max(held, 0.0);
            reason = "long-only: cannot sell short";
        }
        let allowed = costs[symbol].fillable_shares(quantity, price, adv);
        if allowed < quantity {
            quantity = self.round(allowed);
            reason = "volume participation cap";
        }
        let affordable =
            self.leverage_limit(index, quantity, held, price, adv, passive, account, costs);
        if affordable < quantity {
            quantity = affordable;
            reason = "leverage limit";
        }
        if quantity <= 0.0 {
            self.close(index, OrderStatus::Rejected, reason);
            return;
        }

        let paid_costs = self.order_costs(index, quantity, price, adv, passive, costs);
        let direction = side.sign();
        let mut fill_price =
            price + direction * (paid_costs.spread + paid_costs.slippage) / quantity;
        if let Some(limit) = self.orders[index].limit_price {
            // Never worse than the limit that was asked for.
            fill_price = match side {
                Side::Buy => py_min(fill_price, limit),
                Side::Sell => py_max(fill_price, limit),
            };
        }
        let paid = (fill_price - price).abs() * quantity;
        let modelled = paid_costs.spread + paid_costs.slippage;
        let share = if modelled > 0.0 { paid / modelled } else { 0.0 };
        let fill = Fill {
            order_number: self.orders[index].number,
            symbol,
            side,
            quantity,
            price: fill_price,
            commission: paid_costs.commission,
            spread_cost: paid_costs.spread * share,
            slippage_cost: paid_costs.slippage * share,
        };
        self.orders[index].filled_quantity = quantity;
        let status = if quantity >= wanted {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        self.close(index, status, reason);
        let trade = account.apply_fill(&fill);
        done.push((fill, trade));
    }

    /// The largest part of `quantity` that keeps gross exposure within the limit.
    ///
    /// Shares that reduce the position are always allowed, also when the account is
    /// already over the limit because prices moved.
    #[allow(clippy::too_many_arguments)]
    fn leverage_limit(
        &self,
        index: usize,
        quantity: f64,
        held: f64,
        price: f64,
        adv: f64,
        passive: bool,
        account: &Account,
        costs: &[CostParams],
    ) -> f64 {
        let order = &self.orders[index];
        let direction = order.side.sign();
        let reducing = if held * direction < 0.0 {
            py_min(quantity, held.abs())
        } else {
            0.0
        };
        if quantity <= reducing {
            return quantity;
        }
        let own_value = account
            .position(order.symbol)
            .map_or(0.0, |p| p.market_value);
        let other_value = account.market_value() - own_value;
        let other_gross = account.gross_exposure() - own_value.abs();
        let cash = account.cash;
        let limit = self.constraints.max_gross_leverage;

        // A *closure*: an inline function that can use the variables around it
        // (`cash`, `price`, ...), like a lambda in Java or C++.
        let within = |size: f64| -> bool {
            let costs = self.order_costs(index, size, price, adv, passive, costs);
            let after = held + direction * size;
            let equity =
                cash - direction * size * price - costs.total() + other_value + after * price;
            other_gross + after.abs() * price <= limit * equity * (1.0 + 1e-12)
        };

        if within(quantity) {
            return quantity;
        }
        let (mut low, mut high) = (reducing, quantity); // low is always allowed, high is not
        if self.whole_shares {
            // Whole shares: bisect over integers. `as i64` converts a float to an integer
            // (rounding towards zero after `floor`, saturating if it is out of range).
            let (mut low_n, mut high_n) = (low.floor() as i64, high.floor() as i64);
            while high_n - low_n > 1 {
                let middle = (low_n + high_n).div_euclid(2); // Python's `//`
                if within(middle as f64) {
                    low_n = middle;
                } else {
                    high_n = middle;
                }
            }
            return if within(low_n as f64) {
                py_max(low_n as f64, reducing)
            } else {
                reducing
            };
        }
        for _ in 0..BISECTION_STEPS {
            let half = (low + high) / 2.0;
            if within(half) {
                low = half;
            } else {
                high = half;
            }
        }
        low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::costs::tests::ibkr;

    const NAN: f64 = f64::NAN;

    /// One symbol, two dates: (open, high, low, close) = (100, 104, 99, 103) then
    /// (102, 106, 101, 105). Volume 1,000 a day; average dollar volume unknown on day 0.
    fn store() -> BarStore {
        let raw = [
            vec![100.0, 102.0],
            vec![104.0, 106.0],
            vec![99.0, 101.0],
            vec![103.0, 105.0],
            vec![1000.0, 1000.0],
        ];
        let adjusted = [
            raw[0].clone(),
            raw[1].clone(),
            raw[2].clone(),
            raw[3].clone(),
        ];
        BarStore::new(2, 1, raw, adjusted, vec![NAN, 103_000.0]).expect("a valid store")
    }

    fn free() -> CostParams {
        CostParams {
            per_share: 0.0,
            min_per_order: 0.0,
            spread: 0.0,
            slippage: 0.0,
            ..ibkr()
        }
    }

    fn submit(
        broker: &mut Broker,
        side: Side,
        quantity: f64,
        kind: OrderType,
        limit: Option<f64>,
    ) -> usize {
        broker
            .submit(0, side, quantity, kind, limit, TimeInForce::Gtc)
            .expect("a valid order")
            .expect("not rounded to zero")
    }

    #[test]
    fn market_order_fills_at_the_open_plus_costs() {
        let (mut account, mut broker) = (
            Account::new(10_000.0),
            Broker::new(Constraints::default(), true),
        );
        let number = submit(&mut broker, Side::Buy, 10.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[ibkr()], &store(), 0);
        assert_eq!(done.len(), 1);
        let (fill, trade) = done[0];
        // 1,000 of value: 1 bp spread + 2 bp slippage = 0.30, spread over 10 shares.
        assert!((fill.price - 100.03).abs() < 1e-12);
        assert_eq!(fill.commission, 0.35); // the IBKR minimum
        assert!((fill.spread_cost - 0.1).abs() < 1e-12);
        assert_eq!(trade.number, 1);
        assert_eq!(broker.order(number).status, OrderStatus::Filled);
        assert!(broker.open_orders(None).is_empty());
        // Marked at the close after the fill.
        assert_eq!(account.position(0).map(|p| p.market_value), Some(1030.0));
    }

    #[test]
    fn moc_order_fills_at_the_close() {
        let (mut account, mut broker) = (
            Account::new(10_000.0),
            Broker::new(Constraints::default(), true),
        );
        submit(&mut broker, Side::Buy, 10.0, OrderType::Moc, None);
        let done = broker.run_session(&mut account, &[free()], &store(), 0);
        assert_eq!(done[0].0.price, 103.0);
    }

    #[test]
    fn limit_orders_pay_no_spread_when_resting() {
        let (mut account, mut broker) = (
            Account::new(10_000.0),
            Broker::new(Constraints::default(), true),
        );
        // Buy limit at 99.5: the low (99) trades through it, the open (100) does not.
        submit(&mut broker, Side::Buy, 10.0, OrderType::Limit, Some(99.5));
        let done = broker.run_session(&mut account, &[ibkr()], &store(), 0);
        let fill = done[0].0;
        assert_eq!(fill.price, 99.5);
        assert_eq!((fill.spread_cost, fill.slippage_cost), (0.0, 0.0));
        assert_eq!(fill.commission, 0.35);
    }

    #[test]
    fn limit_order_that_gaps_through_fills_at_the_open_but_never_above_the_limit() {
        let (mut account, mut broker) = (
            Account::new(10_000.0),
            Broker::new(Constraints::default(), true),
        );
        submit(&mut broker, Side::Buy, 10.0, OrderType::Limit, Some(100.01));
        let done = broker.run_session(&mut account, &[ibkr()], &store(), 0);
        // Open 100 plus 3 bp of costs would be 100.03, capped at the limit.
        assert_eq!(done[0].0.price, 100.01);
    }

    #[test]
    fn day_limit_expires_and_gtc_limit_stays() {
        let (mut account, mut broker) = (
            Account::new(10_000.0),
            Broker::new(Constraints::default(), true),
        );
        let day = broker
            .submit(
                0,
                Side::Buy,
                1.0,
                OrderType::Limit,
                Some(90.0),
                TimeInForce::Day,
            )
            .expect("valid")
            .expect("nonzero");
        let gtc = submit(&mut broker, Side::Buy, 1.0, OrderType::Limit, Some(90.0));
        broker.run_session(&mut account, &[free()], &store(), 0);
        assert_eq!(broker.order(day).status, OrderStatus::Expired);
        assert_eq!(broker.open_orders(None), vec![gtc]);
        assert_eq!(broker.take_touched(), vec![day]);
    }

    #[test]
    fn sells_are_matched_before_buys() {
        let mut account = Account::new(0.0);
        let mut broker = Broker::new(Constraints::default(), true);
        account.apply_fill(&Fill {
            order_number: 0,
            symbol: 0,
            side: Side::Buy,
            quantity: 10.0,
            price: 100.0,
            commission: 0.0,
            spread_cost: 0.0,
            slippage_cost: 0.0,
        });
        account.cash = 0.0;
        // The buy was submitted first, but only the sale pays for it.
        submit(&mut broker, Side::Buy, 5.0, OrderType::Market, None);
        submit(&mut broker, Side::Sell, 5.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[free()], &store(), 0);
        let sides: Vec<Side> = done.iter().map(|(fill, _)| fill.side).collect();
        assert_eq!(sides, vec![Side::Sell, Side::Buy]);
    }

    #[test]
    fn leverage_limit_cuts_a_purchase_to_what_cash_allows() {
        let (mut account, mut broker) = (
            Account::new(1_000.0),
            Broker::new(Constraints::default(), true),
        );
        let number = submit(&mut broker, Side::Buy, 50.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[ibkr()], &store(), 0);
        // 1,000 of cash buys 9 shares at ~100.03 plus the 0.35 commission, not 10.
        assert_eq!(done[0].0.quantity, 9.0);
        let order = broker.order(number);
        assert_eq!(
            (order.status, order.reason),
            (OrderStatus::PartiallyFilled, "leverage limit")
        );
        assert!(account.cash >= 0.0);
    }

    #[test]
    fn fractional_shares_use_bisection() {
        let (mut account, mut broker) = (
            Account::new(1_000.0),
            Broker::new(Constraints::default(), false),
        );
        submit(&mut broker, Side::Buy, 50.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[free()], &store(), 0);
        // Without costs, exactly the cash: 10 shares at 100, up to the 60-step bisection.
        assert!((done[0].0.quantity - 10.0).abs() < 1e-9);
    }

    #[test]
    fn long_only_cannot_sell_short() {
        let constraints = Constraints {
            long_only: true,
            ..Constraints::default()
        };
        let (mut account, mut broker) = (Account::new(1_000.0), Broker::new(constraints, true));
        let number = submit(&mut broker, Side::Sell, 5.0, OrderType::Market, None);
        assert!(
            broker
                .run_session(&mut account, &[free()], &store(), 0)
                .is_empty()
        );
        let order = broker.order(number);
        assert_eq!(
            (order.status, order.reason),
            (OrderStatus::Rejected, "long-only: cannot sell short")
        );
    }

    #[test]
    fn participation_cap_needs_volume_history() {
        let capped = CostParams {
            max_participation: 0.1,
            ..free()
        };
        let (mut account, mut broker) = (
            Account::new(100_000.0),
            Broker::new(Constraints::default(), true),
        );
        let first = submit(&mut broker, Side::Buy, 10.0, OrderType::Market, None);
        broker.run_session(&mut account, &[capped], &store(), 0); // no average volume yet
        assert_eq!(broker.order(first).status, OrderStatus::Rejected);
        // Day 1: 10% of 103,000 dollars is 10,300, or 100 shares at the open of 102.
        let second = submit(&mut broker, Side::Buy, 500.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[capped], &store(), 1);
        assert_eq!(done[0].0.quantity, 100.0);
        assert_eq!(broker.order(second).reason, "volume participation cap");
    }

    #[test]
    fn short_sale_without_cash_is_allowed_within_the_leverage_limit() {
        let (mut account, mut broker) = (
            Account::new(1_000.0),
            Broker::new(Constraints::default(), true),
        );
        submit(&mut broker, Side::Sell, 5.0, OrderType::Market, None);
        let done = broker.run_session(&mut account, &[free()], &store(), 0);
        assert_eq!(done[0].0.quantity, 5.0);
        assert_eq!(account.quantity(0), -5.0);
    }

    #[test]
    fn invalid_orders_are_refused_with_pythons_messages() {
        let mut broker = Broker::new(Constraints::default(), true);
        let refuse = |broker: &mut Broker, kind, limit, quantity| {
            broker
                .submit(0, Side::Buy, quantity, kind, limit, TimeInForce::Gtc)
                .expect_err("an invalid order")
                .message
        };
        assert_eq!(
            refuse(&mut broker, OrderType::Limit, None, 1.0),
            "a limit order needs a positive limit_price"
        );
        assert_eq!(
            refuse(&mut broker, OrderType::Limit, Some(-1.0), 1.0),
            "a limit order needs a positive limit_price"
        );
        assert_eq!(
            refuse(&mut broker, OrderType::Market, Some(1.0), 1.0),
            "a market order takes no limit_price"
        );
        assert_eq!(
            refuse(&mut broker, OrderType::Market, None, NAN),
            "order quantity must be finite, got nan"
        );
        // Rounding to zero shares is not an error, just no order.
        assert_eq!(
            broker.submit(0, Side::Buy, 0.4, OrderType::Market, None, TimeInForce::Gtc),
            Ok(None)
        );
    }

    #[test]
    fn cancel_closes_only_open_orders() {
        let mut broker = Broker::new(Constraints::default(), true);
        let first = submit(&mut broker, Side::Buy, 1.0, OrderType::Market, None);
        let second = submit(&mut broker, Side::Buy, 1.0, OrderType::Market, None);
        broker.cancel(first);
        broker.cancel(first); // already closed: ignored
        broker.cancel(99); // unknown: ignored
        assert_eq!(broker.order(first).status, OrderStatus::Cancelled);
        assert_eq!(broker.open_orders(Some(0)), vec![second]);
        assert_eq!(broker.open_orders(Some(1)), Vec::<usize>::new());
    }
}
