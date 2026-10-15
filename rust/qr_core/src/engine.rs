//! The event loop: advance one session at a time, in the same order as the Python core.
//!
//! Mirrors `qr.backtest.event_engine.PythonCore`. A session `t` runs in this order:
//!
//! 1. Corporate actions with ex-date `t` (splits first, then dividends).
//! 2. The broker matches the orders submitted on earlier sessions against bar `t`.
//! 3. Positions are marked at the close and shorts pay their borrow fee.
//!
//! Then Python calls the strategy, which reads history and submits orders for `t + 1`.
//! The strategy is never called from here: the loop stays in Python (`run_event`), so
//! strategies, metrics, and reports work unchanged with either core.

use crate::account::{Account, Trade};
use crate::broker::{Broker, Constraints};
use crate::corporate::CorporateAction;
use crate::costs::CostParams;
use crate::error::EngineError;
use crate::numeric::{py_max, py_min};
use crate::orders::{Fill, OrderType, Side, TimeInForce};
use crate::store::{BarStore, Field, Window};

/// Everything that happened in one session.
#[derive(Debug, Default)]
pub struct StepOutput {
    pub fills: Vec<(Fill, Trade)>,
    pub borrow_fee: f64,
    /// Numbers of the orders that changed (filled, rejected, expired, split-adjusted).
    pub touched: Vec<usize>,
}

/// The settings of one run that are not bars or costs.
#[derive(Clone, Debug)]
pub struct RunSettings {
    pub initial_cash: f64,
    pub constraints: Constraints,
    pub whole_shares: bool,
    /// Row of the first session in the store (earlier rows only warm up history).
    pub first: usize,
    /// Number of sessions to run.
    pub sessions: usize,
}

/// One backtest's core: bars, account, order book, and where we are in time.
#[derive(Debug)]
pub struct Engine {
    store: BarStore,
    symbols: Vec<String>,
    costs: Vec<CostParams>,
    /// Corporate actions by the date row they take effect on.
    actions: Vec<Vec<CorporateAction>>,
    /// Whether each session is a rebalance date.
    rebalance: Vec<bool>,
    /// Calendar days since the previous date row, for borrow fees.
    days: Vec<f64>,
    first: usize,
    sessions: usize,
    /// The current session, `None` before the first step.
    k: Option<usize>,
    account: Account,
    broker: Broker,
    /// Equity at the latest close: read many times per session, computed once.
    equity: f64,
}

impl Engine {
    /// Build an engine. `actions` are `(row, action)` pairs, already in the order they
    /// must be applied within a row.
    pub fn new(
        store: BarStore,
        symbols: Vec<String>,
        costs: Vec<CostParams>,
        actions: Vec<(usize, CorporateAction)>,
        rebalance: Vec<bool>,
        days: Vec<f64>,
        settings: RunSettings,
    ) -> Result<Engine, EngineError> {
        let n_dates = store.n_dates();
        if symbols.len() != store.n_symbols() || costs.len() != store.n_symbols() {
            return Err(EngineError::invalid(
                "one symbol name and one cost set per column",
            ));
        }
        if rebalance.len() != settings.sessions || days.len() != n_dates {
            return Err(EngineError::invalid(
                "one rebalance flag per session, one day count per date",
            ));
        }
        if settings.sessions == 0 || settings.first + settings.sessions > n_dates {
            return Err(EngineError::invalid(
                "the sessions must lie within the stored dates",
            ));
        }
        let mut by_row = vec![Vec::new(); n_dates];
        for (row, action) in actions {
            if row >= n_dates || action.symbol >= store.n_symbols() {
                return Err(EngineError::invalid(
                    "a corporate action outside the stored data",
                ));
            }
            by_row[row].push(action);
        }
        let account = Account::new(settings.initial_cash);
        Ok(Engine {
            store,
            symbols,
            costs,
            actions: by_row,
            rebalance,
            days,
            first: settings.first,
            sessions: settings.sessions,
            k: None,
            equity: account.equity(),
            account,
            broker: Broker::new(settings.constraints, settings.whole_shares),
        })
    }

    pub fn store(&self) -> &BarStore {
        &self.store
    }

    pub fn account(&self) -> &Account {
        &self.account
    }

    pub fn broker(&self) -> &Broker {
        &self.broker
    }

    /// The current session, `None` before the first step.
    pub fn session(&self) -> Option<usize> {
        self.k
    }

    /// The date row of the current session. Before the first step it is the row before
    /// the first session (`None` if that is before the data), so history shows only the
    /// warm-up bars.
    pub fn current_row(&self) -> Option<usize> {
        match self.k {
            Some(k) => Some(self.first + k),
            None => self.first.checked_sub(1),
        }
    }

    pub fn equity(&self) -> f64 {
        self.equity
    }

    pub fn is_rebalance(&self) -> bool {
        self.k.is_some_and(|k| self.rebalance[k])
    }

    /// Advance one session. `Ok(None)` after the last one.
    pub fn step(&mut self) -> Result<Option<StepOutput>, EngineError> {
        let next = self.k.map_or(0, |k| k + 1);
        if next >= self.sessions {
            return Ok(None);
        }
        self.k = Some(next);
        let t = self.first + next;

        for action in &self.actions[t] {
            action.apply(
                &mut self.account,
                &mut self.broker,
                &self.symbols[action.symbol],
            )?;
        }
        let fills = self
            .broker
            .run_session(&mut self.account, &self.costs, &self.store, t);

        let mut borrow_fee = 0.0;
        for (symbol, position) in self.account.positions() {
            if position.quantity < 0.0 {
                borrow_fee += self.costs[*symbol].borrow_fee(position.market_value, self.days[t]);
            }
        }
        // `!= 0.0` is true for NaN, like Python's `if borrow:`.
        if borrow_fee != 0.0 {
            self.account.charge_fee(borrow_fee);
        }
        self.equity = self.account.equity();
        Ok(Some(StepOutput {
            fills,
            borrow_fee,
            touched: self.broker.take_touched(),
        }))
    }

    /// The last `n` bars of `symbol` up to the current session.
    pub fn window(&self, symbol: usize, n: usize) -> Window<'_> {
        self.store.window(symbol, n, self.current_row())
    }

    /// The values of `field` over the last `n` bars of `symbol` (a copy).
    pub fn history(&self, symbol: usize, n: usize, field: Field, adjusted: bool) -> Vec<f64> {
        let window = self.window(symbol, n);
        self.store.values(symbol, &window, field, adjusted)
    }

    /// Shares that make `symbol` worth `weight` of equity at its latest close.
    ///
    /// `None` when the symbol has no price yet or the weight is not finite. Applies the
    /// long-only and max-weight limits and the no-trade band, then rounds towards zero to
    /// whole shares when the run uses them.
    pub fn target_quantity(&self, symbol: usize, weight: f64) -> Option<f64> {
        // The `?` on an `Option` returns `None` from this function when it is `None`.
        let price = self.store.last_close(symbol, self.current_row())?;
        if !weight.is_finite() {
            return None;
        }
        let constraints = self.broker.constraints;
        let mut weight = weight;
        if constraints.long_only {
            weight = py_max(weight, 0.0);
        }
        if let Some(max_weight) = constraints.max_weight {
            weight = py_min(py_max(weight, -max_weight), max_weight);
        }
        let quantity = weight * py_max(self.equity, 0.0) / price;
        let held = self.account.quantity(symbol);
        let too_small =
            (quantity - held).abs() * price < constraints.min_trade_weight * self.equity;
        if weight != 0.0 && too_small {
            return Some(held); // inside the no-trade band: leave the position as it is
        }
        Some(if self.broker.whole_shares {
            (quantity + 1e-9_f64.copysign(quantity)).trunc()
        } else {
            quantity
        })
    }

    /// Queue an order for the next session (see [`Broker::submit`]).
    pub fn submit(
        &mut self,
        symbol: usize,
        side: Side,
        quantity: f64,
        order_type: OrderType,
        limit_price: Option<f64>,
        tif: TimeInForce,
    ) -> Result<Option<usize>, EngineError> {
        if symbol >= self.store.n_symbols() {
            return Err(EngineError::invalid(format!(
                "no symbol in column {symbol}"
            )));
        }
        self.broker
            .submit(symbol, side, quantity, order_type, limit_price, tif)
    }

    /// Cancel an open order; returns whether anything changed.
    pub fn cancel(&mut self, number: usize) -> bool {
        self.broker.cancel(number);
        !self.broker.take_touched().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corporate::ActionKind;
    use crate::costs::tests::ibkr;
    use crate::store::tests::store;

    fn engine(actions: Vec<(usize, CorporateAction)>, first: usize, sessions: usize) -> Engine {
        let settings = RunSettings {
            initial_cash: 10_000.0,
            constraints: Constraints::default(),
            whole_shares: true,
            first,
            sessions,
        };
        Engine::new(
            store(),
            vec!["AAA".into(), "BBB".into()],
            vec![ibkr(); 2],
            actions,
            vec![true; sessions],
            vec![1.0, 1.0, 3.0, 1.0],
            settings,
        )
        .expect("a valid engine")
    }

    #[test]
    fn steps_through_the_sessions_and_stops() {
        let mut engine = engine(vec![], 1, 3);
        assert_eq!(engine.current_row(), Some(0)); // warm-up row before the first session
        let mut rows = Vec::new();
        while engine.step().expect("no errors").is_some() {
            rows.push(engine.current_row());
        }
        assert_eq!(rows, vec![Some(1), Some(2), Some(3)]);
        assert!(engine.step().expect("no errors").is_none());
    }

    #[test]
    fn an_order_is_filled_on_the_next_session_never_the_current_one() {
        let mut engine = engine(vec![], 0, 4);
        engine.step().expect("no errors");
        let number = engine
            .submit(
                0,
                Side::Buy,
                10.0,
                OrderType::Market,
                None,
                TimeInForce::Gtc,
            )
            .expect("a valid order")
            .expect("not rounded away");
        let output = engine.step().expect("no errors").expect("a session");
        assert_eq!(output.fills.len(), 1);
        assert_eq!(output.fills[0].0.order_number, number);
        // Filled at row 1's open (102) plus 1 bp of spread and 2 bp of slippage.
        assert!((output.fills[0].0.price - 102.0 * 1.0003).abs() < 1e-9);
        assert_eq!(output.touched, vec![number]);
    }

    #[test]
    fn a_split_on_the_ex_date_restates_the_position() {
        let split = CorporateAction {
            symbol: 0,
            kind: ActionKind::Split,
            value: 2.0,
        };
        let mut engine = engine(vec![(2, split)], 0, 4);
        engine.step().expect("no errors");
        engine
            .submit(
                0,
                Side::Buy,
                10.0,
                OrderType::Market,
                None,
                TimeInForce::Gtc,
            )
            .expect("valid")
            .expect("nonzero");
        engine.step().expect("no errors"); // row 1: bought 10
        engine.step().expect("no errors"); // row 2: the split
        assert_eq!(engine.account().quantity(0), 20.0);
    }

    #[test]
    fn shorts_pay_borrow_per_calendar_day() {
        let mut engine = engine(vec![], 0, 4);
        engine.step().expect("no errors");
        engine
            .submit(
                0,
                Side::Sell,
                10.0,
                OrderType::Market,
                None,
                TimeInForce::Gtc,
            )
            .expect("valid")
            .expect("nonzero");
        engine.step().expect("no errors"); // row 1: short 10, marked at 102
        let output = engine.step().expect("no errors").expect("a session"); // row 2: 3 days
        // Short 10 marked at row 2's close of 51, for the 3 calendar days since row 1.
        let expected = (10.0 * 51.0) * 0.01 * 3.0 / 360.0;
        assert!(output.borrow_fee > 0.0);
        assert!((output.borrow_fee - expected).abs() < 1e-9);
    }

    #[test]
    fn target_quantity_rounds_towards_zero_and_respects_the_band() {
        let engine = {
            let mut engine = engine(vec![], 0, 4);
            engine.step().expect("no errors");
            engine
        };
        // 10,000 of equity, half of it at a close of 100: 50 shares.
        assert_eq!(engine.target_quantity(0, 0.5), Some(50.0));
        assert_eq!(engine.target_quantity(0, 0.0333), Some(3.0)); // 3.33 -> 3
        assert_eq!(engine.target_quantity(0, -0.0333), Some(-3.0)); // towards zero
        assert_eq!(engine.target_quantity(0, f64::NAN), None);
    }

    #[test]
    fn invalid_settings_are_refused() {
        let settings = RunSettings {
            initial_cash: 1.0,
            constraints: Constraints::default(),
            whole_shares: true,
            first: 3,
            sessions: 2, // rows 3 and 4, but the store has only 4 rows
        };
        let result = Engine::new(
            store(),
            vec!["A".into(), "B".into()],
            vec![ibkr(); 2],
            vec![],
            vec![true; 2],
            vec![1.0; 4],
            settings,
        );
        assert!(result.is_err());
    }
}
