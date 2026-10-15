//! Corporate actions: splits and cash dividends, applied before a session's trading.
//!
//! Mirrors the first stage of `PythonCore.step`. Python works out which date row each
//! action takes effect on and in what order (splits first, then by symbol), so this
//! module only applies them.

use crate::account::Account;
use crate::broker::Broker;
use crate::error::EngineError;

/// What happened to a symbol.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionKind {
    /// A `ratio`-for-one split (2.0 for a 2:1 split).
    Split,
    /// A cash dividend of `value` dollars per share.
    Dividend,
}

/// A split (`value` is the ratio) or a cash dividend (`value` is dollars per share).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorporateAction {
    pub symbol: usize,
    pub kind: ActionKind,
    pub value: f64,
}

impl CorporateAction {
    /// Apply the action to the account and, for a split, to the open orders.
    ///
    /// A split restates positions and open orders (more shares, lower price), as a real
    /// broker does; a dividend is paid in cash on the shares held, and shorts pay it.
    /// `symbol_name` is only used in the error message.
    pub fn apply(
        &self,
        account: &mut Account,
        broker: &mut Broker,
        symbol_name: &str,
    ) -> Result<(), EngineError> {
        match self.kind {
            ActionKind::Split => {
                if !(self.value > 0.0) {
                    return Err(EngineError::invalid(format!(
                        "invalid split ratio {} for {symbol_name}",
                        crate::numeric::py_repr(self.value)
                    )));
                }
                account.apply_split(self.symbol, self.value);
                broker.apply_split(self.symbol, self.value);
            }
            ActionKind::Dividend => {
                account.apply_dividend(self.symbol, self.value);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::Constraints;
    use crate::orders::{Fill, OrderType, Side, TimeInForce};

    fn holding_ten_at_100() -> (Account, Broker) {
        let mut account = Account::new(0.0);
        account.apply_fill(&Fill {
            order_number: 1,
            symbol: 0,
            side: Side::Buy,
            quantity: 10.0,
            price: 100.0,
            commission: 0.0,
            spread_cost: 0.0,
            slippage_cost: 0.0,
        });
        (account, Broker::new(Constraints::default(), true))
    }

    #[test]
    fn two_for_one_split_restates_positions_and_open_orders() {
        let (mut account, mut broker) = holding_ten_at_100();
        let limit = broker
            .submit(
                0,
                Side::Sell,
                10.0,
                OrderType::Limit,
                Some(120.0),
                TimeInForce::Gtc,
            )
            .expect("a valid order")
            .expect("not rounded away");
        let split = CorporateAction {
            symbol: 0,
            kind: ActionKind::Split,
            value: 2.0,
        };
        split
            .apply(&mut account, &mut broker, "AAA")
            .expect("a valid split");
        assert_eq!(account.quantity(0), 20.0);
        assert_eq!(account.position(0).map(|p| p.avg_cost), Some(50.0));
        let order = broker.order(limit);
        assert_eq!((order.quantity, order.limit_price), (20.0, Some(60.0)));
    }

    #[test]
    fn dividend_is_paid_in_cash() {
        let (mut account, mut broker) = holding_ten_at_100();
        let dividend = CorporateAction {
            symbol: 0,
            kind: ActionKind::Dividend,
            value: 0.75,
        };
        dividend
            .apply(&mut account, &mut broker, "AAA")
            .expect("dividends cannot fail");
        assert_eq!(account.cash, -1000.0 + 7.5);
        assert_eq!(account.realized_pnl, 7.5);
    }

    #[test]
    fn a_split_ratio_must_be_positive() {
        let (mut account, mut broker) = holding_ten_at_100();
        let bad = CorporateAction {
            symbol: 0,
            kind: ActionKind::Split,
            value: 0.0,
        };
        let error = bad
            .apply(&mut account, &mut broker, "AAA")
            .expect_err("ratio 0 is invalid");
        assert_eq!(error.message, "invalid split ratio 0.0 for AAA");
    }
}
