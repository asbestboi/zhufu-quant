//! Orders, fills, and the rule that decides whether a limit order trades on a bar.
//!
//! Mirrors `qr.backtest.events.Order` and `Fill`. In Rust an `enum` is a type with a fixed
//! set of variants, like a Java enum; `match` must handle every variant, so adding one
//! later makes the compiler point at every place that needs updating.

/// Buy or sell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// +1 for a buy, -1 for a sell.
    pub fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

/// When and at what price an order may execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    /// At the next session's open.
    Market,
    /// When a later bar trades through the limit price.
    Limit,
    /// Market-on-close: at the next session's close.
    Moc,
}

impl OrderType {
    /// The name Python uses (`"market"`, `"limit"`, `"moc"`).
    pub fn name(self) -> &'static str {
        match self {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::Moc => "moc",
        }
    }

    /// The type with Python name `name`, or `None` for an unknown name.
    ///
    /// `Option<T>` is Rust's "maybe a value": `Some(value)` or `None`. There is no null in
    /// safe Rust, so a missing value must be spelled out like this and checked before use.
    pub fn from_name(name: &str) -> Option<OrderType> {
        match name {
            "market" => Some(OrderType::Market),
            "limit" => Some(OrderType::Limit),
            "moc" => Some(OrderType::Moc),
            _ => None,
        }
    }
}

/// How long a limit order stays open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till cancelled.
    Gtc,
    /// Expires at the end of the first session it could have traded in.
    Day,
}

/// Where an order is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    Open,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
    Expired,
}

impl OrderStatus {
    /// The name Python uses.
    pub fn name(self) -> &'static str {
        match self {
            OrderStatus::Open => "open",
            OrderStatus::Filled => "filled",
            OrderStatus::PartiallyFilled => "partially_filled",
            OrderStatus::Cancelled => "cancelled",
            OrderStatus::Rejected => "rejected",
            OrderStatus::Expired => "expired",
        }
    }
}

/// An instruction to trade `quantity` (always positive) shares of one symbol.
///
/// Symbols are referred to by their column number in the bar store, not by name.
/// `number` is the 1-based sequence number Python turns into the id `O0000001`.
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub number: usize,
    pub symbol: usize,
    pub side: Side,
    pub quantity: f64,
    pub order_type: OrderType,
    pub limit_price: Option<f64>,
    pub tif: TimeInForce,
    pub status: OrderStatus,
    pub filled_quantity: f64,
    /// Why the order was cut down or refused. `&'static str` is a string that lives for
    /// the whole program (a literal in the binary), so no allocation is needed.
    pub reason: &'static str,
}

/// What the broker reports when an order executes.
///
/// `price` already includes spread and slippage; `spread_cost` and `slippage_cost` say
/// how many dollars of the trade's value that was. Commission is charged on top.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub order_number: usize,
    pub symbol: usize,
    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub spread_cost: f64,
    pub slippage_cost: f64,
}

/// What a limit order does on one bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitFill {
    /// The bar opened at a better price than the limit: fill at the open.
    AtOpen,
    /// The bar traded *through* the limit: fill at the limit.
    AtLimit,
    /// The bar did not reach the limit, or only touched it.
    No,
}

/// The limit rule. A buy fills at the open when the open is at or below the limit, else
/// at the limit when the low is strictly below it; a bar that only touches the limit
/// does not fill it (with one bar a day there is no telling whether the order was
/// reached in the queue). Sells mirror this with the high.
pub fn limit_fill(side: Side, limit: f64, open: f64, high: f64, low: f64) -> LimitFill {
    let (at_open, through) = match side {
        Side::Buy => (open <= limit, low < limit),
        Side::Sell => (open >= limit, high > limit),
    };
    if at_open {
        LimitFill::AtOpen
    } else if through {
        LimitFill::AtLimit
    } else {
        LimitFill::No
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buy_limit_fills_at_open_when_the_open_is_better() {
        assert_eq!(
            limit_fill(Side::Buy, 100.0, 99.0, 101.0, 98.0),
            LimitFill::AtOpen
        );
        assert_eq!(
            limit_fill(Side::Buy, 100.0, 100.0, 101.0, 98.0),
            LimitFill::AtOpen
        );
    }

    #[test]
    fn buy_limit_fills_at_the_limit_only_when_traded_through() {
        assert_eq!(
            limit_fill(Side::Buy, 100.0, 101.0, 102.0, 99.5),
            LimitFill::AtLimit
        );
        // The low only touches the limit: no fill.
        assert_eq!(
            limit_fill(Side::Buy, 100.0, 101.0, 102.0, 100.0),
            LimitFill::No
        );
    }

    #[test]
    fn sell_limit_mirrors_buy() {
        assert_eq!(
            limit_fill(Side::Sell, 100.0, 101.0, 102.0, 99.0),
            LimitFill::AtOpen
        );
        assert_eq!(
            limit_fill(Side::Sell, 100.0, 99.0, 100.5, 98.0),
            LimitFill::AtLimit
        );
        assert_eq!(
            limit_fill(Side::Sell, 100.0, 99.0, 100.0, 98.0),
            LimitFill::No
        );
    }

    #[test]
    fn names_round_trip() {
        for order_type in [OrderType::Market, OrderType::Limit, OrderType::Moc] {
            assert_eq!(OrderType::from_name(order_type.name()), Some(order_type));
        }
        assert_eq!(OrderType::from_name("stop"), None);
    }
}
