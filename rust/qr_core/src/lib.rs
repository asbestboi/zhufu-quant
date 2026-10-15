//! `qr_core`: the event-driven backtest engine's core, in Rust, for the `qr` package.
//!
//! This file is the Python module (built with PyO3 and maturin). The engine itself lives
//! in plain Rust modules that know nothing about Python, and are tested with
//! `cargo test`:
//!
//! - [`store`]: every bar in flat arrays (the "bar store"), and point-in-time history
//! - [`account`]: cash, positions, P&L, fees
//! - [`orders`]: order types and the limit-order rule
//! - [`costs`]: commission, spread, slippage, borrow, participation cap
//! - [`broker`]: the order book and matching
//! - [`corporate`]: splits and dividends
//! - [`engine`]: the event loop that ties them together
//!
//! Python's `qr.backtest.rust_core.RustCore` wraps the [`PyEngine`] class defined here
//! and implements the same `EngineCore` interface as the Python core, so strategies do
//! not change. Strategy callbacks stay in Python.

pub mod account;
pub mod broker;
pub mod corporate;
pub mod costs;
pub mod engine;
pub mod error;
pub mod numeric;
pub mod orders;
pub mod store;

use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::broker::Constraints;
use crate::corporate::{ActionKind, CorporateAction};
use crate::costs::CostParams;
use crate::engine::{Engine, RunSettings};
use crate::error::EngineError;
use crate::orders::{OrderType, Side, TimeInForce};
use crate::store::{BarStore, Field};

// Rust's `?` operator converts errors with the `From` trait. Implementing `From` for
// PyO3's error type means `?` turns an `EngineError` into a Python `ValueError`.
impl From<EngineError> for PyErr {
    fn from(error: EngineError) -> PyErr {
        PyValueError::new_err(error.message)
    }
}

/// The cost numbers of one symbol, as Python sends them (see `RustCore`).
type CostTuple = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool);

/// One fill and its trade, as Python receives them:
/// `(order, symbol, is_buy, quantity, price, commission, spread, slippage, trade, realized)`.
type FillTuple = (usize, usize, bool, f64, f64, f64, f64, f64, usize, f64);

/// What `step` returns: `(fills, borrow_fee, numbers of the orders that changed)`.
type StepTuple = (Vec<FillTuple>, f64, Vec<usize>);

/// An order's changing fields: `(quantity, limit_price, status, filled_quantity, reason)`.
type OrderTuple = (f64, Option<f64>, &'static str, f64, &'static str);

/// A row-major copy of a 2-D NumPy array, checked against the expected shape.
fn table(array: &PyReadonlyArray2<'_, f64>, shape: (usize, usize)) -> PyResult<Vec<f64>> {
    let view = array.as_array();
    if view.dim() != shape {
        return Err(PyValueError::new_err(format!(
            "every bar table must have shape {shape:?}, got {:?}",
            view.dim()
        )));
    }
    // `iter()` walks the array in logical (row-major) order whatever its memory layout.
    Ok(view.iter().copied().collect())
}

/// Switch off NumPy's write flag, so Python cannot change the data by accident.
fn make_read_only<'py, D: numpy::ndarray::Dimension>(
    array: Bound<'py, numpy::PyArray<f64, D>>,
) -> PyResult<Bound<'py, numpy::PyArray<f64, D>>> {
    let writable: PyReadwriteArray<'py, f64, D> = array
        .try_readwrite()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    writable.make_nonwriteable();
    Ok(array)
}

/// The engine, as a Python class (`qr_core.Engine`).
///
/// `#[pyclass]` makes a Rust struct usable from Python; the struct itself lives inside a
/// Python object. Methods in the `#[pymethods]` block below become Python methods.
#[pyclass(name = "Engine", module = "qr_core")]
pub struct PyEngine {
    engine: Engine,
}

impl PyEngine {
    fn column(&self, symbol: usize) -> PyResult<usize> {
        if symbol < self.engine.store().n_symbols() {
            Ok(symbol)
        } else {
            Err(PyValueError::new_err(format!(
                "no symbol in column {symbol}"
            )))
        }
    }

    fn order_tuple(&self, number: usize) -> PyResult<OrderTuple> {
        let broker = self.engine.broker();
        if !broker.has_order(number) {
            return Err(PyValueError::new_err(format!("no order number {number}")));
        }
        let order = broker.order(number);
        Ok((
            order.quantity,
            order.limit_price,
            order.status.name(),
            order.filled_quantity,
            order.reason,
        ))
    }
}

fn field(index: usize) -> PyResult<Field> {
    Field::from_index(index).ok_or_else(|| PyValueError::new_err(format!("no bar field {index}")))
}

#[pymethods]
impl PyEngine {
    /// Build the engine from the run's bars, costs, corporate actions, and settings.
    ///
    /// The arrays are copied into Rust once; from then on Rust owns the bar store.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        symbols, raw, adjusted, adv, costs, actions, rebalance, days, first,
        initial_cash, long_only, max_gross_leverage, max_weight, min_trade_weight, whole_shares,
    ))]
    fn new(
        symbols: Vec<String>,
        raw: Vec<PyReadonlyArray2<'_, f64>>,
        adjusted: Vec<PyReadonlyArray2<'_, f64>>,
        adv: PyReadonlyArray2<'_, f64>,
        costs: Vec<CostTuple>,
        actions: Vec<(usize, usize, bool, f64)>,
        rebalance: Vec<bool>,
        days: Vec<f64>,
        first: usize,
        initial_cash: f64,
        long_only: bool,
        max_gross_leverage: f64,
        max_weight: Option<f64>,
        min_trade_weight: f64,
        whole_shares: bool,
    ) -> PyResult<PyEngine> {
        let shape = adv.as_array().dim();
        // `try_into` turns a Vec of the right length into a fixed-size array, or fails.
        let raw: [Vec<f64>; 5] = raw
            .iter()
            .map(|a| table(a, shape))
            .collect::<PyResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| PyValueError::new_err("raw needs open, high, low, close, volume"))?;
        let adjusted: [Vec<f64>; 4] = adjusted
            .iter()
            .map(|a| table(a, shape))
            .collect::<PyResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| PyValueError::new_err("adjusted needs open, high, low, close"))?;
        let store = BarStore::new(shape.0, shape.1, raw, adjusted, table(&adv, shape)?)?;
        let costs = costs
            .into_iter()
            .map(|c| CostParams {
                per_share: c.0,
                min_per_order: c.1,
                max_pct: c.2,
                pct: c.3,
                spread: c.4,
                slippage: c.5,
                impact: c.6,
                impact_exponent: c.7,
                borrow_rate: c.8,
                max_participation: c.9,
                reject_over_cap: c.10,
            })
            .collect();
        let actions = actions
            .into_iter()
            .map(|(row, symbol, is_split, value)| {
                let kind = if is_split {
                    ActionKind::Split
                } else {
                    ActionKind::Dividend
                };
                (
                    row,
                    CorporateAction {
                        symbol,
                        kind,
                        value,
                    },
                )
            })
            .collect();
        let settings = RunSettings {
            initial_cash,
            constraints: Constraints {
                long_only,
                max_gross_leverage,
                max_weight,
                min_trade_weight,
            },
            whole_shares,
            first,
            sessions: rebalance.len(),
        };
        let engine = Engine::new(store, symbols, costs, actions, rebalance, days, settings)?;
        Ok(PyEngine { engine })
    }

    /// Advance one session. `None` after the last one, else
    /// `(fills, borrow_fee, numbers of the orders that changed)`.
    fn step(&mut self) -> PyResult<Option<StepTuple>> {
        let Some(output) = self.engine.step()? else {
            return Ok(None);
        };
        let fills = output
            .fills
            .iter()
            .map(|(f, t)| {
                let is_buy = f.side == Side::Buy;
                let (spread, slippage) = (f.spread_cost, f.slippage_cost);
                (
                    f.order_number,
                    f.symbol,
                    is_buy,
                    f.quantity,
                    f.price,
                    f.commission,
                    spread,
                    slippage,
                    t.number,
                    t.realized_pnl,
                )
            })
            .collect();
        Ok(Some((fills, output.borrow_fee, output.touched)))
    }

    /// The current session number, `None` before the first step.
    #[getter]
    fn session(&self) -> Option<usize> {
        self.engine.session()
    }

    #[getter]
    fn cash(&self) -> f64 {
        self.engine.account().cash
    }

    /// Equity at the latest close.
    #[getter]
    fn equity(&self) -> f64 {
        self.engine.equity()
    }

    #[getter]
    fn is_rebalance(&self) -> bool {
        self.engine.is_rebalance()
    }

    /// Signed shares held in the symbol of column `symbol`.
    fn position(&self, symbol: usize) -> PyResult<f64> {
        Ok(self.engine.account().quantity(self.column(symbol)?))
    }

    /// `(cash, [(symbol, quantity, avg_cost, market_value)], equity, gross, net,
    /// realized_pnl, unrealized_pnl, fees_paid)`, positions in the order they were opened.
    #[allow(clippy::type_complexity)]
    fn account_state(
        &self,
    ) -> (
        f64,
        Vec<(usize, f64, f64, f64)>,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    ) {
        let account = self.engine.account();
        let positions = account
            .positions()
            .iter()
            .map(|(s, p)| (*s, p.quantity, p.avg_cost, p.market_value))
            .collect();
        let market_value = account.market_value();
        (
            account.cash,
            positions,
            account.equity(),
            account.gross_exposure(),
            market_value,
            account.realized_pnl,
            account.unrealized_pnl(),
            account.fees_paid,
        )
    }

    /// The last `n` values of bar field `field` (0-4: open, high, low, close, volume) of
    /// symbol column `symbol`, up to the current session. A read-only NumPy array.
    ///
    /// Unadjusted values of a symbol without gaps are a *view* of the bar store: no data
    /// is copied. Adjusted prices are rescaled into a new array (see
    /// [`BarStore::values`]).
    fn history<'py>(
        slf: &Bound<'py, Self>,
        symbol: usize,
        n: i64,
        field: usize,
        adjusted: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let py = slf.py();
        // `borrow()` gives read access to the Rust struct inside the Python object.
        let this = slf.borrow();
        let (symbol, field) = (this.column(symbol)?, crate::field(field)?);
        let store = this.engine.store();
        let window = this.engine.window(symbol, usize::try_from(n).unwrap_or(0));
        let as_view = (field == Field::Volume || !adjusted) && window.contiguous;
        if !as_view {
            let values = store.values(symbol, &window, field, adjusted);
            return make_read_only(PyArray1::from_vec(py, values));
        }
        let n_symbols = store.n_symbols();
        let start = window.rows[0] * n_symbols + symbol;
        // A strided view: one value every `n_symbols` entries of the row-major table.
        let view = ArrayView1::from_shape(
            window.rows.len().strides(n_symbols),
            &store.raw_table(field)[start..],
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // SAFETY: the view points into the engine's bar store, which is never changed or
        // reallocated after `Engine::new` (no method hands out `&mut BarStore`). The new
        // NumPy array keeps the Python object holding the engine (`slf`) alive as its
        // base, so the memory stays valid for as long as the array exists.
        #[allow(unsafe_code)]
        let array = unsafe { PyArray1::borrow_from_array(&view, slf.clone().into_any()) };
        make_read_only(array)
    }

    /// The date rows of the window [`PyEngine::history`] returns for the same arguments.
    fn window(&self, symbol: usize, n: i64) -> PyResult<Vec<usize>> {
        let symbol = self.column(symbol)?;
        Ok(self
            .engine
            .window(symbol, usize::try_from(n).unwrap_or(0))
            .rows
            .to_vec())
    }

    /// A whole unadjusted field as a read-only `(dates, symbols)` view of the bar store.
    fn panel<'py>(slf: &Bound<'py, Self>, field: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let this = slf.borrow();
        let store = this.engine.store();
        let shape = (store.n_dates(), store.n_symbols());
        let view = ArrayView2::from_shape(shape, store.raw_table(crate::field(field)?))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // SAFETY: as in `history`: the store never changes, and `slf` is kept alive.
        #[allow(unsafe_code)]
        let array = unsafe { PyArray2::borrow_from_array(&view, slf.clone().into_any()) };
        make_read_only(array)
    }

    /// Which symbols have a bar on which dates, as a `(dates, symbols)` boolean array.
    fn has_bar<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let store = self.engine.store();
        PyArray1::from_slice(py, store.has_bar_table())
            .reshape([store.n_dates(), store.n_symbols()])
    }

    /// Shares that make `symbol` worth `weight` of equity at its latest close.
    fn target_quantity(&self, symbol: usize, weight: f64) -> PyResult<Option<f64>> {
        Ok(self.engine.target_quantity(self.column(symbol)?, weight))
    }

    /// Queue an order; returns its number, or `None` when it rounds to zero shares.
    #[pyo3(signature = (symbol, is_buy, quantity, order_type, limit_price, day))]
    fn submit(
        &mut self,
        symbol: usize,
        is_buy: bool,
        quantity: f64,
        order_type: &str,
        limit_price: Option<f64>,
        day: bool,
    ) -> PyResult<Option<usize>> {
        let symbol = self.column(symbol)?;
        let order_type = OrderType::from_name(order_type)
            .ok_or_else(|| PyValueError::new_err(format!("unknown order type {order_type:?}")))?;
        let side = if is_buy { Side::Buy } else { Side::Sell };
        let tif = if day {
            TimeInForce::Day
        } else {
            TimeInForce::Gtc
        };
        Ok(self
            .engine
            .submit(symbol, side, quantity, order_type, limit_price, tif)?)
    }

    /// Cancel an open order; returns whether it was open.
    fn cancel(&mut self, number: usize) -> bool {
        self.engine.cancel(number)
    }

    /// Numbers of the open orders, in submission order, optionally for one symbol.
    #[pyo3(signature = (symbol=None))]
    fn open_orders(&self, symbol: Option<usize>) -> Vec<usize> {
        self.engine.broker().open_orders(symbol)
    }

    /// `(quantity, limit_price, status, filled_quantity, reason)` of an order.
    fn order(&self, number: usize) -> PyResult<OrderTuple> {
        self.order_tuple(number)
    }
}

/// The Python module `qr_core`. The function name must match the module name.
#[pymodule]
fn qr_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEngine>()?;
    Ok(())
}
