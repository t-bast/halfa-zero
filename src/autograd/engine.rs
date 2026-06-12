//! A scalar-valued reverse-mode autograd engine using the arena/tape design.
//!
//! Core idea:
//!   - Every scalar lives in a single `Vec<Node>` (the arena/tape).
//!   - A `Value` is just an index into that arena, plus a handle to the arena.
//!   - The forward pass appends nodes to the tape in creation order.
//!   - The backward pass walks the tape in REVERSE, pushing each node's gradient onto its inputs
//!     via the chain rule.
//!
//! Because nodes only ever refer to *earlier* nodes (by index), there is no aliasing and the borrow
//! checker stays happy. No need for Rc or RefCell to update parents.

use std::cell::RefCell;
use std::rc::Rc;

/// What operation produced a node. The indices always point at *earlier* tape entries (parents).
#[derive(Clone, Copy, Debug)]
enum Op {
    /// A leaf: created directly, not from other nodes (e.g. an input or weight).
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    /// x^exponent, where exponent is a constant (not a tracked Value).
    Pow(usize, f64),
    Exp(usize),
    Log(usize),
    Tanh(usize),
    Relu(usize),
    Neg(usize),
}

/// One entry on the tape.
struct Node {
    data: f64, // the forward value
    grad: f64, // accumulated gradient d(output)/d(this), filled in by backward()
    op: Op,    // how this node was produced (its inputs, if any)
}

/// The arena: owns every node. Shared via Rc<RefCell<..>> so that many `Value` handles can append
/// to and read from the same tape. Note: the *graph* nodes themselves don't alias each other: only
/// this one container is shared, which keeps the design simple.
#[derive(Clone)]
pub struct Tape {
    nodes: Rc<RefCell<Vec<Node>>>,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            nodes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Append a node and return its index in the tape.
    fn push(&self, data: f64, op: Op) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            data,
            grad: 0.0,
            op,
        });
        idx
    }

    /// Create a leaf value at the end of the tape.
    pub fn value(&self, data: f64) -> Value {
        let idx = self.push(data, Op::Leaf);
        Value {
            tape: self.clone(),
            idx,
        }
    }
}

/// A handle to one scalar on the tape. Cheap to clone (it's an index + Rc).
#[derive(Clone)]
pub struct Value {
    tape: Tape,
    idx: usize,
}

impl Value {
    /// Read the current forward value.
    pub fn data(&self) -> f64 {
        self.tape.nodes.borrow()[self.idx].data
    }

    /// Read the gradient (which is only computed after calling `backward`).
    pub fn grad(&self) -> f64 {
        self.tape.nodes.borrow()[self.idx].grad
    }

    fn op(&self, data: f64, op: Op) -> Value {
        let idx = self.tape.push(data, op);
        Value {
            tape: self.tape.clone(),
            idx,
        }
    }

    pub fn add(&self, other: &Value) -> Value {
        self.op(self.data() + other.data(), Op::Add(self.idx, other.idx))
    }

    pub fn mul(&self, other: &Value) -> Value {
        self.op(self.data() * other.data(), Op::Mul(self.idx, other.idx))
    }

    pub fn sub(&self, other: &Value) -> Value {
        // Note that we decompose (a - b) into two operations (a + (-b)).
        // This ensures that gradients flow through both operations correctly.
        let neg = other.neg();
        self.add(&neg)
    }

    pub fn neg(&self) -> Value {
        self.op(-self.data(), Op::Neg(self.idx))
    }

    pub fn powf(&self, exponent: f64) -> Value {
        self.op(self.data().powf(exponent), Op::Pow(self.idx, exponent))
    }

    pub fn exp(&self) -> Value {
        self.op(self.data().exp(), Op::Exp(self.idx))
    }

    pub fn log(&self) -> Value {
        self.op(self.data().ln(), Op::Log(self.idx))
    }

    pub fn tanh(&self) -> Value {
        self.op(self.data().tanh(), Op::Tanh(self.idx))
    }

    pub fn relu(&self) -> Value {
        self.op(self.data().max(0.0), Op::Relu(self.idx))
    }

    /// Reverse-mode back-propagation from this node as the scalar output.
    ///
    /// Sets this node's gradient to 1.0 (d(out)/d(out) == 1), then walks the tape from the output
    /// index down to 0, distributing each node's gradient to its parents using the local derivative
    /// matching the operation used to compute it. Because every input index is strictly less than
    /// the node's own index, a single reverse pass correctly accumulates all contributions before
    /// we reach each input.
    pub fn backward(&self) {
        let mut nodes = self.tape.nodes.borrow_mut();

        // Zero every gradient first, so repeated backward() calls don't accumulate
        // across runs. (Within a single run, += is what we want.)
        for n in nodes.iter_mut() {
            n.grad = 0.0;
        }
        nodes[self.idx].grad = 1.0;

        for i in (0..=self.idx).rev() {
            let g = nodes[i].grad;
            if g == 0.0 {
                // No gradient flowing through this node; nothing to propagate.
                continue;
            }
            match nodes[i].op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    // d/da (a+b) = 1, d/db (a+b) = 1
                    nodes[a].grad += g;
                    nodes[b].grad += g;
                }
                Op::Mul(a, b) => {
                    // d/da (a*b) = b, d/db (a*b) = a
                    nodes[a].grad += nodes[b].data * g;
                    nodes[b].grad += nodes[a].data * g;
                }
                Op::Pow(a, p) => {
                    // d/da (a^p) = p * a^(p-1)
                    nodes[a].grad += p * nodes[a].data.powf(p - 1.0) * g;
                }
                Op::Exp(a) => {
                    // d/dx (exp(x)) = exp(x); we already stored exp(x) as data.
                    let e = nodes[i].data;
                    nodes[a].grad += e * g;
                }
                Op::Log(a) => {
                    // d/dx (ln(x)) = 1/x
                    nodes[a].grad += g / nodes[a].data;
                }
                Op::Tanh(a) => {
                    // d/dx tanh(x) = 1 - tanh(x)^2; we already stored tanh(x) as data.
                    let t = nodes[i].data;
                    nodes[a].grad += (1.0 - t * t) * g;
                }
                Op::Relu(a) => {
                    // d/dx relu(x) = 1 if x > 0 else 0
                    nodes[a].grad += if nodes[a].data > 0.0 { g } else { 0.0 };
                }
                Op::Neg(a) => {
                    // d/dx (-x) = -1
                    nodes[a].grad += -g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerically estimate d(out)/d(input) using central finite differences.
    fn numeric_grad<F>(f: F, x: f64, eps: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        (f(x + eps) - f(x - eps)) / (2.0 * eps)
    }

    #[test]
    fn multi_var_tanh_graph() {
        // Build the expression f(a, b) = tanh(a * b + a), returning the output Value.
        let build = |t: &Tape, a: f64, b: f64| -> (Value, Value, Value) {
            let av = t.value(a);
            let bv = t.value(b);
            let ab = av.mul(&bv);
            let inner = ab.add(&av); // a*b + a
            let out = inner.tanh();
            (out, av, bv)
        };

        // Compute gradients using the tape.
        let tape = Tape::new();
        let a0 = 1.5;
        let b0 = -2.0;
        let (out, a, b) = build(&tape, a0, b0);
        out.backward();
        let ga = a.grad();
        let gb = b.grad();

        // Compute numeric gradients for comparison.
        let eps = 1e-6;
        // Vary a with b constant.
        let na = numeric_grad(
            |x| {
                let t = Tape::new();
                let (o, _, _) = build(&t, x, b0);
                o.data()
            },
            a0,
            eps,
        );
        assert!((ga - na).abs() < 1e-6);
        // Vary b with a constant.
        let nb = numeric_grad(
            |x| {
                let t = Tape::new();
                let (o, _, _) = build(&t, a0, x);
                o.data()
            },
            b0,
            eps,
        );
        assert!((gb - nb).abs() < 1e-6);
    }

    #[test]
    fn single_var_relu_composition_graph() {
        // Build the expression f(x) = relu(x^2 - 3).
        let build = |t: &Tape, x: f64| -> (Value, Value) {
            let xv = t.value(x);
            let x2 = xv.powf(2.0);
            let three = t.value(3.0);
            let shifted = x2.sub(&three);
            let f = shifted.relu();
            (f, xv)
        };
        for &x0 in &[2.5_f64, 1.0_f64] {
            let t = Tape::new();
            // Compute gradients using the tape.
            let (f, x) = build(&t, x0);
            f.backward();
            let analytic = x.grad();
            // Compute numeric gradients for comparison.
            let eps = 1e-6;
            let numeric = numeric_grad(
                |z| {
                    let tt = Tape::new();
                    let (gg, _) = build(&tt, z);
                    gg.data()
                },
                x0,
                eps,
            );
            assert!((analytic - numeric).abs() < 1e-6);
        }
    }

    #[test]
    fn single_var_exp_ln_composition_graph() {
        // Build the expression f(x) = ln(x + exp(2x)).
        let build = |t: &Tape, x: f64| -> (Value, Value) {
            let xv = t.value(x);
            let c = t.value(2_f64);
            let e = xv.mul(&c).exp();
            let out = e.add(&xv).log();
            (out, xv)
        };
        for &x0 in &[0.5_f64, 3.0_f64] {
            let t = Tape::new();
            // Compute gradients using the tape.
            let (f, x) = build(&t, x0);
            f.backward();
            let analytic = x.grad();
            // Compute numeric gradients for comparison.
            let eps = 1e-6;
            let numeric = numeric_grad(
                |z| {
                    let tt = Tape::new();
                    let (gg, _) = build(&tt, z);
                    gg.data()
                },
                x0,
                eps,
            );
            assert!((analytic - numeric).abs() < 1e-6);
        }
    }
}
