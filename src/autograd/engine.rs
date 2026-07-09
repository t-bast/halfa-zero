//! A tensor-valued reverse-mode autograd engine using the arena/tape design.
//!
//! Core idea:
//!   - Every tensor lives in a single `Vec<Node>` (the arena/tape).
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

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// We store the tensor data as a flattened vector (rows are concatenated).
    data: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl Tensor {
    pub fn new(constant: f64, rows: usize, columns: usize) -> Tensor {
        Tensor {
            data: vec![constant; rows * columns],
            rows,
            columns,
        }
    }

    pub fn map_data(&self, f: impl FnMut(&f64) -> f64) -> Tensor {
        Tensor {
            data: self.data.iter().map(f).collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        Tensor {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        Tensor {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }

    /// Element-wise multiplication between two tensors.
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        Tensor {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }

    /// Matrix multiplication between two tensors.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Note that we implement the most basic matrix multiplication algorithm.
        // That's the part which may worth optimizing.
        assert_eq!(self.columns, other.rows);
        let mut result: Vec<f64> = vec![0.0; self.rows * other.columns];
        for r in 0..self.rows {
            for c in 0..other.columns {
                result[r * other.columns + c] = (0..self.columns)
                    .map(|k| self.data[r * self.columns + k] * other.data[k * other.columns + c])
                    .sum();
            }
        }
        Tensor {
            data: result,
            rows: self.rows,
            columns: other.columns,
        }
    }

    pub fn transpose(&self) -> Tensor {
        let mut result = vec![0.0; self.rows * self.columns];
        for r in 0..self.rows {
            for c in 0..self.columns {
                // element [r,c] of self goes to [c,r] of result
                result[c * self.rows + r] = self.data[r * self.columns + c];
            }
        }
        Tensor {
            data: result,
            rows: self.columns,
            columns: self.rows,
        }
    }
}

/// One entry on the tape.
struct Node {
    data: Tensor, // the forward value
    grad: Tensor, // accumulated gradient d(output)/d(this), filled in by backward()
    op: Op,       // how this node was produced (its inputs, if any)
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
    fn push(&self, data: Tensor, op: Op) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        let grad = Tensor::new(0.0, data.rows, data.columns);
        nodes.push(Node { data, grad, op });
        idx
    }

    /// Create a leaf value at the end of the tape.
    pub fn value(&self, data: Tensor) -> Value {
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
    /// Read the current forward value (return a deep copy of it).
    pub fn data(&self) -> Tensor {
        self.tape.nodes.borrow()[self.idx].data.clone()
    }

    /// Apply the following function to each data entry of the current forward value and return
    /// the resulting tensor.
    pub fn map_data(&self, f: impl FnMut(&f64) -> f64) -> Tensor {
        self.tape.nodes.borrow()[self.idx].data.map_data(f)
    }

    /// Read the gradient (which is only computed after calling `backward`).
    pub fn grad(&self) -> Tensor {
        self.tape.nodes.borrow()[self.idx].grad.clone()
    }

    fn op(&self, data: Tensor, op: Op) -> Value {
        let idx = self.tape.push(data, op);
        Value {
            tape: self.tape.clone(),
            idx,
        }
    }

    pub fn add(&self, other: &Value) -> Value {
        self.op(self.data().add(&other.data()), Op::Add(self.idx, other.idx))
    }

    pub fn mul(&self, other: &Value) -> Value {
        self.op(
            self.data().matmul(&other.data()),
            Op::Mul(self.idx, other.idx),
        )
    }

    pub fn sub(&self, other: &Value) -> Value {
        // Note that we decompose (a - b) into two operations (a + (-b)).
        // This ensures that gradients flow through both operations correctly.
        let neg = other.neg();
        self.add(&neg)
    }

    pub fn neg(&self) -> Value {
        self.op(self.map_data(|d| -d), Op::Neg(self.idx))
    }

    pub fn powf(&self, exponent: f64) -> Value {
        self.op(
            self.map_data(|d| d.powf(exponent)),
            Op::Pow(self.idx, exponent),
        )
    }

    pub fn exp(&self) -> Value {
        self.op(self.map_data(|d| d.exp()), Op::Exp(self.idx))
    }

    pub fn log(&self) -> Value {
        self.op(self.map_data(|d| d.ln()), Op::Log(self.idx))
    }

    pub fn tanh(&self) -> Value {
        self.op(self.map_data(|d| d.tanh()), Op::Tanh(self.idx))
    }

    pub fn relu(&self) -> Value {
        self.op(self.map_data(|d| d.max(0.0)), Op::Relu(self.idx))
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
            for i in 0..n.grad.data.len() {
                n.grad.data[i] = 0.0;
            }
        }
        // Set the gradient of the starting node to 1.0.
        for i in 0..nodes[self.idx].grad.data.len() {
            nodes[self.idx].grad.data[i] = 1.0;
        }

        for i in (0..=self.idx).rev() {
            let g = nodes[i].grad.clone();
            match nodes[i].op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    // d/da (a+b) = 1, d/db (a+b) = 1
                    nodes[a].grad = nodes[a].grad.add(&g);
                    nodes[b].grad = nodes[b].grad.add(&g);
                }
                Op::Mul(a, b) => {
                    // dA = d(A*B) * T(B), dB = T(A) * d(A*B)
                    nodes[a].grad = nodes[a].grad.add(&g.matmul(&nodes[b].data.transpose()));
                    nodes[b].grad = nodes[b].grad.add(&nodes[a].data.transpose().matmul(&g));
                }
                Op::Pow(a, p) => {
                    // d/da (a^p) = p * a^(p-1)
                    let pow = nodes[a].data.map_data(|x| p * x.powf(p - 1.0)).hadamard(&g);
                    nodes[a].grad = nodes[a].grad.add(&pow);
                }
                Op::Exp(a) => {
                    // d/dx (exp(x)) = exp(x); we already stored exp(x) as data.
                    let exp = nodes[i].data.hadamard(&g);
                    nodes[a].grad = nodes[a].grad.add(&exp);
                }
                Op::Log(a) => {
                    // d/dx (ln(x)) = 1/x
                    let log = nodes[a].data.map_data(|x| 1.0 / x).hadamard(&g);
                    nodes[a].grad = nodes[a].grad.add(&log);
                }
                Op::Tanh(a) => {
                    // d/dx tanh(x) = 1 - tanh(x)^2; we already stored tanh(x) as data.
                    let tanh = nodes[i].data.map_data(|t| 1.0 - t * t).hadamard(&g);
                    nodes[a].grad = nodes[a].grad.add(&tanh);
                }
                Op::Relu(a) => {
                    // d/dx relu(x) = 1 if x > 0 else 0
                    let relu = nodes[a]
                        .data
                        .map_data(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                        .hadamard(&g);
                    nodes[a].grad = nodes[a].grad.add(&relu);
                }
                Op::Neg(a) => {
                    // d/dx (-x) = -1
                    nodes[a].grad = nodes[a].grad.sub(&g);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_basic_operations() {
        let t1 = Tensor {
            #[rustfmt::skip]
            data: vec![
                1_f64, 1_f64, 1_f64, 1_f64, 1_f64,
                2_f64, 2_f64, 2_f64, 2_f64, 2_f64,
                3_f64, 3_f64, 3_f64, 3_f64, 3_f64,
            ],
            rows: 3,
            columns: 5,
        };
        let t2 = Tensor {
            #[rustfmt::skip]
            data: vec![
                1_f64, 2_f64, 3_f64, 4_f64, 5_f64,
                6_f64, 7_f64, 8_f64, 9_f64, 10_f64,
                11_f64, 12_f64, 13_f64, 14_f64, 15_f64,
            ],
            rows: 3,
            columns: 5,
        };
        let expected_add = Tensor {
            #[rustfmt::skip]
            data: vec![
                2_f64, 3_f64, 4_f64, 5_f64, 6_f64,
                8_f64, 9_f64, 10_f64, 11_f64, 12_f64,
                14_f64, 15_f64, 16_f64, 17_f64, 18_f64,
            ],
            rows: 3,
            columns: 5,
        };
        assert_eq!(expected_add, t1.add(&t2));
        let expected_sub = Tensor {
            #[rustfmt::skip]
            data: vec![
                0_f64, 1_f64, 2_f64, 3_f64, 4_f64,
                4_f64, 5_f64, 6_f64, 7_f64, 8_f64,
                8_f64, 9_f64, 10_f64, 11_f64, 12_f64,
            ],
            rows: 3,
            columns: 5,
        };
        assert_eq!(expected_sub, t2.sub(&t1));
        let expected_hadamard = Tensor {
            #[rustfmt::skip]
            data: vec![
                1_f64, 2_f64, 3_f64, 4_f64, 5_f64,
                12_f64, 14_f64, 16_f64, 18_f64, 20_f64,
                33_f64, 36_f64, 39_f64, 42_f64, 45_f64,
            ],
            rows: 3,
            columns: 5,
        };
        assert_eq!(expected_hadamard, t1.hadamard(&t2));
        let expected_transpose = Tensor {
            #[rustfmt::skip]
            data: vec![
                1_f64, 6_f64, 11_f64,
                2_f64, 7_f64, 12_f64,
                3_f64, 8_f64, 13_f64,
                4_f64, 9_f64, 14_f64,
                5_f64, 10_f64, 15_f64,
            ],
            rows: 5,
            columns: 3,
        };
        assert_eq!(expected_transpose, t2.transpose());
        let expected_map = Tensor {
            #[rustfmt::skip]
            data: vec![
                1.5_f64, 1.5_f64, 1.5_f64, 1.5_f64, 1.5_f64,
                3_f64, 3_f64, 3_f64, 3_f64, 3_f64,
                4.5_f64, 4.5_f64, 4.5_f64, 4.5_f64, 4.5_f64,
            ],
            rows: 3,
            columns: 5,
        };
        assert_eq!(expected_map, t1.map_data(|x| x + x * 0.5))
    }

    #[test]
    fn tensor_mul() {
        let t1 = Tensor {
            #[rustfmt::skip]
            data: vec![
                7_f64, 1_f64, 2_f64,
                5_f64, 3_f64, 2_f64,
                3_f64, 3_f64, 4_f64,
            ],
            rows: 3,
            columns: 3,
        };
        let expected_square = Tensor {
            #[rustfmt::skip]
            data: vec![
                60_f64, 16_f64, 24_f64,
                56_f64, 20_f64, 24_f64,
                48_f64, 24_f64, 28_f64,
            ],
            rows: 3,
            columns: 3,
        };
        assert_eq!(expected_square, t1.matmul(&t1));
        let t2 = Tensor {
            #[rustfmt::skip]
            data: vec![
                7_f64, 1_f64, 2_f64,
                5_f64, 3_f64, 2_f64,
            ],
            rows: 2,
            columns: 3,
        };
        let t3 = Tensor {
            #[rustfmt::skip]
            data: vec![
                3_f64, 6_f64,
                5_f64, 5_f64,
                8_f64, 2_f64,
            ],
            rows: 3,
            columns: 2,
        };
        let expected_mul = Tensor {
            #[rustfmt::skip]
            data: vec![
                42_f64, 51_f64,
                46_f64, 49_f64,
            ],
            rows: 2,
            columns: 2,
        };
        assert_eq!(expected_mul, t2.matmul(&t3));
    }

    /// Sum all elements of a tensor to a single scalar: used to compare tensor gradients computed
    /// by the engine with finite differences.
    fn sum(t: &Tensor) -> f64 {
        t.data.iter().sum()
    }

    /// Numerically estimate d(out)/d(input) using central finite differences with regards to each
    /// element of each tensor, while keeping the other tensor fixed.
    fn compare_multi_var_numeric_grad<F>(
        f: F,
        a0: &Tensor,
        b0: &Tensor,
        eps: f64,
        ga: &Tensor,
        gb: &Tensor,
    ) -> ()
    where
        F: Fn(&Tape, Tensor, Tensor) -> Value,
    {
        // Compute numeric gradients gradient w.r.t. each element of A.
        for idx in 0..a0.data.len() {
            let plus = {
                let mut a = a0.clone();
                a.data[idx] += eps;
                let t = Tape::new();
                let o = f(&t, a, b0.clone());
                sum(&o.data())
            };
            let minus = {
                let mut a = a0.clone();
                a.data[idx] -= eps;
                let t = Tape::new();
                let o = f(&t, a, b0.clone());
                sum(&o.data())
            };
            let numeric = (plus - minus) / (2.0 * eps);
            assert!((ga.data[idx] - numeric).abs() < 1e-5);
        }
        // Compute numeric gradients gradient w.r.t. each element of B.
        for idx in 0..b0.data.len() {
            let plus = {
                let mut b = b0.clone();
                b.data[idx] += eps;
                let t = Tape::new();
                let o = f(&t, a0.clone(), b);
                sum(&o.data())
            };
            let minus = {
                let mut b = b0.clone();
                b.data[idx] -= eps;
                let t = Tape::new();
                let o = f(&t, a0.clone(), b);
                sum(&o.data())
            };
            let numeric = (plus - minus) / (2.0 * eps);
            assert!((gb.data[idx] - numeric).abs() < 1e-5);
        }
    }

    #[test]
    fn multi_var_tanh_graph() {
        // f(A, B) = sum(tanh(A * B + A)), where * is matrix multiplication.
        // A is [2,2], B is [2,2], so A*B is [2,2] and A*B + A is well-shaped.
        let build = |t: &Tape, a: Tensor, b: Tensor| -> (Value, Value, Value) {
            let av = t.value(a);
            let bv = t.value(b);
            let ab = av.mul(&bv);
            let inner = ab.add(&av); // a*b + a
            let out = inner.tanh();
            (out, av, bv)
        };
        let a0 = Tensor {
            data: vec![1.5, -0.5, 0.3, 2.0],
            rows: 2,
            columns: 2,
        };
        let b0 = Tensor {
            data: vec![-2.0, 1.0, 0.7, -1.3],
            rows: 2,
            columns: 2,
        };

        // Compute gradients using the tape.
        let tape = Tape::new();
        let (out, a, b) = build(&tape, a0.clone(), b0.clone());
        out.backward();
        let ga = a.grad();
        let gb = b.grad();
        // Compute numeric gradients for comparison.
        compare_multi_var_numeric_grad(|t, a, b| build(t, a, b).0, &a0, &b0, 1e-6, &ga, &gb);
    }

    #[test]
    fn multi_var_multiply_graph() {
        // f(A, B) = sum(tanh(A) * exp(B) + A * (B^2)) where * is matrix multiplication and A and B
        // have compatible dimensions for multiplication.
        let build = |t: &Tape, a: Tensor, b: Tensor| -> (Value, Value, Value) {
            let av = t.value(a);
            let bv = t.value(b);
            let left = av.tanh().mul(&bv.exp()); // tanh(A) * exp(B)
            let right = av.mul(&bv.powf(2.0)); // A * (B^2)
            let out = left.add(&right);
            (out, av, bv)
        };
        let a0 = Tensor {
            data: vec![1.5, -0.5, 0.3, 2.0, -0.4, 0.6],
            rows: 2,
            columns: 3,
        };
        let b0 = Tensor {
            data: vec![-2.1, 1.4, 0.3, -0.9, 1.8, 1.8],
            rows: 3,
            columns: 2,
        };

        // Compute gradients using the tape.
        let tape = Tape::new();
        let (out, a, b) = build(&tape, a0.clone(), b0.clone());
        out.backward();
        let ga = a.grad();
        let gb = b.grad();
        // Compute numeric gradients for comparison.
        compare_multi_var_numeric_grad(|t, a, b| build(t, a, b).0, &a0, &b0, 1e-6, &ga, &gb);
    }

    /// Numerically estimate d(out)/d(input) using central finite differences with regards to each
    /// element of the tensor.
    fn compare_single_var_numeric_grad<F>(f: F, a0: &Tensor, eps: f64, analytic: &Tensor) -> ()
    where
        F: Fn(&Tape, Tensor) -> Value,
    {
        for idx in 0..a0.data.len() {
            let plus = {
                let mut a = a0.clone();
                a.data[idx] += eps;
                let t = Tape::new();
                let o = f(&t, a);
                sum(&o.data())
            };
            let minus = {
                let mut a = a0.clone();
                a.data[idx] -= eps;
                let t = Tape::new();
                let o = f(&t, a);
                sum(&o.data())
            };
            let numeric = (plus - minus) / (2.0 * eps);
            assert!((analytic.data[idx] - numeric).abs() < 1e-5);
        }
    }

    #[test]
    fn single_var_relu_composition_graph() {
        // Build the expression f(A) = relu(A^2 - 3).
        let build = |t: &Tape, a: Tensor| -> (Value, Value) {
            let av = t.value(a.clone());
            let x2 = av.powf(2.0);
            let three = t.value(Tensor {
                data: vec![3.0; a.data.len()],
                rows: a.rows,
                columns: a.columns,
            });
            let shifted = x2.sub(&three);
            let f = shifted.relu();
            (f, av)
        };
        let a0 = Tensor {
            data: vec![2.5, 1.0, -0.5, 1.7, -1.4, 0.2],
            rows: 2,
            columns: 3,
        };

        // Compute gradients using the tape.
        let t = Tape::new();
        let (f, a) = build(&t, a0.clone());
        f.backward();
        let analytic = a.grad();
        // Compute numeric gradients for comparison.
        compare_single_var_numeric_grad(|t, a| build(t, a).0, &a0, 1e-6, &analytic);
    }

    #[test]
    fn single_var_exp_ln_composition_graph() {
        // Build the expression f(A) = ln(A + exp(A)).
        let build = |t: &Tape, a: Tensor| -> (Value, Value) {
            let av = t.value(a.clone());
            let e = av.exp();
            let out = e.add(&av).log();
            (out, av)
        };
        let a0 = Tensor {
            data: vec![1.73, 0.81, 0.64, 1.32, 2.13, 0.27],
            rows: 3,
            columns: 2,
        };

        // Compute gradients using the tape.
        let t = Tape::new();
        let (f, a) = build(&t, a0.clone());
        f.backward();
        let analytic = a.grad();
        // Compute numeric gradients for comparison.
        compare_single_var_numeric_grad(|t, a| build(t, a).0, &a0, 1e-6, &analytic);
    }
}
