// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The Iceberg expression model: the spec's predicate tree, addressed by
//! field id, with three-valued evaluation.
//!
//! Evaluation is the scan-planning contract: an expression evaluates against
//! a partition value resolver to `Some(true)` / `Some(false)` / `None`
//! (unknown — a null operand or an unresolvable field). A file is pruned
//! only when the expression is definitively `Some(false)`; `None` keeps the
//! file, because a null partition value may still contain matching rows.

use serde::{Deserialize, Serialize};

/// A spec-typed literal. The variants follow the Iceberg type system's
/// primitive types; the numeric variants map to the spec's `int`, `long`,
/// `float`, `double`, and the date/time variants carry the spec's epoch
/// conventions (days for `date`, microseconds for `timestamp`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Boolean(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    /// Days since the epoch (the `date` type).
    Date(i32),
    /// Microseconds since the epoch (the `timestamp` type).
    TimestampMicros(i64),
    /// Nanoseconds since the epoch (the `timestamptz`/`timestamp_ns` family).
    TimestampNanos(i64),
    /// Raw bytes (the `binary`/`fixed` types).
    Binary(Vec<u8>),
}

impl Literal {
    /// Total order within a variant family. Comparing literals of different
    /// variants is `None` — the spec types are not cross-comparable, and a
    /// caller that produced such a comparison has a type error to fix.
    pub fn cmp_same_type(&self, other: &Literal) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Literal::Boolean(a), Literal::Boolean(b)) => Some(a.cmp(b)),
            (Literal::Int(a), Literal::Int(b)) => Some(a.cmp(b)),
            (Literal::Long(a), Literal::Long(b)) => Some(a.cmp(b)),
            (Literal::Float(a), Literal::Float(b)) => a.partial_cmp(b),
            (Literal::Double(a), Literal::Double(b)) => a.partial_cmp(b),
            (Literal::String(a), Literal::String(b)) => Some(a.cmp(b)),
            (Literal::Date(a), Literal::Date(b)) => Some(a.cmp(b)),
            (Literal::TimestampMicros(a), Literal::TimestampMicros(b)) => Some(a.cmp(b)),
            (Literal::TimestampNanos(a), Literal::TimestampNanos(b)) => Some(a.cmp(b)),
            (Literal::Binary(a), Literal::Binary(b)) => Some(a.cmp(b)),
            // Int promotes to Long within the integer family: the spec's
            // `int` and `long` share an ordering, and a bound written as
            // `int` against a `long` column is a common and legal case.
            (Literal::Int(a), Literal::Long(b)) => Some((*a as i64).cmp(b)),
            (Literal::Long(a), Literal::Int(b)) => Some(a.cmp(&(*b as i64))),
            _ => None,
        }
    }
}

/// The term side of a predicate. The spec addresses fields by id; name
/// resolution happens once at the boundary (the schema), never inside
/// evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Term {
    ByFieldId(i32),
}

/// The comparison and membership operators the spec defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredicateOp {
    IsNull,
    NotNull,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eq,
    NotEq,
    In,
    NotIn,
    StartsWith,
    NotStartsWith,
}

/// A leaf predicate: one operator, one term, and the comparison operand(s).
/// `In`/`NotIn` carry the set; the null checks carry no literal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    Unary {
        op: PredicateOp,
        term: Term,
    },
    Binary {
        op: PredicateOp,
        term: Term,
        literal: Literal,
    },
    Set {
        op: PredicateOp,
        term: Term,
        literals: Vec<Literal>,
    },
}

/// The spec's boolean expression tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    AlwaysTrue,
    AlwaysFalse,
    Not(Box<Expression>),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Predicate(Predicate),
}

impl Expression {
    /// Build `term op literal`.
    pub fn predicate(op: PredicateOp, field_id: i32, literal: Literal) -> Self {
        Self::Predicate(Predicate::Binary {
            op,
            term: Term::ByFieldId(field_id),
            literal,
        })
    }

    /// Build `term IN (literals)`.
    pub fn in_set(field_id: i32, literals: Vec<Literal>) -> Self {
        Self::Predicate(Predicate::Set {
            op: PredicateOp::In,
            term: Term::ByFieldId(field_id),
            literals,
        })
    }

    /// Three-valued evaluation against a partition value resolver.
    ///
    /// `resolve` maps a field id to the partition's literal for that field
    /// (the ALREADY TRANSFORMED value — a partition's stored value is the
    /// transform applied to the source column), or `None` when the field is
    /// not a partition key or its value is null.
    ///
    /// The result is `Some(false)` only when the expression is definitively
    /// false; `None` (unknown) propagates through the boolean operators by
    /// the standard three-valued algebra, and callers prune only on
    /// `Some(false)`.
    pub fn test(&self, resolve: &mut impl FnMut(i32) -> Option<Literal>) -> Option<bool> {
        match self {
            Self::AlwaysTrue => Some(true),
            Self::AlwaysFalse => Some(false),
            Self::Not(inner) => inner.test(resolve).map(|b| !b),
            Self::And(a, b) => {
                // Three-valued AND: false dominates, then unknown.
                match (a.test(resolve), b.test(resolve)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Self::Or(a, b) => {
                // Three-valued OR: true dominates, then unknown.
                match (a.test(resolve), b.test(resolve)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            Self::Predicate(p) => p.test(resolve),
        }
    }
}

impl Predicate {
    pub fn test(&self, resolve: &mut impl FnMut(i32) -> Option<Literal>) -> Option<bool> {
        match self {
            Self::Unary { op, term } => {
                // The resolver returns None for a null or absent value, so a
                // resolved value is non-null by construction.
                resolve_field(term, resolve)?;
                match op {
                    PredicateOp::IsNull => Some(false),
                    PredicateOp::NotNull => Some(true),
                    _ => None, // a unary op with a non-null value is unanswerable
                }
            }
            Self::Binary { op, term, literal } => {
                let v = resolve_field(term, resolve)?;
                let ord = v.cmp_same_type(literal)?;
                match op {
                    PredicateOp::Lt => Some(ord == std::cmp::Ordering::Less),
                    PredicateOp::LtEq => Some(ord != std::cmp::Ordering::Greater),
                    PredicateOp::Gt => Some(ord == std::cmp::Ordering::Greater),
                    PredicateOp::GtEq => Some(ord != std::cmp::Ordering::Less),
                    PredicateOp::Eq => Some(ord == std::cmp::Ordering::Equal),
                    PredicateOp::NotEq => Some(ord != std::cmp::Ordering::Equal),
                    PredicateOp::StartsWith | PredicateOp::NotStartsWith => match (&v, literal) {
                        (Literal::String(s), Literal::String(prefix)) => {
                            let starts = s.starts_with(prefix.as_str());
                            Some(if *op == PredicateOp::StartsWith {
                                starts
                            } else {
                                !starts
                            })
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            Self::Set { op, term, literals } => {
                let v = resolve_field(term, resolve)?;
                let contained = literals
                    .iter()
                    .any(|l| v.cmp_same_type(l) == Some(std::cmp::Ordering::Equal));
                match op {
                    PredicateOp::In => Some(contained),
                    PredicateOp::NotIn => Some(!contained),
                    _ => None,
                }
            }
        }
    }
}

fn resolve_field(term: &Term, resolve: &mut impl FnMut(i32) -> Option<Literal>) -> Option<Literal> {
    match term {
        Term::ByFieldId(id) => resolve(*id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Partition values for the tests: field 1 = int 5, field 2 = "abc",
    // field 3 absent (not a partition key).
    fn resolver() -> impl FnMut(i32) -> Option<Literal> {
        let mut values = std::collections::HashMap::new();
        values.insert(1, Literal::Int(5));
        values.insert(2, Literal::String("abc".to_string()));
        move |id: i32| values.get(&id).cloned()
    }

    #[test]
    fn eq_prunes_definitively() {
        let e = Expression::predicate(PredicateOp::Eq, 1, Literal::Int(5));
        assert_eq!(e.test(&mut resolver()), Some(true));
        let e = Expression::predicate(PredicateOp::Eq, 1, Literal::Int(6));
        assert_eq!(e.test(&mut resolver()), Some(false));
    }

    #[test]
    fn int_promotes_to_long() {
        let e = Expression::predicate(PredicateOp::Lt, 1, Literal::Long(10));
        assert_eq!(e.test(&mut resolver()), Some(true));
    }

    #[test]
    fn unresolvable_field_is_unknown_not_false() {
        // Field 3 is not a partition key: the file must be KEPT.
        let e = Expression::predicate(PredicateOp::Eq, 3, Literal::Int(5));
        assert_eq!(e.test(&mut resolver()), None);
    }

    #[test]
    fn and_or_propagate_unknown() {
        // true AND unknown = unknown (the file is kept).
        let e = Expression::And(
            Box::new(Expression::predicate(PredicateOp::Eq, 1, Literal::Int(5))),
            Box::new(Expression::predicate(PredicateOp::Eq, 3, Literal::Int(5))),
        );
        assert_eq!(e.test(&mut resolver()), None);
        // true OR unknown = true (the file is pruned).
        let e = Expression::Or(
            Box::new(Expression::predicate(PredicateOp::Eq, 1, Literal::Int(5))),
            Box::new(Expression::predicate(PredicateOp::Eq, 3, Literal::Int(5))),
        );
        assert_eq!(e.test(&mut resolver()), Some(true));
        // false AND anything = false.
        let e = Expression::And(
            Box::new(Expression::predicate(PredicateOp::Eq, 1, Literal::Int(6))),
            Box::new(Expression::predicate(PredicateOp::Eq, 3, Literal::Int(5))),
        );
        assert_eq!(e.test(&mut resolver()), Some(false));
    }

    #[test]
    fn not_flips_unknown_to_unknown() {
        let e = Expression::Not(Box::new(Expression::predicate(
            PredicateOp::Eq,
            3,
            Literal::Int(5),
        )));
        assert_eq!(e.test(&mut resolver()), None);
    }

    #[test]
    fn in_and_not_in() {
        let e = Expression::in_set(1, vec![Literal::Int(4), Literal::Int(5)]);
        assert_eq!(e.test(&mut resolver()), Some(true));
        let e = Expression::in_set(1, vec![Literal::Int(4), Literal::Int(6)]);
        assert_eq!(e.test(&mut resolver()), Some(false));
        let e = Expression::Predicate(Predicate::Set {
            op: PredicateOp::NotIn,
            term: Term::ByFieldId(1),
            literals: vec![Literal::Int(5)],
        });
        assert_eq!(e.test(&mut resolver()), Some(false));
    }

    #[test]
    fn starts_with_on_strings() {
        let e = Expression::predicate(
            PredicateOp::StartsWith,
            2,
            Literal::String("ab".to_string()),
        );
        assert_eq!(e.test(&mut resolver()), Some(true));
        let e = Expression::predicate(
            PredicateOp::NotStartsWith,
            2,
            Literal::String("ab".to_string()),
        );
        assert_eq!(e.test(&mut resolver()), Some(false));
    }

    #[test]
    fn cross_type_comparison_is_unknown() {
        // A string literal against an int partition value: not comparable.
        let e = Expression::predicate(PredicateOp::Eq, 1, Literal::String("5".to_string()));
        assert_eq!(e.test(&mut resolver()), None);
    }

    #[test]
    fn null_check_on_resolved_value() {
        // A resolved (non-null) partition value answers IsNull definitively:
        // the resolver returns None for null/absent, so a resolved value is
        // non-null by construction.
        let e = Expression::Predicate(Predicate::Unary {
            op: PredicateOp::IsNull,
            term: Term::ByFieldId(1),
        });
        assert_eq!(e.test(&mut resolver()), Some(false));
        let e = Expression::Predicate(Predicate::Unary {
            op: PredicateOp::NotNull,
            term: Term::ByFieldId(1),
        });
        assert_eq!(e.test(&mut resolver()), Some(true));
    }
}
