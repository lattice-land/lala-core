// Copyright 2022 Pierre Talbot

#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include "ast.hpp"

namespace lala {

/** \return `true` if the formula `f` has the shape `variable <sig> constant`. */
template<class Allocator, class ExtendedSig>
CUDA bool is_v_op_constant(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is_constant();
}

/** \return `true` if the formula `f` has the shape `variable op integer constant`, e.g., `x < 4`. */
template<class Allocator, class ExtendedSig>
CUDA bool is_v_op_z(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is(F::Z);
}

template<typename Allocator>
CUDA TFormula<Allocator> make_v_op_z(LVar<Allocator> v, Sig sig, logic_int z, AType aty = UNTYPED, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_lvar(UNTYPED, std::move(v)), sig, F::make_z(z), aty, allocator);
}

template<typename Allocator>
CUDA TFormula<Allocator> make_v_op_z(AVar v, Sig sig, logic_int z, AType aty = UNTYPED, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_avar(v), sig, F::make_z(z), aty, allocator);
}

namespace impl {
  template<typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found);

  template<size_t n, typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& find_var_in_seq(const TFormula<Allocator, ExtendedSig>& f, bool& found) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    for(int i = 0; i < children.size(); ++i) {
      const auto& subformula = var_in_impl(children[i], found);
      if(found) {
        return subformula;
      }
    }
    return f;
  }

  template<typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found)
  {
    using F = TFormula<Allocator, ExtendedSig>;
    switch(f.index()) {
      case F::Z:
      case F::R:
      case F::S:
        return f;
      case F::V:
      case F::E:
      case F::LV:
        found = true;
        return f;
      case F::Seq: return find_var_in_seq<F::Seq>(f, found);
      case F::ESeq: return find_var_in_seq<F::ESeq>(f, found);
      default: printf("var_in: formula not handled.\n"); assert(false); return f;
    }
  }

  template<size_t n, class F>
  CUDA int num_vars_in_seq(const F& f) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    int total = 0;
    for(int i = 0; i < children.size(); ++i) {
      total += num_vars(children[i]);
    }
    return total;
  }

  template<class F>
  CUDA int num_qf_vars(const F& f, bool must_be_untyped);

  template<size_t n, class F>
  CUDA int num_qf_vars_in_seq(const F& f, bool must_be_untyped) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    int total = 0;
    for(int i = 0; i < children.size(); ++i) {
      total += num_qf_vars(children[i], must_be_untyped);
    }
    return total;
  }

  template<class F>
  CUDA int num_qf_vars(const F& f, bool must_be_untyped) {
    switch(f.index()) {
      case F::E:
        if(must_be_untyped) {
          return f.type() == UNTYPED ? 1 : 0;
        }
        else {
          return 1;
        }
      case F::Seq: return impl::num_qf_vars_in_seq<F::Seq>(f, must_be_untyped);
      case F::ESeq: return impl::num_qf_vars_in_seq<F::ESeq>(f, must_be_untyped);
      default: return 0;
    }
  }
}

/** \return The first variable occurring in the formula, or any other subformula if the formula does not contain a variable.
    It returns either a logical variable, an abstract variable or a quantifier. */
template<typename Allocator, typename ExtendedSig>
CUDA const TFormula<Allocator, ExtendedSig>& var_in(const TFormula<Allocator, ExtendedSig>& f) {
  bool found = false;
  return impl::var_in_impl(f, found);
}

/** \return The number of variables occurring in the formula `F` including existential quantifier, logical variables and abstract variables.
 * Each occurrence of a variable is added up (duplicates are counted). */
template<class F>
CUDA int num_vars(const F& f)
{
  switch(f.index()) {
    case F::V:
    case F::E:
    case F::LV:
      return 1;
    case F::Seq: return impl::num_vars_in_seq<F::Seq>(f);
    case F::ESeq: return impl::num_vars_in_seq<F::ESeq>(f);
    default: return 0;
  }
}

/** \return The number of existential quantifiers. */
template<class F>
CUDA int num_quantified_vars(const F& f) {
  return impl::num_qf_vars(f, false);
}

/** \return The number of variables occurring in an existential quantifier that are untyped. */
template<class F>
CUDA int num_quantified_untyped_vars(const F& f) {
  return impl::num_qf_vars(f, true);
}

template<class F>
CUDA AType type_of_conjunction(const typename F::Sequence& seq) {
  AType ty = UNTYPED;
  for(int i = 0; i < seq.size(); ++i) {
    if(seq[i].type() == UNTYPED) {
      return UNTYPED;
    }
    else if(ty == UNTYPED) {
      ty = seq[i].type();
    }
    else if(ty != seq[i].type()) {
      return UNTYPED;
    }
  }
  return ty;
}

/** Given a conjunctive formula `f` of the form \f$ c_1 \land ... \land c_n \f$, it returns a pair \f$ \langle c_i \land .. \land c_j, c_k \land ... \land c_l \rangle \f$ such that the first component contains all formulas with the type `ty`, and the second component, all other formulas. */
template<class F>
CUDA battery::tuple<F,F> extract_ty(const F& f, AType ty) {
  assert(f.is(F::Seq));
  const auto& seq = f.seq();
  typename F::Sequence fty;
  typename F::Sequence other;
  for(int i = 0; i < seq.size(); ++i) {
    // In case of nested conjunction.
    if(seq[i].is(F::Seq) && seq[i].sig() == AND) {
      auto r = extract_ty(seq[i], ty);
      auto& fty_ = battery::get<0>(r).seq();
      auto& other_ = battery::get<1>(r).seq();
      for(int i = 0; i < fty_.size(); ++i) {
        fty.push_back(std::move(fty_[i]));
      }
      for(int i = 0; i < other_.size(); ++i) {
        other.push_back(std::move(other_[i]));
      }
    }
    else if(seq[i].type() == ty) {
      fty.push_back(seq[i]);
    }
    else {
      other.push_back(seq[i]);
    }
  }
  AType other_ty = type_of_conjunction<F>(other);
  return battery::make_tuple(
    F::make_nary(AND, std::move(fty), ty),
    F::make_nary(AND, std::move(other), other_ty));
}

template <class F>
CUDA thrust::optional<F> negate(const F& f);

/** not(f1 \/ ... \/ fn) --> not(f1) /\ ... /\ not(fn)
    not(f1 /\ ... /\ fn) --> not(f1) \/ ... \/ not(fn) */
template <class F>
CUDA thrust::optional<F> de_morgan_law(Sig sig_neg, const F& f) {
  auto seq = f.seq();
  typename F::Sequence neg_seq(seq.size());
  for(int i = 0; i < f.seq().size(); ++i) {
    auto neg_i = negate(seq[i]);
    if(neg_i.has_value()) {
      neg_seq[i] = *neg_i;
    }
    else {
      return {};
    }
  }
  return F::make_nary(sig_neg, neg_seq, f.type());
}

template <class F>
CUDA thrust::optional<F> negate(const F& f) {
  if(f.is(F::Seq)) {
    Sig neg_sig;
    switch(f.sig()) {
      // Total order predicates can be reversed.
      case LEQ: neg_sig = GT; break;
      case GEQ: neg_sig = LT; break;
      case EQ: neg_sig = NEQ; break;
      case NEQ: neg_sig = EQ; break;
      case LT: neg_sig = GEQ; break;
      case GT: neg_sig = LEQ; break;
      // Partial order predicates cannot simply be reversed.
      case SUBSET:
      case SUBSETEQ:
      case SUPSET:
      case SUPSETEQ:
        return F::make_unary(NOT, f, f.type());
      case AND:
        return de_morgan_law(OR, f);
      case OR:
        return de_morgan_law(AND, f);
      default:
        return {};
    }
    return F::make_nary(neg_sig, f.seq(), f.type());
  }
  return {};
}

/** True for the operators <=, <, >, >=, =, !=, subset, subseteq, supset, supseteq */
template <class F>
CUDA bool is_comparison(const F& f) {
  if(f.is(F::Seq)) {
    switch(f.sig()) {
      case LEQ:
      case GEQ:
      case LT:
      case GT:
      case EQ:
      case NEQ:
      case SUBSET:
      case SUBSETEQ:
      case SUPSET:
      case SUPSETEQ:
        return true;
    }
  }
  return false;
}

/** Returns the converse of a comparison operator (see `is_comparison`). */
CUDA inline Sig converse_comparison(Sig sig) {
  switch(sig) {
    case LEQ: return GEQ;
    case GEQ: return LEQ;
    case GT: return LT;
    case LT: return GT;
    case EQ: return sig;
    case NEQ: return sig;
    case SUBSET: return SUPSET;
    case SUBSETEQ: return SUPSETEQ;
    case SUPSET: return SUBSET;
    case SUPSETEQ: return SUBSETEQ;
    default:
      assert(false); // converse not supported for all operators.
      break;
  }
  return sig;
}

/** Given a predicate of the form `t <op> u` (e.g., `x + y <= z + 4`), it transforms it into an equivalent predicate of the form `s <op> k` where `k` is a constant (e.g., `x + y - (z + 4) <= 0`).
If the formula is not a predicate, it is returned unchanged. */
template <class F>
CUDA F move_constants_on_rhs(const F& f) {
  if(is_comparison(f) && !f.seq(1).is_constant()) {
    AType aty = f.type();
    if(f.seq(0).is_constant()) {
      return F::make_binary(f.seq(1), converse_comparison(f.sig()), f.seq(0), aty);
    }
    else {
      return F::make_binary(
        F::make_binary(f.seq(0), SUB, f.seq(1), aty),
        f.sig(),
        F::make_z(0),
        aty);
    }
  }
  return f;
}

}

#endif
