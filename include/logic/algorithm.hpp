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
CUDA TFormula<Allocator> make_v_op_z(LVar<Allocator> v, Sig sig, logic_int z, AType aty = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_lvar(UNTYPED, std::move(v)), sig, F::make_z(z), UNTYPED, a, allocator);
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
    case F::Z:
    case F::R:
    case F::S:
      return 0;
    case F::V:
    case F::E:
    case F::LV:
      return 1;
    case F::Seq: return impl::num_vars_in_seq<F::Seq>(f);
    case F::ESeq: return impl::num_vars_in_seq<F::ESeq>(f);
    default: printf("num_vars: formula not handled.\n"); assert(false); return 0;
  }
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
    F::make_nary(AND, std::move(fty), ty, f.approx()),
    F::make_nary(AND, std::move(other), other_ty, f.approx()));
}

}

#endif
