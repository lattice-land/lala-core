// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_ALGORITHM_HPP
#define LALA_CORE_ALGORITHM_HPP

#include "ast.hpp"

namespace lala {

/** \return `true` if the formula `f` has the shape `variable <sig> constant`. */
template<class Allocator, class ExtendedSig>
CUDA NI bool is_v_op_constant(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is_constant();
}

/** \return `true` if the formula `f` has the shape `variable op integer constant`, e.g., `x < 4`. */
template<class Allocator, class ExtendedSig>
CUDA NI bool is_v_op_z(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is(F::Z);
}

/** \return `true` if the formula `f` has the shape `variable = variable` or `variable <=> variable`. */
template<class Allocator, class ExtendedSig>
CUDA NI bool is_var_equality(const TFormula<Allocator, ExtendedSig>& f) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && (f.sig() == EQ || f.sig() == EQUIV)
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && (f.seq(1).is(F::LV) || f.seq(1).is(F::V));
}

template<class Allocator>
CUDA NI TFormula<Allocator> make_v_op_z(LVar<Allocator> v, Sig sig, logic_int z, AType aty = UNTYPED, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_lvar(UNTYPED, std::move(v)), sig, F::make_z(z), aty, allocator);
}

template<class Allocator>
CUDA NI TFormula<Allocator> make_v_op_z(AVar v, Sig sig, logic_int z, AType aty = UNTYPED, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_avar(v), sig, F::make_z(z), aty, allocator);
}

template <class Allocator>
CUDA Sig geq_of_constant(const TFormula<Allocator>& f) {
  using F = TFormula<Allocator>;
  assert(f.is(F::S) || f.is(F::Z) || f.is(F::R));
  return f.is(F::S) ? SUPSETEQ : GEQ;
}

template <class Allocator>
CUDA Sig leq_of_constant(const TFormula<Allocator>& f) {
  using F = TFormula<Allocator>;
  assert(f.is(F::S) || f.is(F::Z) || f.is(F::R));
  return f.is(F::S) ? SUBSETEQ : LEQ;
}

namespace impl {
  template<class Allocator, class ExtendedSig>
  CUDA NI const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found);

  template<size_t n, class Allocator, class ExtendedSig>
  CUDA NI const TFormula<Allocator, ExtendedSig>& find_var_in_seq(const TFormula<Allocator, ExtendedSig>& f, bool& found) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    for(int i = 0; i < children.size(); ++i) {
      const auto& subformula = var_in_impl(children[i], found);
      if(found) {
        return subformula;
      }
    }
    return f;
  }

  template<class Allocator, class ExtendedSig>
  CUDA NI const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found)
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
  CUDA NI int num_vars_in_seq(const F& f) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    int total = 0;
    for(int i = 0; i < children.size(); ++i) {
      total += num_vars(children[i]);
    }
    return total;
  }

  template<class F>
  CUDA NI int num_qf_vars(const F& f, bool type_filter, AType aty);

  template<size_t n, class F>
  CUDA NI int num_qf_vars_in_seq(const F& f, bool type_filter, AType aty) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    int total = 0;
    for(int i = 0; i < children.size(); ++i) {
      total += num_qf_vars(children[i], type_filter, aty);
    }
    return total;
  }

  template<class F>
  CUDA NI int num_qf_vars(const F& f, bool type_filter, AType aty) {
    switch(f.index()) {
      case F::E:
        if(type_filter) {
          return f.type() == aty ? 1 : 0;
        }
        else {
          return 1;
        }
      case F::Seq: return impl::num_qf_vars_in_seq<F::Seq>(f, type_filter, aty);
      case F::ESeq: return impl::num_qf_vars_in_seq<F::ESeq>(f, type_filter, aty);
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
CUDA NI int num_vars(const F& f)
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
  return impl::num_qf_vars(f, false, UNTYPED);
}

/** \return The number of variables occurring in an existential quantifier that have type `aty`. */
template<class F>
CUDA int num_quantified_vars(const F& f, AType aty) {
  return impl::num_qf_vars(f, true, aty);
}

template<class F>
CUDA NI AType type_of_conjunction(const typename F::Sequence& seq) {
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
CUDA NI battery::tuple<F,F> extract_ty(const F& f, AType ty) {
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
CUDA NI thrust::optional<F> negate(const F& f);

/** not(f1 \/ ... \/ fn) --> not(f1) /\ ... /\ not(fn)
    not(f1 /\ ... /\ fn) --> not(f1) \/ ... \/ not(fn) */
template <class F>
CUDA NI thrust::optional<F> de_morgan_law(Sig sig_neg, const F& f) {
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
CUDA NI thrust::optional<F> negate_eq(const F& f) {
  assert(f.is_binary() && f.sig() == EQ);
  if(f.seq(0).is(F::B)) {
    auto b = f.seq(0);
    b.b() = !b.b();
    return F::make_binary(b, EQ, f.seq(1), f.type(), f.seq().get_allocator());
  }
  else if(f.seq(1).is(F::B)) {
    auto b = f.seq(1);
    b.b() = !b.b();
    return F::make_binary(f.seq(0), EQ, b, f.type(), f.seq().get_allocator());
  }
  return F::make_nary(NEQ, f.seq(), f.type());
}

template <class F>
CUDA NI thrust::optional<F> negate(const F& f) {
  if(f.is_true()) {
    return F::make_false();
  }
  else if(f.is_false()) {
    return F::make_true();
  }
  else if(f.is(F::Seq)) {
    Sig neg_sig;
    switch(f.sig()) {
      case NOT: return f.seq(0);
      case EQ: return negate_eq(f);
      // Total order predicates can be reversed.
      case LEQ: neg_sig = GT; break;
      case GEQ: neg_sig = LT; break;
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

/** True for the operators <=, <, >, >=, =, != */
template <class F>
CUDA NI bool is_arithmetic_comparison(const F& f) {
  if(f.is(F::Seq)) {
    switch(f.sig()) {
      case LEQ:
      case GEQ:
      case LT:
      case GT:
      case EQ:
      case NEQ:
        return true;
      default: break;
    }
  }
  return false;
}

/** True for the operators =, !=, subset, subseteq, supset, supseteq */
template <class F>
CUDA NI bool is_set_comparison(const F& f) {
 if(f.is(F::Seq)) {
    switch(f.sig()) {
      case EQ:
      case NEQ:
      case SUBSET:
      case SUBSETEQ:
      case SUPSET:
      case SUPSETEQ:
        return true;
      default: break;
    }
  }
  return false;
}

/** True for the operators <=, <, >, >=, =, !=, subset, subseteq, supset, supseteq */
template <class F>
CUDA NI bool is_comparison(const F& f) {
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
      default: break;
    }
  }
  return false;
}

/** Returns the converse of a comparison operator (see `is_comparison`). */
CUDA NI inline Sig converse_comparison(Sig sig) {
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
CUDA NI F move_constants_on_rhs(const F& f) {
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

/** Given a formula `f`, we transform all occurrences of `AVar` into logical variables. */
template <class F, class Env>
CUDA NI void map_avar_to_lvar(F& f, const Env& env, bool erase_type = false) {
  switch(f.index()) {
    case F::V:
      f = F::make_lvar(erase_type ? UNTYPED : f.v().aty(), env.name_of(f.v()));
      break;
    case F::Seq:
      for(int i = 0; i < f.seq().size(); ++i) {
        map_avar_to_lvar(f.seq(i), env, erase_type);
      }
      break;
    case F::ESeq:
      for(int i = 0; i < f.eseq().size(); ++i) {
        map_avar_to_lvar(f.eseq(i), env, erase_type);
      }
      break;
    default: break;
  }
}

namespace impl {
  template <class F>
  CUDA bool is_bz(const F& f) {
    return f.is(F::Z) || f.is(F::B);
  }

  template <class F>
  CUDA bool is_binary_bz(const typename F::Sequence& seq) {
    return seq.size() == 2 && is_bz(seq[0]) && is_bz(seq[1]);
  }

  template <class F>
  CUDA bool is_binary_z(const typename F::Sequence& seq) {
    return seq.size() == 2 && seq[0].is(F::Z) && seq[1].is(F::Z);
  }

  /** Evaluate the function or predicate `sig` on the sequence of constants `seq`.
   * It only works on Boolean and integers constants for now.
   */
  template <class F>
  CUDA F eval_seq(Sig sig, const typename F::Sequence& seq, AType atype) {
    switch(sig) {
      case AND: {
        typename F::Sequence residual;
        for(int i = 0; i < seq.size(); ++i) {
          if(seq[i].is_false()) {
            return F::make_false();
          }
          else if(!seq[i].is_true()) {
            residual.push_back(seq[i]);
          }
        }
        if(residual.size() == 0) {
          return F::make_true();
        }
        else if(residual.size() == 1) {
          return residual[0];
        }
        else {
          return F::make_nary(AND, std::move(residual), atype);
        }
      }
      case OR: {
        typename F::Sequence residual;
        for(int i = 0; i < seq.size(); ++i) {
          if(seq[i].is_true()) {
            return F::make_true();
          }
          else if(!seq[i].is_false()) {
            residual.push_back(seq[i]);
          }
        }
        if(residual.size() == 0) {
          return F::make_false();
        }
        else if(residual.size() == 1) {
          return residual[0];
        }
        else {
          return F::make_nary(OR, std::move(residual), atype);
        }
      }
      case IMPLY: {
        if(is_binary_bz<F>(seq)) {
          return seq[0].is_false() || seq[1].is_true() ? F::make_true() : F::make_false();
        }
        else if(seq.size() == 2) {
          if(seq[0].is_true()) { return seq[1]; }
          else if(seq[0].is_false()) { return F::make_true(); }
          else if(seq[1].is_true()) { return F::make_true(); }
          else if(seq[1].is_false()) { return F::make_unary(NOT, seq[0], atype); }
        }
        break;
      }
      case EQUIV: {
        if(is_binary_bz<F>(seq)) {
          return (seq[0].is_true() == seq[1].is_true()) ? F::make_true() : F::make_false();
        }
        else if(seq.size() == 2) {
          if(seq[0].is_true()) { return seq[1]; }
          else if(seq[0].is_false()) { return F::make_unary(NOT, seq[1], atype); }
          else if(seq[1].is_true()) { return seq[0]; }
          else if(seq[1].is_false()) { return F::make_unary(NOT, seq[0], atype); }
        }
        break;
      }
      case NOT: {
        if(seq.size() == 1 && is_bz(seq[0])) {
          return seq[0].is_true() ? F::make_false() : F::make_true();
        }
        break;
      }
      case XOR: {
        if(is_binary_bz<F>(seq)) {
          return (seq[0].is_true() != seq[1].is_true()) ? F::make_true() : F::make_false();
        }
        break;
      }
      case ITE: {
        if(seq.size() == 3 && is_bz(seq[0])) {
          return seq[0].is_true() ? seq[1] : seq[2];
        }
        break;
      }
      case EQ: {
        if(seq.size() == 2 && seq[0].is_constant() && seq[1].is_constant()) {
          return seq[0] == seq[1] ? F::make_true() : F::make_false();
        }
        break;
      }
      case NEQ: {
        if(seq.size() == 2 && seq[0].is_constant() && seq[1].is_constant()) {
          return seq[0] != seq[1] ? F::make_true() : F::make_false();
        }
        break;
      }
      case LT: {
        if(is_binary_z<F>(seq)) {
          return seq[0].z() < seq[1].z() ? F::make_true() : F::make_false();
        }
        break;
      }
      case LEQ: {
        if(is_binary_z<F>(seq)) {
          return seq[0].z() <= seq[1].z() ? F::make_true() : F::make_false();
        }
        break;
      }
      case GT: {
        if(is_binary_z<F>(seq)) {
          return seq[0].z() > seq[1].z() ? F::make_true() : F::make_false();
        }
        break;
      }
      case GEQ: {
        if(is_binary_z<F>(seq)) {
          return seq[0].z() >= seq[1].z() ? F::make_true() : F::make_false();
        }
        break;
      }
      case MIN: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::min(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case MAX: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::max(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case NEG: {
        if(seq.size() == 1 && is_bz(seq[0])) {
          return F::make_z(-seq[0].to_z());
        }
        break;
      }
      case ABS: {
        if(seq.size() == 1 && is_bz(seq[0])) {
          return F::make_z(abs(seq[0].to_z()));
        }
        break;
      }
      case ADD: {
        typename F::Sequence residual;
        logic_int sum = 0;
        for(int i = 0; i < seq.size(); ++i) {
          if(is_bz(seq[i])) {
            sum += seq[i].to_z();
          }
          else {
            residual.push_back(seq[i]);
          }
        }
        if(residual.size() == 0) {
          return F::make_z(sum);
        }
        else {
          if(sum != 0) {
            residual.push_back(F::make_z(sum));
          }
          if(residual.size() == 1) {
            return residual[0];
          }
          else {
            return F::make_nary(ADD, std::move(residual), atype);
          }
        }
      }
      case MUL: {
        typename F::Sequence residual;
        logic_int prod = 1;
        for(int i = 0; i < seq.size(); ++i) {
          if(is_bz(seq[i])) {
            prod *= seq[i].to_z();
          }
          else {
            residual.push_back(seq[i]);
          }
        }
        if(residual.size() == 0) {
          return F::make_z(prod);
        }
        else {
          if(prod == 0) {
            return F::make_z(0);
          }
          else if(prod == -1 && residual.size() > 0) {
            residual[0] = F::make_unary(NEG, std::move(residual[0]), atype);
          }
          else if(prod != 1) {
            residual.push_back(F::make_z(prod));
          }
          if(residual.size() == 1) {
            return residual[0];
          }
          else {
            return F::make_nary(MUL, std::move(residual), atype);
          }
        }
      }
      case SUB: {
        if(is_binary_bz<F>(seq)) {
          return F::make_z(seq[0].to_z() - seq[1].to_z());
        }
        else if(seq.size() == 2 && is_bz(seq[1]) && seq[1].to_z() == 0) {
          return seq[0];
        }
        break;
      }
      case TDIV: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::tdiv(seq[0].z(), seq[1].z()));
        }
        else if(seq.size() == 2 && is_bz(seq[0]) && seq[0].to_z() == 0) {
          return F::make_z(0);
        }
        break;
      }
      case FDIV: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::fdiv(seq[0].z(), seq[1].z()));
        }
        else if(seq.size() == 2 && is_bz(seq[0]) && seq[0].to_z() == 0) {
          return F::make_z(0);
        }
        break;
      }
      case CDIV: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::cdiv(seq[0].z(), seq[1].z()));
        }
        else if(seq.size() == 2 && is_bz(seq[0]) && seq[0].to_z() == 0) {
          return F::make_z(0);
        }
        break;
      }
      case EDIV: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::ediv(seq[0].z(), seq[1].z()));
        }
        else if(seq.size() == 2 && is_bz(seq[0]) && seq[0].to_z() == 0) {
          return F::make_z(0);
        }
        break;
      }
      case TMOD: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::tmod(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case FMOD: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::fmod(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case CMOD: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::cmod(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case EMOD: {
        if(is_binary_z<F>(seq)) {
          return F::make_z(battery::emod(seq[0].z(), seq[1].z()));
        }
        break;
      }
      case POW: {
        if(is_binary_bz<F>(seq)) {
          return F::make_z(battery::ipow(seq[0].to_z(), seq[1].to_z()));
        }
        else if(seq.size() == 2 && is_bz(seq[1]) && seq[1].to_z() == 0) {
          return F::make_z(1);
        }
        break;
      }
      case IN: {
        if(seq.size() == 2 && is_bz(seq[0]) && seq[1].is(F::S)) {
          const auto& set = seq[1].s();
          logic_int left = seq[0].to_z();
          for(int i = 0; i < set.size(); ++i) {
            const auto& lb = battery::get<0>(set[i]);
            const auto& ub = battery::get<1>(set[i]);
            if(is_bz(lb) && is_bz(ub) && left >= lb.to_z() && left <= ub.to_z()) {
              return F::make_true();
            }
          }
          return F::make_false();
        }
        else if(seq.size() == 2 && seq[1].is(F::S) && seq[1].s().size() == 0) {
          return F::make_false();
        }
        break;
      }
    }
    return F::make_nary(sig, seq, atype);
  }
}

template <class F>
CUDA NI F eval(const F& f) {
  switch(f.index()) {
    case F::Z: return f;
    case F::R: return f;
    case F::S: return f;
    case F::B: return f;
    case F::V: return f;
    case F::E: return f;
    case F::LV: return f;
    case F::Seq: {
      const auto& seq = f.seq();
      typename F::Sequence evaluated_seq;
      for(int i = 0; i < seq.size(); ++i) {
        evaluated_seq.push_back(eval(seq[i]));
      }
      return impl::eval_seq<F>(f.sig(), evaluated_seq, f.type());
    }
    case F::ESeq: {
      auto eseq = f.eseq();
      typename F::Sequence evaluated_eseq;
      for(int i = 0; i < eseq.size(); ++i) {
        evaluated_eseq.push_back(eval(eseq[i]));
      }
      return F::make_nary(f.esig(), std::move(evaluated_eseq), f.type());
    }
    default: assert(false); return f;
  }
}

}

#endif
