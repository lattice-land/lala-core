// Copyright 2022 Pierre Talbot

#ifndef UPSET_UNIVERSE_HPP
#define UPSET_UNIVERSE_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include <iostream>
#include "thrust/optional.h"
#include "../logic/logic.hpp"
#include "chain_pre_dual.hpp"
#include "pre_binc.hpp"
#include "pre_finc.hpp"
#include "pre_zinc.hpp"
#include "memory.hpp"

/** A pre-abstract universe is a lattice (with usual operations join, order, ...) equipped with a simple logical interpretation function and a next/prev functions.
    We consider pre-abstract universes with an upset semantics.
    For any lattice \f$ L \f$, we consider an element \f$ a \in L \f$ to represent all the concrete elements equal to or above it.
    This set is called the upset of \f$ a \f$ and is denoted \f$ \mathord{\uparrow}{a} \f$.
    The concretization function \f$ \gamma \f$ formalizes this idea: \f$ \gamma(a) = \{x \mapsto b \;|\; b \in \mathord{\uparrow}{a} \cap U \} \f$ where \f$ U \f$ is the universe of discourse.
    The intersection with \f$ U \f$ is necessary to remove potential elements in the abstract universe that are not in the concrete universe of discourse (e.g., \f$ -\infty, \infty \f$ below).

    The upset semantics associates each element of a lattice to its concrete upset.
    It is possible to decide that each element is associated to the concrete downset instead.
    Doing so will reverse our usage of the lattice-theoretic operations (join instead of meet, <= instead of >=, etc.).
    Instead of considering the upset semantics, it is more convenient to consider the downset semantics of the dual lattice.

    Example:
      * The lattice of increasing integer \f$ \mathit{ZInc} = \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ is ordered by the natural arithmetic comparison operator.
        Using the upset semantics, we can represent simple constraints such as \f$ x \geq 3 \f$, in which case the upset \f$ \mathord{\uparrow}{3} = \{3, 4, \ldots\} \f$ represents all the values of \f$ x \f$ satisfying the constraints \f$ x \geq 3 \f$, that is, the solutions of the constraints.
      * By taking the downset semantics of \f$ \mathit{ZInc} \f$, we can represent constraints such as \f$ x \leq 3 \f$.
      * Alternatively, we can take the dual lattice of decreasing integers \f$ \mathit{ZDec} = \langle \{\infty, \ldots, 2, 1, 0, -1, -2, \ldots, -\infty\}, \geq \rangle \f$.
        The upset semantics of \f$ \mathit{ZDec} \f$ corresponds to the downset semantics of \f$ \mathit{ZInc} \f$.

  From a pre-abstract universe, we obtain an abstract universe using the `Universe` class below.
  We also define various aliases to abstract universes such as `ZInc`, `ZDec`, etc.
*/

namespace lala {

template<class PreUniverse, class Mem>
class UpsetUniverse;

/** Lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<class VT, class Mem>
using ZInc = UpsetUniverse<PreZInc<VT>, Mem>;

/** Lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<class VT, class Mem>
using ZDec = UpsetUniverse<ChainPreDual<PreZInc<VT>>, Mem>;

/** Lattice of increasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \leq y\} \f$. */
template<class VT, class Mem>
using FInc = UpsetUniverse<PreFInc<VT>, Mem>;

/** Lattice of decreasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \geq y\} \f$. */
template<class VT, class Mem>
using FDec = UpsetUniverse<ChainPreDual<PreFInc<VT>>, Mem>;

/** Lattice of increasing Boolean where \f$ \mathit{false} \leq \mathit{true} \f$. */
template<class Mem>
using BInc = UpsetUniverse<PreBInc, Mem>;

/** Lattice of decreasing Boolean where \f$ \mathit{true} \leq \mathit{false} \f$. */
template<class Mem>
using BDec = UpsetUniverse<ChainPreDual<PreBInc>, Mem>;

/** Aliases for lattice allocated on the stack (as local variable) and accessed by only one thread.
 * To make things simpler, the underlying type is also chosen (when required). */
namespace local {
  using ZInc = ::lala::ZInc<int, battery::LocalMemory>;
  using ZDec = ::lala::ZDec<int, battery::LocalMemory>;
  using FInc = ::lala::FInc<double, battery::LocalMemory>;
  using FDec = ::lala::FDec<double, battery::LocalMemory>;
  using BInc = ::lala::BInc<battery::LocalMemory>;
  using BDec = ::lala::BDec<battery::LocalMemory>;
}

namespace impl {
  template<class T>
  struct is_upset_universe {
    static constexpr bool value = false;
  };

  template<class PreUniverse, class Mem>
  struct is_upset_universe<UpsetUniverse<PreUniverse, Mem>> {
    static constexpr bool value = true;
  };

  template <class T>
  inline constexpr bool is_upset_universe_v = is_upset_universe<T>::value;

}

/** This function is useful when we need to convert a value to its dual.
    The dual is the downset of the current element, therefore, if we have \f$ x >= 10 \f$, the dual is given by the formula \f$ x <= 10 \f$ interpreted in the dual lattice.
    In that case, it just changes the type of the lattice without changing the value.
    A difference occurs on the bottom and top element.
    Indeed, by our representation of bot and top, the bottom value in a lattice L equals the top value in its dual, but we need them to remain the same, so the dual of `L::bot()` is `LDual::bot()`.*/
template <class LDual, class L>
CUDA LDual dual(L x) {
  if(x.is_bot()) return LDual::bot();
  if(x.is_top()) return LDual::top();
  return LDual(x.value());
}

template<class PreUniverse, class Mem>
class UpsetUniverse
{
  using U = PreUniverse;
public:
  using pre_universe = PreUniverse;
  using value_type = typename pre_universe::value_type;
  using memory_type = Mem;
  using this_type = UpsetUniverse<pre_universe, memory_type>;
  using reverse_type = UpsetUniverse<typename pre_universe::reverse_type, memory_type>;

  template<class M>
  using this_type2 = UpsetUniverse<pre_universe, M>;

  template<class F>
  using iresult = IResult<this_type, F>;

  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = pre_universe::is_totally_ordered;
  constexpr static const bool preserve_bot = pre_universe::preserve_bot;
  constexpr static const bool preserve_top = pre_universe::preserve_top;
  constexpr static const bool injective_concretization = pre_universe::injective_concretization;
  constexpr static const bool preserve_inner_covers = pre_universe::preserve_inner_covers;
  constexpr static const bool complemented = pre_universe::complemented;
  constexpr static const bool increasing = pre_universe::increasing;
  constexpr static const char* name = pre_universe::name;
  inline static const value_type zero = this_type(pre_universe::zero);
  inline static const value_type one = this_type(pre_universe::one);

private:
  using atomic_type = typename memory_type::atomic_type<value_type>;
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$ if `preserve_bot` is true. */
  CUDA static this_type bot() {
    return this_type();
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$ if `preserve_top` is true. */
  CUDA static this_type top() {
    return this_type(U::top());
  }
  /** Initialize an upset universe to bottom. */
  CUDA UpsetUniverse(): val(U::bot()) {}
  /** Similar to \f$[\![x \geq_A i]\!]\f$ for any name `x` where \f$ \geq_A \f$ is the lattice order. */
  CUDA UpsetUniverse(value_type x): val(x) {}
  CUDA UpsetUniverse(const this_type& other): UpsetUniverse(other.value()) {}
  CUDA UpsetUniverse(this_type&& other): val(std::move(other.val)) {}

  template <class M>
  CUDA UpsetUniverse(const this_type2<M>& other): UpsetUniverse(other.value()) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class M>
  CUDA std::enable_if_t<sequential, this_type&> operator=(const this_type2<M>& other) {
    memory_type::store(val, other.value());
    return *this;
  }

  CUDA std::enable_if_t<sequential, this_type&> operator=(const this_type& other) {
    memory_type::store(val, other.value());
    return *this;
  }

  CUDA value_type value() const { return memory_type::load(val); }

  CUDA operator value_type() const { return value(); }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return value() == U::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA local::BDec is_bot() const {
    return value() == U::bot();
  }

  CUDA this_type& tell_top() {
    memory_type::store(val, U::top());
    return *this;
  }

  template<class M1, class M2>
  CUDA this_type& tell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    value_type r1 = value();
    value_type r2 = is_totally_ordered ? other.value() : U::join(r1, other.value());
    if(U::strict_order(r1, r2)) {
      memory_type::store(val, r2);
      has_changed.tell_top();
    }
    return *this;
  }

  template<class M1>
  CUDA this_type& tell(const this_type2<M1>& other) {
    value_type r1 = value();
    value_type r2 = is_totally_ordered ? other.value() : U::join(r1, other.value());
    if(U::strict_order(r1, r2)) {
      memory_type::store(val, r2);
    }
    return *this;
  }

  CUDA this_type& dtell_bot() {
    memory_type::store(val, U::bot());
    return *this;
  }

  template<class M1, class M2>
  CUDA this_type& dtell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    value_type r1 = value();
    value_type r2 = is_totally_ordered ? other.value() : U::meet(r1, other.value());
    if(U::strict_order(r2, r1)) {
      memory_type::store(val, r2);
      has_changed.tell_top();
    }
    return *this;
  }

  template<class M1>
  CUDA this_type& dtell(const this_type2<M1>& other) {
    value_type r1 = value();
    value_type r2 = is_totally_ordered ? other.value() : U::meet(r1, other.value());
    if(U::strict_order(r2, r1)) {
      memory_type::store(val, r2);
    }
    return *this;
  }

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the current value.
  If `U` preserves bottom `true` is returned whenever \f$ a = \bot \f$, if it preserves top `false` is returned whenever \f$ a = \top \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function. */
  template<class Env>
  CUDA TFormula<typename Env::allocator_type> deinterpret(AVar avar, const Env& env) const {
    using allocator_t = typename Env::allocator_type;
    if(preserve_top && is_top()) {
      return TFormula<allocator_t>::make_false();
    }
    else if(preserve_bot && is_bot()) {
      return TFormula<allocator_t>::make_true();
    }
    return make_v_op_z(avar, U::sig_order(), value(), avar.aty(), EXACT, env.get_allocator());
  }

  /** Under-approximates the current element \f$ a \f$ w.r.t. \f$ \rrbracket a \llbracket \f$ into `ua`.
   * For this abstract universe, it always returns `true` since the current element \f$ a \f$ is an exact representation of \f$ \rrbracket a \llbracket \f$. */
  CUDA bool extract(this_type2<battery::LocalMemory>& ua) const {
    ua.val = value();
    return true;
  }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot()) {
      printf("\u22A5");
    }
    else if(is_top()) {
      printf("\u22A4");
    }
    else {
      ::battery::print(value());
    }
  }

private:

  /** Interpret a formula of the form `k <sig> x`. */
  template<class F, class Env>
  CUDA static iresult<F> interpret_k_op_x(const F& f, const F& k, Sig sig, const F& x, const Env& env) {
    auto r = pre_universe::interpret(k, k.sort().value(), f.approx());
    if(!r.has_value()) {
      return r;
    }
    else if(sig == U::sig_order()) {  // e.g., x <= 4 or x >= 4.24
      return r;
    }
    else if(sig == U::sig_strict_order()) {  // e.g., x < 4
      if(f.is_under() ||
         (preserve_inner_covers && pre_universe::has_unique_next(r.value())))
      {
        return std::move(r).map(pre_universe::next(r.value()));
      }
      else if(f.is_exact()) {
        auto err = IError<F>(true, name, "Exactly interpreting a strict relation, i.e. `x < k`, not supported.", f);
        if constexpr(!preserve_inner_covers) {
          err.add_suberror(IError<F>(true, name, "Inner covers are not preserved: there might be elements between k and next(k).", f));
        }
        if(!pre_universe::has_unique_next(r.value())) {
          err.add_suberror(IError<F>(true, name, "The cover is not unique: there are several incomparable elements satisfying next(k).", f));
        }
        return iresult<F>(std::move(err));
      }
      // In case of over-approximation, interpreting using `U::sig_order` is a correct option.
      return r;
    }
    // Under-approximation of `x != k` as `next(k)`.
    else if(f.is_under() && sig == NEQ) {
      return std::move(r).map(pre_universe::next(r.value()));
    }
    // Over-approximation of `x == k` as `k`.
    else if(f.is_over() && sig == EQ) {
      return r;
    }
    else {
      return iresult<F>(IError<F>(true, name, "The signature of the symbol `" + LVar<typename F::allocator_type>(string_of_sig(sig)) + "` is not supported: either the symbol is unknown, approximation kind is not supported or the type of the arguments of the symbols are not supported.", f));
    }
  }

  template <class F>
  CUDA static iresult<F> interpret_false(const F& f) {
    if(preserve_top || f.is_over()) {
      return iresult<F>(top());
    }
    else {
      return iresult<F>(IError<F>(true, name, "Top is not preserved, hence it cannot exactly interpret or under-approximate formulas equivalent to `false`.", f));
    }
  }

  template<class F>
  CUDA static iresult<F> interpret_true(const F& f) {
    if(preserve_bot || f.is_under()) {
      return bot();
    }
    else {
      return iresult<F>(IError<F>(true, name, "Bottom is not preserved, hence it cannot exactly interpret or over-approximate formula equivalent to `true`.", f));
    }
  }

public:

  /** Expects a predicate of the form `x <op> k` or `k <op> x`, where `x` is any variable's name, and `k` a constant.
    - If `f.approx()` is EXACT: `op` can be `U::sig_order()` or `U::sig_strict_order()`.
    - If `f.approx()` is UNDER: `op` can be, in addition to exact, `!=`.
    - If `f.approx()` is OVER: `op` can be, in addition to exact, `==` and `in`.
    Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom).
    - The interpretation depends on the abstract pre-universe.
    */
  template<class F, class Env>
  CUDA static iresult<F> interpret(const F& f, const Env& env) {
    if(f.is_true()) {
      return interpret_true(f);
    }
    else if(f.is_false()) {
      return interpret_false(f);
    }
    else if(f.is(F::E)) {
      return pre_universe::interpret_type(f);
    }
    else {
      if(f.is_binary()) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable != 1) {
          return iresult<F>(IError<F>(true, name, "Only binary formulas of the form `t1 <sig> t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.", f));
        }
        const auto& k = f.seq(idx_constant);
        const auto& x = f.seq(idx_variable);
        if(f.sig() == IN) {
          if(idx_constant == 0) { // `k in x` is equivalent to `{k} \subseteq x`.
            return interpret_k_op_x(f, F::make_set(logic_set<F, typename F::allocator_type>(
              {battery::make_tuple<F,F>(F(k),F(k))})), SUBSETEQ, x, env);
          }
          else { // `x in k` is equivalent to `x >= meet k` where `>=` is the lattice order `U::sig_order()`.
            const auto& set = k.s();
            if(set.size() == 0) {
              return interpret_false(f);
            }
            if(f.is_over()) {
              value_type meet_s = pre_universe::top();
              constexpr int bound_index = increasing ? 0 : 1;
              for(int i = 0; i < set.size(); ++i) {
                auto bound = battery::get<bound_index>(set[i]);
                auto sort = bound.sort();
                if(!sort.has_value()) {
                  return iresult<F>(IError<F>(true, name, "Could not compute the sort of the bound in a set constant.", k));
                }
                auto res = pre_universe::interpret(bound, *sort, bound.approx());
                if(!res.has_value()) {
                  return std::move(iresult<F>(IError<F>(true, name, "Could not interpret an element of a set in the interval's bound.", f))
                    .join_errors(std::move(res)));
                }
                meet_s = pre_universe::meet(meet_s, res.value());
              }
              return iresult<F>(this_type(meet_s));
            }
            else {
              return iresult<F>(IError<F>(true, name, "Exact or under-approximation of 'set' constant is not yet supported.", f));
            }
          }
        }
        else {
          Sig sig = idx_constant == 1 ? converse_comparison(f.sig()) : f.sig();
          return interpret_k_op_x(f, k, sig, x, env);
        }
      }
      else {
        return iresult<F>(IError<F>(true, name, "Only binary constraints are supported.", f));
      }
    }
  }

  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    if(sig == ABS && !increasing) { return false; } // The absolute function is linked to increasing abstract universe, and cannot be represented in a decreasing abstract domain (unless by mapping to `top` but that's uninteresting).
    return pre_universe::is_supported_fun(appx, sig);
  }

private:
  template<Approx appx, Sig sig, class A, class B>
  CUDA static constexpr this_type2<battery::LocalMemory> div_per_zero(const A& a, const B&) {
    if constexpr(A::increasing && B::increasing == increasing) {
      if(a >= A::zero) return zero;
    }
    else if(!A::increasing && B::increasing != increasing) {
      if(a <= A::zero) return zero;
    }
    return this_type2<battery::LocalMemory>::bot();
  }

public:
  /** Unary function over `value_type`.
   * Due to some complications, the function acts as if `a` was of the lattice type "Flat Lattice" instead of an upset.
   * To obtain the upset semantics of the functions, you can rely on Interval, where you set one of the bounds to `bot`.  */
  template<Approx appx, Sig sig, class A>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(const A& a) {
    if constexpr(A::preserve_top) {
      if(a.is_top()) {
        return this_type2<battery::LocalMemory>::top();
      }
    }
    if constexpr(sig == ABS) {
      if(a < zero) {
        return zero;
      }
      else {
        return a;
      }
    }
    else if constexpr(A::preserve_bot) {
      if(a.is_bot()) {
        return this_type2<battery::LocalMemory>::bot();
      }
    }
    return pre_universe::template fun<appx, sig>(a);
  }

  /** Binary functions over `value_type`.
   * Due to some complications, the function acts as if `a` and `b` were of the lattice type "Flat Lattice" instead of an upset.
   * To obtain the upset semantics of these functions, you can rely on Interval, where you set one of the bounds of each `a` and `b` to `bot`. */
  template<Approx appx, Sig sig, class A, class B>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(const A& a, const B& b) {
    if constexpr(A::preserve_top) {
      if(a.is_top()) {
        return this_type2<battery::LocalMemory>::top();
      }
    }
    if constexpr(B::preserve_top) {
      if(b.is_top()) {
        return this_type2<battery::LocalMemory>::top();
      }
    }
    if constexpr(A::preserve_bot) {
      if(a.is_bot()) {
        return this_type2<battery::LocalMemory>::bot();
      }
    }
    if constexpr(B::preserve_bot) {
      if(b.is_bot()) {
        return this_type2<battery::LocalMemory>::bot();
      }
    }
    if constexpr(is_division(sig) && impl::is_upset_universe_v<A> && impl::is_upset_universe_v<B>) {
      if(b == B::zero) {
        if(B::preserve_inner_covers && B::pre_universe::has_unique_next(b)) {
          return pre_universe::template fun<appx, sig>(a, B::pre_universe::next(b));
        }
        else {
          return div_per_zero<appx, sig>(a, b);
        }
      }
    }
    return pre_universe::template fun<appx, sig>(a, b);
  }

  template<class Pre2, class Mem2>
  friend class UpsetUniverse;
};

// Lattice operators

template<class Pre, class M1, class M2>
CUDA UpsetUniverse<Pre, battery::LocalMemory> join(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::join(a, b);
}

template<class Pre, class M1, class M2>
CUDA UpsetUniverse<Pre, battery::LocalMemory> meet(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::meet(a, b);
}

template<class Pre, class M1, class M2>
CUDA bool operator<=(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::order(a, b);
}

template<class Pre, class M1, class M2>
CUDA bool operator<(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::strict_order(a, b);
}

template<class Pre, class M1, class M2>
CUDA bool operator>=(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::order(b, a);
}

template<class Pre, class M1, class M2>
CUDA bool operator>(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return Pre::strict_order(b, a);
}

template<class Pre, class M1, class M2>
CUDA bool operator==(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return a.value() == b.value();
}

template<class Pre, class M1, class M2>
CUDA bool operator!=(const UpsetUniverse<Pre, M1>& a, const UpsetUniverse<Pre, M2>& b) {
  return a.value() != b.value();
}

template<class Pre, class M>
std::ostream& operator<<(std::ostream &s, const UpsetUniverse<Pre, M> &upset) {
  if(upset.is_bot()) {
    s << "\u22A5";
  }
  else if(upset.is_top()) {
    s << "\u22A4";
  }
  else {
    s << upset.value();
  }
  return s;
}

} // namespace lala

#endif
