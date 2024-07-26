// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_INTERPRETATION_HPP
#define LALA_CORE_INTERPRETATION_HPP

#include "logic/logic.hpp"
#include <optional>

/**
 * This file provides an extended interface to interpret formulas in abstract domains and abstract universes.
 * It also provides a unified interface for tell and ask interpretation using a template parameter `IKind`.
*/

namespace lala {

/** Interpret `true` in the lattice `L`.
 * \return `true` if `L` preserves the bottom element w.r.t. the concrete domain or if `true` is interpreted by under-approximation (kind == ASK).
 */
template <class L, IKind kind, bool diagnose = false, class F>
CUDA bool ginterpret_true(const F& f, IDiagnostics& diagnostics) {
  assert(f.is_true());
  if constexpr(kind == IKind::ASK || L::preserve_bot) {
    return true;
  }
  else {
    const char* name = L::name;
    RETURN_INTERPRETATION_ERROR("Bottom is not preserved, hence we cannot over-approximate `true` in this abstract domain.");
  }
}

/** This function provides an extended and unified interface to ask and tell interpretation of formula in abstract domains.
 * It provides default interpretation for common formulas such as `true`, `false` and conjunction of formulas whenever `A` satisfies some lattice-theoretic conditions.
 */
template <IKind kind, bool diagnose = false, class A, class F, class Env, class I>
CUDA bool ginterpret_in(const A& a, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  const char* name = A::name;
  if(f.is_true()) {
    return ginterpret_true<A, kind, diagnose>(f, diagnostics);
  }
  else if(f.is_false()) {
    if constexpr(kind == IKind::TELL || A::preserve_top) {
      return a.template interpret<kind, diagnose>(f, env, intermediate, diagnostics); // We don't know how `top` is represented by this abstract domain, so we just forward the interpretation call.
    }
    else {
      RETURN_INTERPRETATION_ERROR("Top is not preserved, hence we cannot under-approximate `true` in this abstract domain.");
    }
  }
  else if(f.is(F::Seq) && f.sig() == AND) {
    if constexpr(kind == IKind::ASK || A::preserve_join) {
      for(int i = 0; i < f.seq().size(); ++i) {
        if(!ginterpret_in<kind, diagnose>(a, f.seq(i), env, intermediate, diagnostics)) {
          return false;
        }
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Join is not preserved, hence we cannot over-approximate conjunctions in this abstract domain.");
    }
  }
  // In the other cases, we cannot provide a default interpretation, so we forward the call to the abstract domain.
  return a.template interpret<kind, diagnose>(f, env, intermediate, diagnostics);
}

/** This function provides an extended and unified interface to ask and tell interpretation of formula in abstract universes.
 * It provides default interpretation for common formulas such as `true`, `false`, conjunction and disjunction of formulas whenever `U` satisfies some lattice-theoretic conditions. */
template <IKind kind, bool diagnose = false, class F, class Env, class U>
CUDA bool ginterpret_in(const F& f, const Env& env, U& value, IDiagnostics& diagnostics) {
  const char* name = U::name;
  if(f.is_true()) {
    return ginterpret_true<U, kind, diagnose>(f, diagnostics);
  }
  else if(f.is_false()) {
    if constexpr(kind == IKind::TELL || U::preserve_top) {
      value.join_top();
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Top is not preserved, hence we cannot under-approximate `true` in this abstract universe.");
    }
  }
  else if(f.is(F::Seq)) {
    if(f.sig() == AND) {
      if constexpr(kind == IKind::ASK || U::preserve_join) {
        for(int i = 0; i < f.seq().size(); ++i) {
          if(!ginterpret_in<kind, diagnose>(f.seq(i), env, value, diagnostics)) {
            return false;
          }
        }
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Join is not preserved, hence we cannot over-approximate conjunctions in this abstract universe.");
      }
    }
    else if(f.sig() == OR) {
      if constexpr(kind == IKind::TELL || U::preserve_meet) {
        using U2 = typename U::local_type;
        U2 meet_value = U2::top();
        for(int i = 0; i < f.seq().size(); ++i) {
          U2 x = U2::bot();
          if(!ginterpret_in<kind, diagnose>(f.seq(i), env, x, diagnostics)) {
            return false;
          }
          meet_value.meet(x);
        }
        value.join(meet_value);
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Meet is not preserved, hence we cannot under-approximate disjunctions in this abstract universe.");
      }
    }
  }
  // In the other cases, we cannot provide a default interpretation, so we forward the call to the abstract element.
  return U::template interpret<kind, diagnose>(f, env, value, diagnostics);
}

/** Top-level version of `ginterpret_in`, we restore `env` and `intermediate` in case of failure. */
template <IKind kind, bool diagnose = false, class A, class F, class Env, class I>
CUDA bool top_level_ginterpret_in(const A& a, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  auto snap = env.snapshot();
  I copy = intermediate;
  if(ginterpret_in<kind, diagnose>(a, f, env, intermediate, diagnostics)) {
    return true;
  }
  else {
    env.restore(snap);
    intermediate = std::move(copy);
    return false;
  }
}

template <class A, class Alloc = battery::standard_allocator, class Env>
CUDA A make_bot(Env& env, Alloc alloc = Alloc{}) {
  if constexpr(A::is_abstract_universe) {
    return A::bot();
  }
  else {
    return A::bot(env, alloc);
  }
}

template <bool diagnose = false, class TellAlloc = battery::standard_allocator, class F, class Env, class L>
CUDA bool interpret_and_tell(const F& f, Env& env, L& value, IDiagnostics& diagnostics, TellAlloc tell_alloc = TellAlloc{}) {
  if constexpr(L::is_abstract_universe) {
    return ginterpret_in<IKind::TELL, diagnose>(f, env, value, diagnostics);
  }
  else {
    typename L::template tell_type<TellAlloc> tell(tell_alloc);
    if(top_level_ginterpret_in<IKind::TELL, diagnose>(value, f, env, tell, diagnostics)) {
      value.refine(tell);
      return true;
    }
    else {
      return false;
    }
  }
}

template <class A, bool diagnose = false, class F, class Env, class TellAlloc = typename A::allocator_type>
CUDA std::optional<A> create_and_interpret_and_tell(const F& f,
 Env& env, IDiagnostics& diagnostics,
 typename A::allocator_type alloc = typename A::allocator_type{},
 TellAlloc tell_alloc = TellAlloc{})
{
  auto snap = env.snapshot();
  A a{make_bot<A>(env, alloc)};
  if(interpret_and_tell<diagnose>(f, env, a, diagnostics, tell_alloc)) {
    return {std::move(a)};
  }
  else {
    env.restore(snap);
    return {};
  }
}


}

#endif
