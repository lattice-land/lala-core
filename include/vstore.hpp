// Copyright 2021 Pierre Talbot

#ifndef VSTORE_HPP
#define VSTORE_HPP

#include "ast.hpp"

namespace lala {

/** The variable store abstract domain is a _domain transformer_ built on top of an abstract universe `U`.
Concretization function: \f$ \gamma(\rho) \sqcap_{x \in \pi(\rho)} \gamma_{U_x}(\rho(x)) \f$.
The top element is smashed and the equality between two stores is represented by the following equivalence relation, for two stores \f$ S \f$ and \f$ T \f$:
\f$ S \equiv T \Leftrightarrow \forall{x \in \mathit{Vars}},~S(x) = T(x) \lor \exists{x \in \mathit{dom}(S)},\exists{y \in \mathit{dom}(T)},~S(x) = \top \land T(x) = \top \f$.
Intuitively, it means that either all elements are equal or both stores have a top element, in which case they "collapse" to the top element, and are considered equal.

  `DataAlloc` is the allocator of the underlying array of universes.
  `EnvAlloc` is the allocator of the mapping between names (LVar) and indices in the store (AVar).
On some architecture, such as GPU, it is better to store the data in the shared memory whenever possible, while the environment can stay in the global memory because it is usually not queried during solving (but before or after). */
template<class U, class Alloc>
class VStore {
public:
  using Universe = U;
  using Allocator = Alloc;
  using EnvAllocator = Allocator;
  using DataAllocator = typename FasterAllocator<Allocator>::type;
  using this_type = VStore<U, Allocator>;

  using TellType = DArray<battery::tuple<int, Universe>, EnvAllocator>;

private:
  using Array = DArray<Universe, DataAllocator>;
  using Env = VarEnv<EnvAllocator>;
  Array data;
  Env env;
  bool is_at_top;

public:
  /** Initialize an empty store equivalent to \f$ \bot \f$ with memory reserved for `n` variables. */
  CUDA VStore(AType uid, size_t n,
    DataAllocator alloc = DataAllocator()): data(n, Universe::bot(), alloc), env(uid, n), is_at_top(false) {}
  CUDA VStore(const VStore& other): data(other.data), env(other.env), is_at_top(other.is_at_top) {}
  CUDA VStore(VStore&& other): data(std::move(other.data)), env(std::move(other.env)), is_at_top(other.is_at_top) {}

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA int vars() const {
    return env.size();
  }

  CUDA static this_type bot(AType uid = UNTYPED) {
    return VStore(uid, 0);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType uid = UNTYPED) {
    VStore s(uid, 0);
    s.is_at_top = true;
    return std::move(s);
  }

  /** `true` if at least one element is equal to top in the store, `false` otherwise. */
  CUDA bool is_top() const {
    return is_at_top;
  }

  /** Bottom is represented by the empty variable store. */
  CUDA bool is_bot() const {
    return !is_at_top && vars() == 0;
  }

  template<typename U2, typename Alloc2>
  CUDA bool operator==(const VStore<U2, Alloc2>& other) const {
    if(is_top()) {
      return other.is_top();
    }
    if(vars() != other.vars()) {
      return false;
    }
    for(int i = 0; i < vars(); ++i) {
      if(data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  template<typename U2, typename Alloc2>
  CUDA bool operator!=(const VStore<U2, Alloc2>& other) const {
    return !(*this == other);
  }

private:
  CUDA void resize(int newsize) {
    Array data2(newsize, Universe::bot(), data.get_allocator());
    for(int i = 0; i < vars(); ++i) {
      data2[i] = data[i];
    }
    data = std::move(data2);
    env.reserve(newsize);
  }

  // Redeclaration of variable is not supported.
  // That does not depend on the approximation because ignoring it might lead to some subformulas to rely on the preexisting variable instead of the one we just ignored, which might eventually lead to unsound results.
  template <typename F>
  CUDA bool check_declaration_errors(const typename F::Sequence& seq, int i) const {
    auto var = var_in(seq[i], env);
    if(seq[i].is(F::E)) {
      // check for redeclaration error either in the environment or in the sequence [0..i-1].
      if(var.has_value()) { return true; }
      for(int j = 0; j < i; ++j) {
        if(seq[j].is(F::E)) {
          if(get<0>(seq[i].exists()) == get<0>(seq[j].exists())) {
            return true;
          }
        }
      }
    }
    else {
      // check that the variable used by the formula `seq[i]` has been declared in env or in the sequence [0..i-1].
      if(!var.has_value()) {
        const auto& fv = var_in(seq[i]);
        if(fv.is(F::LV)) {
          for(int j = 0; j < i; ++j) {
            if(seq[j].is(F::E)) {
              if(fv.lv() == get<0>(seq[j].exists())) {
                return false;
              }
            }
          }
          return true;
        }
      }
    }
    return false;
  }

public:
  /** The store of variables lattice expects a conjunctive formula \f$ c_1 \land \ldots \land c_n \f$ in which all components \f$ c_i \f$ are formulas with a single variable that can be handled by the abstract universe `U`, or existential quantifiers.
    Optionally, `declaration_errors` can be set to `false` to skip the check of redeclared and undeclared variables (which is done in \f$ \O(n^2)\f$ with \f$ n \f$ the number of variables).

    I. Approximation
    ================

    Exact or under-approximation of \f$ \land \f$ is treated in the same way (nothing special is performed for under-approximation).
    Over-approximation of \f$ \land \f$ allows the abstract domain to ignore components of the formula that cannot be interpreted in the underlying abstract universe.

    II. Existential quantifier
    ==========================
    Variables must be existentially quantified before a formula containing such a variable can be interpreted.
    Variables are immediately added to `VStore` and initialized to \f$ \bot_U \f$, hence existential quantifiers never need to be joined in this abstract domain.
    Shadowing/redeclaration of variables with existential quantifier is not supported.
    Variables are added to the current abstract element only if `interpret(f) != {}`.

    III. Technical note on allocators
    =================================

    PoolAllocator (for shared memory) does not support deallocation of memory, therefore you should call interpret once with all existential quantifiers first to avoid reallocating this array (and wasting shared memory space).
  */
  template <typename F>
  CUDA thrust::optional<TellType> interpret(const F& f, bool declaration_errors = true) {
    if(f.is(F::Seq) && f.sig() == AND) {
      const typename F::Sequence& seq = f.seq();
      // 1. We collect all existential quantifiers to resize `data` and `env` if needed.
      //    We also check that all subformulas are interpretable in `U`.
      int newvars = 0;
      for(int i = 0; i < seq.size(); ++i) {
        if(U::interpret(seq[i]).has_value()) {
          if(seq[i].is(F::E)) {
            newvars++;
          }
          if(declaration_errors && check_declaration_errors<F>(seq, i)) {
            return {};
          }
        }
        else if(f.approx() != OVER) {
          // If a formula is not interpretable in `U`, and that we do not allow over-approximation, we cannot interpret the formula.
          return {};
        }
      }
      if(vars() + newvars >= env.capacity()) {
        resize(vars() + newvars);
      }
      // 2. We extend the environment with new bindings.
      for(int i = 0; i < seq.size(); ++i) {
        if(seq[i].is(F::E)) {
          const auto& var_name = get<0>(seq[i].exists());
          env.add(var_name);
        }
      }
      // 3. We convert each subformula to its corresponding universe element.
      if(seq.size() > newvars) {
        DArray<U, EnvAllocator> tell_data(vars(), Universe::bot());
        for(int i = 0; i < seq.size(); ++i) {
          if(!seq[i].is(F::E)) {
            auto u = Universe::interpret(seq[i]);
            if(u.has_value()) {
              thrust::optional<AVar> v = var_in(seq[i], env);
              tell_data[VID(*v)].join(*u);
            }
          }
        }
        // 4. We compress the resulting `tell_data`.
        int tell_elements = 0;
        for(int i = 0; i < tell_data.size(); ++i) {
          if(tell_data[i].is_top()) {
            return TellType(1, make_tuple(i, tell_data[i]));
          }
          if(!tell_data[i].is_bot()) {
            ++tell_elements;
          }
        }
        TellType res(tell_elements, make_tuple(0,tell_data[0]));
        for(int i = 0, j = 0; i < tell_data.size(); ++i) {
          if(!tell_data[i].is_bot()) {
            res[j] = make_tuple(i, tell_data[i]);
            j++;
          }
        }
        return res;
      }
      else {
        return TellType();
      }
    }
    else {
      auto u = U::interpret(f);
      if(u.has_value()) {
        if(f.is(F::E)) {
          if(var_in(f, env).has_value()) {
            return {}; // redeclaration
          }
          else {
            resize(vars() + 1);
            const auto& var_name = get<0>(f.exists());
            env.add(var_name);
            return TellType();
          }
        }
        else {
          thrust::optional<AVar> v = var_in(f, env);
          if(v.has_value()) {
            if(u->is_bot()) {
              return TellType();
            }
            else {
              return TellType(1, make_tuple(VID(*v), *u));
            }
          }
        }
      }
    }
    return {};
  }

  // /** See `AbstractUniverse.interpret` and `TellElement`.
  // \return An empty optional if the formula cannot be interpreted in the abstract domain, or if \f$\bot\f$ would be trivially returned in case of over-approximation (dually for \f$ \top \f$ and under-approximation).
  // Otherwise, it returns the interpreted formula.
  // The returned tell element must be joined later in the current abstract element `this` and not in another abstract element. */
  // CUDA virtual thrust::optional<TellElement> interpret(Approx appx, const Formula& f) = 0;

  // /** Similar to `interpret` but for the ask queries.
  // A reasonable default implementation is `return interpret(UNDER, f)`, with `AskElement = TellElement`.
  // If `f` is under-approximated, then \f$ entailment(f) \f$ will hold only if the solutions of `this` are included in the solution of `f`.
  // See also `AskElement`. */
  // CUDA virtual thrust::optional<AskElement> interpret_ask(const Formula& f) = 0;

  // /** Compute \f$ a \sqcup [\![\varphi]\!] \f$ where \f$a\f$ (`this`) is the current element and \f$ [\![\varphi]\!] \f$ (`other`) an interpreted formula. */
  // CUDA virtual this_type& join(const TellElement& other) = 0;

  // /** Compute \f$ a \sqcap [\![\varphi]\!] \f$, see also `join`. */
  // CUDA virtual this_type& meet(const TellElement& other) = 0;

  // /** `refine` is an extensive function (\f$\forall{a \in A},~\mathit{refine}(a) \geq a \f$) refining an abstract element \f$a\f$.
  // It can have additional properties such as being under- or over-approximating depending on the abstract domain.
  // \return `true` if the abstract element has changed and `false` if we reached a fixed point. */
  // CUDA virtual bool refine() = 0;

  // * The entailment, formally written \f$a \models \varphi\f$, is `true` whenever we can deduce a formula \f$\varphi\f$ from an abstract element \f$a\f$, i.e., \f$\gamma(a) \subseteq [\![\varphi]\!]\f$.
  // Note that if it returns `false`, it can either mean \f$\lnot\varphi\f$ is entailed, or that we do not know yet if it is entailed or not.
  // Therefore, to test for _disentailment_, you should ask if the negation of the formula is entailed.
  // CUDA virtual bool entailment(const AskElement& element) const = 0;

  // /** The projection of term onto the underlying abstract universe `Universe`.
  // A common example is to project the domain of a variable `x` or a term such as `x + y` onto an interval or set variable domain.
  // If you want to project a formula onto a Boolean, you should use `entailment` instead. */
  // CUDA virtual Universe project(const TellElement& x) const = 0;

  // /** The function `embed(x, dom)` is similar to \f$ a \sqcup [\![\varphi]\!] \f$ where \f$\varphi\f$ is a formula with a single variable equals to \f$ x \f$ and interpretable in `Universe`.
  //  Here, the underlying element `Universe` has already been created. */
  // CUDA virtual void embed(AVar x, const Universe& dom) = 0;

  // /** See `AbstractUniverse.split`. */
  // CUDA virtual DArray<TellElement, Allocator> split(/*const SearchStrategy& strat*/) const = 0;

  // /** See `AbstractUniverse.reset`. */
  // CUDA virtual void reset(const this_type& b) = 0;

  // /** See `AbstractUniverse.clone`. */
  // CUDA virtual this_type* clone() const = 0;

  // /** This function is the inverse of `interpret`, but directly maps to a general `Formula`.
  //     Let \f$ a = [\![\varphi]\!]_A \f$, then we must have \f$ \gamma(a) = [\![[\![a]\!]^{-1}]\!]^\flat \f$. */
  // CUDA virtual Formula deinterpret() const = 0;

  // /** This function is similar to `deinterpret` but for a specific tell element, that is not necessarily in the abstract element yet. */
  // CUDA virtual Formula deinterpret_tell(const TellElement& element) const = 0;

  // /** This function is similar to `deinterpret` but for a specific ask element. */
  // CUDA virtual Formula deinterpret_ask(const AskElement& element) const = 0;

  // /** Print the current element with the logical name of the variables. */
  // CUDA virtual void print() const = 0;
};

} // namespace lala

#endif
