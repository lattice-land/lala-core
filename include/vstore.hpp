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
  using Env = VarEnv<EnvAllocator>;

private:
  using Array = DArray<Universe, DataAllocator>;

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
    if(f.type() == env.ad_uid() && f.is(F::Seq) && f.sig() == AND) {
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

  CUDA const Universe& project(AVar x) const {
    return data[VID(x)];
  }

  CUDA Universe& project(AVar x) {
    return data[VID(x)];
  }

  CUDA this_type& embed(AVar x, const Universe& dom, bool& has_changed) {
    is_at_top |= data[VID(x)].tell(dom, has_changed).is_top();
    return *this;
  }

  CUDA this_type& tell(const TellType& t, bool& has_changed) {
    for(int i = 0; i < t.size(); ++i) {
      embed(make_var(env.ad_uid(), get<0>(t[i])), get<1>(t[i]), has_changed);
    }
    return *this;
  }

  CUDA const Env& environment() const { return env; }

};

} // namespace lala

#endif
