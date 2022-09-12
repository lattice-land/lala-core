// Copyright 2021 Pierre Talbot

#ifndef ENV_HPP
#define ENV_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "logic/ast.hpp"

namespace lala {

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables.
This class is supposed to be used inside an abstract domain, to help with the conversion. */
template<class Allocator>
class VarEnv {
public:
  using allocator_type = Allocator;
  using LName = LVar<allocator_type>;
  using this_type = VarEnv<Allocator>;

private:
  AType uid_;
  /** Given an abstract variable `v`, `avar2lvar[VID(v)]` is the name of the variable. */
  battery::vector<LName, allocator_type> avar2lvar;

public:
  CUDA VarEnv(AType uid, const Allocator& allocator = Allocator())
   : uid_(uid), avar2lvar(allocator) {}

  CUDA VarEnv(AType uid, int capacity, const Allocator& allocator = Allocator())
   : uid_(uid), avar2lvar(allocator)
  {
    avar2lvar.reserve(capacity);
  }

  CUDA VarEnv(VarEnv&& other): uid_(other.uid_), avar2lvar(std::move(other.avar2lvar)) {}

  template<class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator())
   : uid_(other.uid_), avar2lvar(other.avar2lvar, allocator) {}

  CUDA this_type& operator=(const this_type& other) {
    uid_ = other.uid_;
    avar2lvar = other.avar2lvar;
    return *this;
  }

  CUDA AType uid() const {
    return uid_;
  }

  CUDA size_t capacity() const {
    return avar2lvar.capacity();
  }

  CUDA size_t size() const {
    return avar2lvar.size();
  }

  CUDA const LName& operator[](size_t i) const {
    assert(i < size());
    return avar2lvar[i];
  }

  CUDA const LName& to_lvar(AVar av) const {
    assert(VID(av) < size());
    return avar2lvar[VID(av)];
  }

  CUDA thrust::optional<AVar> to_avar(const LName& lv) const {
    for(int i = 0; i < size(); ++i) {
      if(avar2lvar[i] == lv) {
        return make_var(uid(), i);
      }
    }
    return {};
  }

  CUDA AVar add(LName lv) {
    avar2lvar.push_back(std::move(lv));
    return make_var(uid(), size() - 1);
  }

  CUDA void reserve(int new_cap) {
    avar2lvar.reserve(new_cap);
  }
};

/** Given a formula `f` and an environment, return the first abstract variable (`AVar`) occurring in `f` or `{}` if `f` has no variable in `env`. */
template <class F, class Allocator>
CUDA thrust::optional<AVar> var_in(const F& f, const VarEnv<Allocator>& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      return g.v();
    case F::E:
      return env.to_avar(battery::get<0>(g.exists()));
    case F::LV:
      return env.to_avar(g.lv());
    default:
      return {};
  }
}
}

#endif
