// Copyright 2021 Pierre Talbot

#ifndef CARTESIAN_PRODUCT_HPP
#define CARTESIAN_PRODUCT_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "vector.hpp"
#include "ast.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "universes/upset_universe.hpp"

namespace lala {

template<class... As>
class CartesianProduct;

namespace impl {
  template<class... As>
  auto index_sequence_of(const CartesianProduct<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template<class... As>
  auto index_sequence_of(const battery::tuple<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template <class A, class B>
  auto index_sequence_of(const A& a, const B& b) {
    static_assert(impl::index_sequence_of(a)::size() == impl::index_sequence_of(b)::size());
    return impl::index_sequence_of(a);
  }

  template<class... Os>
  CUDA CartesianProduct<Os...> make_cp(Os... os) {
    return CartesianProduct<Os...>(os...);
  }

  template<class A>
  struct is_product {
    static constexpr bool value = false;
  };

  template<class... As>
  struct is_product<CartesianProduct<As...>> {
    static constexpr bool value = true;
  };

  template<class... As>
  struct is_product<battery::tuple<As...>> {
    static constexpr bool value = true;
  };

  template<class A>
  inline static constexpr bool is_product_v = is_product<A>::value;
}

/** The Cartesian product abstract domain is a _domain transformer_ combining several abstract domains.
Concretization function: \f$ \gamma((a_1, \ldots, a_n)) = \sqcap_{i \leq n} \gamma_i(a_i) \f$. */
template<class... As>
class CartesianProduct {
public:
  template<size_t i>
  using type_of = typename battery::tuple_element<i, battery::tuple<As...>>::type;
  constexpr static size_t n = battery::tuple_size<battery::tuple<As...>>{};
  using this_type = CartesianProduct<As...>;
  using memory_type = typename type_of<0>::memory_type;

  template<class...Bs> friend class CartesianProduct;

  template<class F>
  using iresult = IResult<this_type, F>;

  using value_type = battery::tuple<typename As::value_type...>;

  constexpr static const char* name = "CartesianProduct";

private:
  battery::tuple<As...> val;

public:
  CUDA CartesianProduct(const As&... as): val(battery::make_tuple(as...)) {}
  CUDA CartesianProduct(As&&... as): val(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA CartesianProduct(typename As::value_type... vs): val(battery::make_tuple(As(vs)...)) {}

  template<class... Bs>
  CUDA CartesianProduct(const CartesianProduct<Bs...>& other): val(other.val) {}

  /** Cartesian product initialized to \f$ (\bot_1, \ldots, \bot_n) \f$. */
  CUDA static this_type bot() {
    return CartesianProduct(As::bot()...);
  }

  /** Cartesian product initialized to \f$ (\top_1, \ldots, \top_n) \f$. */
  CUDA static this_type top() {
    return CartesianProduct(As::top()...);
  }

  /** Interpret the formula in the component `i`. */
  template<size_t i, class F, class Env>
  CUDA static iresult<F> interpret_one(const F& f, const Env& env) {
    auto one = type_of<i>::interpret(f, env);
    if(one.is_ok()) {
      auto res = bot();
      project<i>(res).tell(one.value());
      return one.map(res);
    }
    else {
      auto res = iresult<F>(IError<F>(true, name, "The requested component of this Cartesian product cannot interpret this formula.", f));
      return res.join_errors(std::move(one));
    }
  }

private:
  template<size_t i = 0, class F, class Env>
  CUDA static iresult<F> interpret_all(const F& f, this_type res, bool empty, const Env& env) {
    if constexpr(i == n) {
      if(empty) {
        return iresult<F>(IError<F>(true, name, "No component of this Cartesian product can interpret this formula.", f));
      }
      else {
        return iresult<F>(std::move(res));
      }
    }
    else {
      auto one = type_of<i>::interpret(f, env);
      if(one.is_ok()) {
        project<i>(res).tell(one.value());
        return interpret_all<i+1>(f, std::move(res), false, env).join_warnings(one);
      }
      else {
        auto r = interpret_all<i+1>(f, std::move(res), empty, env);
        if(!r.is_ok()) {
          r.join_errors(std::move(one));
        }
        return std::move(r);
      }
    }
  }

public:
  /** Interpret the formula `f` in all sub-universes in which `f` is interpretable. */
  template<class F, class Env>
  CUDA static iresult<F> interpret(const F& f, const Env& env) {
    return interpret_all(f, bot(), true, env);
  }

private:
  // The non-const version must stay private, otherwise it violates the PCCP model since the caller might not check if the updated value is strictly greater w.r.t. lattice order.
  template<size_t i>
  CUDA type_of<i>& project() {
    return battery::get<i>(val);
  }

  template<size_t... I>
  CUDA value_type value_(std::index_sequence<I...>) const {
    return value_type(project<I>().value()...);
  }

  template<size_t... I>
  CUDA local::BInc is_top_(std::index_sequence<I...>) const {
    return ... || project<I>().is_top();
  }

  template<size_t... I>
  CUDA local::BDec is_bot_(std::index_sequence<I...>) const {
    return ... && project<I>().is_bot();
  }

public:
  template<size_t i>
  CUDA const type_of<i>& project() const {
    return battery::get<i>(val);
  }

  CUDA value_type value() const {
    return value_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \exists{j \geq i},~\gamma(a_j) = \top^\flat \f$, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return is_top_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \forall{j \geq i},~\gamma(a_j) = \bot^\flat \f$, `false` otherwise. */
  CUDA local::BDec is_bot() const {
    return is_bot_(std::index_sequence_for<As...>{});
  }

private:
  template<size_t i = 0, class M, class... Bs>
  CUDA this_type& tell_(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    if constexpr (i < n) {
      project<i>().tell(other.project<i>(), has_changed);
      return tell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA this_type& tell_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      project<i>().tell(other.project<i>());
      return tell_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class M, class... Bs>
  CUDA this_type& dtell_(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    if constexpr (i < n) {
      project<i>().dtell(other.project<i>(), has_changed);
      return dtell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA this_type& dtell_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      project<i>().dtell(other.project<i>());
      return dtell_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA this_type& dtell_bot_() {
    if constexpr (i < n) {
      project<i>().dtell_bot();
      return dtell_bot_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA bool extract_(CartesianProduct<Bs...>& ua) {
    if constexpr (i < n) {
      bool is_under = project<i>().extract(ua.project<i>());
      return is_under && extract_<i+1>(ua);
    }
    else {
      return true;
    }
  }

public:
  /** \f$ \top \f$ is only told in the first component (since it is sufficient for the whole Cartesian product to become top as well).  */
  CUDA this_type& tell_top() {
    project<0>().tell_top();
    return *this;
  }

  template <class M, class Bs...>
  CUDA this_type& tell(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    return tell_(other, has_changed);
  }

  template<size_t i, class Ai, class M>
  CUDA this_type& tell(const Ai& a, BInc<M>& has_changed) {
    project<i>().tell(a, has_changed);
    return *this;
  }

  template <class Bs...>
  CUDA this_type& tell(const CartesianProduct<Bs...>& other) {
    return tell_(other);
  }

  template<size_t i, class Ai>
  CUDA this_type& tell(const Ai& a) {
    project<i>().tell(a);
    return *this;
  }

  CUDA this_type& dtell_bot() {
    dtell_bot_();
    return *this;
  }

  template <class M, class Bs...>
  CUDA this_type& dtell(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    return dtell_(other, has_changed);
  }

  template<size_t i, class Ai, class M>
  CUDA this_type& dtell(const Ai& a, BInc<M>& has_changed) {
    project<i>().dtell(a, has_changed);
    return *this;
  }

  template <class Bs...>
  CUDA this_type& dtell(const CartesianProduct<Bs...>& other) {
    return dtell_(other);
  }

  template<size_t i, class Ai>
  CUDA this_type& dtell(const Ai& a) {
    project<i>().dtell(a);
    return *this;
  }

  /** For correctness, the parameter `ua` must be stored in a local memory. */
  template <class... Bs>
  CUDA bool extract(CartesianProduct<Bs...>& ua) const {
    return extract_(ua);
  }

// Implementation of the logical signature.

private:
  template<Approx appx, Sig sig, class A, size_t... I>
  CUDA static constexpr auto fun_(const A& a, std::index_sequence<I...>)
  {
    return make_cp(fun<appx, sig>(project<I>(a))...);
  }

  template<Approx appx, Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(fun<appx, sig>(project<I>(a), project<I>(b))...);
  }

  template<Approx appx, Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_left(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(fun<appx, sig>(project<I>(a), b)...);
  }

  template<Approx appx, Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_right(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(fun<appx, sig>(a, project<I>(b))...);
  }

public:
  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    return ... && As::is_supported_fun(appx, sig);
  }

  /** Given a product \f$ (x_1, \ldots, x_n) \f$, returns \f$ (f(x_1), \ldots, f(x_n)) \f$. */
  template<Approx appx, Sig sig, class A>
  CUDA static constexpr auto fun(const A& a) {
    if constexpr(impl::is_product_v<A>) {
      return impl::fun_<appx, sig>(a, impl::index_sequence_of(a));
    }
    else {
      static_assert(impl::is_product_v<A>, "The argument of a unary function on Cartesian product must be a product.");
    }
  }

  /** Given two product \f$ (x_1, \ldots, x_n) \f$ and \f$ (y_1, \ldots, y_n) \f$, returns \f$ (f(x_1, y_1), \ldots, f(x_n, y_n)) \f$.
      If either the left or right operand is not a product, returns \f$ (f(x_1, c), \ldots, f(x_n, c)) \f$ or  \f$ (f(c, y_1), \ldots, f(c, y_n)) \f$. */
  template<Approx appx, Sig sig, class A, class B>
  CUDA static constexpr auto fun(const A& a, const B& b) {
    if constexpr(impl::is_product_v<A> && impl::is_product_v<B>) {
      return impl::fun_<appx, sig>(a, b, impl::index_sequence_of(a, b));
    }
    else if constexpr(impl::is_product_v<A> && !impl::is_product_v<B>) {
      return impl::fun_left<appx, sig>(a, b, impl::index_sequence_of(a));
    }
    else if constexpr(!impl::is_product_v<A> && impl::is_product_v<B>) {
      return impl::fun_right<appx, sig>(a, b, impl::index_sequence_of(b));
    }
    else {
      static_assert(impl::is_product_v<A> || impl::is_product_v<B>, "At least one argument of a binary function on Cartesian product must be a product.");
    }
  }

private:
  template<size_t i, class Env>
  CUDA TFormula<Allocator> deinterpret_(AVar x,
    typename TFormula<Allocator>::Sequence&& seq, const Env& env) const
  {
    if constexpr(i < n) {
      seq[i] = project<i>().deinterpret(x, env);
      return deinterpret_<i+1, Allocator>(x, std::move(seq), env);
    }
    else {
      return TFormula<Allocator>::make_nary(
        AND,
        std::move(seq),
        AID(x), EXACT, env.get_allocator());
    }
  }

public:
  template<class Env>
  CUDA TFormula<typename Env::allocator_type> deinterpret(AVar x, const Env& env) const {
    using allocator_t = typename Env::allocator_type;
    return deinterpret_<0, Env>(x, typename TFormula<allocator_t>::Sequence(n, env.get_allocator()), env);
  }

private:
  template<size_t i = 0>
  CUDA void print_() const {
    if constexpr(i < n) {
      project<i>().print();
      if constexpr(i < n - 1) {
        printf("\n");
        print<i+1>();
      }
    }
  }

public:
  CUDA void print() const {
    print_();
  }
};

/// Similar to `cp.template project<i>()`, just to avoid the ".template" syntax.
template<size_t i, class... As>
CUDA const typename CartesianProduct<As...>::type_of<i>&
project(const CartesianProduct<As...>& cp) {
  return cp.template project<i>();
}

template<size_t i, class... As>
CUDA const battery::tuple<As...>& project(const battery::tuple<As...>& t) {
  return battery::get<i>(t);
}

// Lattice operators
namespace impl {
  template<class A, class B, size_t... I>
  CUDA auto join_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(join(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA auto meet_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(meet(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA bool leq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... && (project<I>(a) <= project<I>(b));
  }

  template<class A, class B, size_t... I>
  CUDA bool lt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... && (project<I>(a) < project<I>(b));
  }

  template<class A, class B, size_t... I>
  CUDA bool geq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... && (project<I>(a) >= project<I>(b));
  }

  template<class A, class B, size_t... I>
  CUDA bool gt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... && (project<I>(a) > project<I>(b));
  }

  template<class A, class B, size_t... I>
  CUDA bool eq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... && (project<I>(a) == project<I>(b));
  }

  template<class A, class B, size_t... I>
  CUDA bool neq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return ... || (project<I>(a) != project<I>(b));
  }
}

/** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
template<class A, class B>
CUDA auto join(const A& a, const B& b)
{
  return impl::join_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
template<class A, class B>
CUDA auto meet(const A& a, const B& b)
{
  return impl::meet_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \leq (b_1, \ldots, b_n) \f$ holds when \f$ \forall{i \leq n},~a_i \leq_i b_i \f$. */
template<class A, class B>
CUDA bool operator<=(const A& a, const B& b)
{
  return impl::leq_(a, b, impl::index_sequence_of(a, b));
}

template<class A, class B>
CUDA bool operator<(const A& a, const B& b)
{
  return impl::lt_(a, b, impl::index_sequence_of(a, b));
}

template<class A, class B>
CUDA bool operator>=(const A& a, const B& b)
{
  return impl::geq_(a, b, impl::index_sequence_of(a, b));
}

template<class A, class B>
CUDA bool operator>(const A& a, const B& b)
{
  return impl::gt_(a, b, impl::index_sequence_of(a, b));
}

template<class A, class B>
CUDA bool operator==(const A& a, const B& b)
{
  return impl::eq_(a, b, impl::index_sequence_of(a, b));
}

template<class A, class B>
CUDA bool operator!=(const A& a, const B& b)
{
  return impl::neq_(a, b, impl::index_sequence_of(a, b));
}

} // namespace lala

#endif
