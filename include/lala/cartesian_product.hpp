// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_CARTESIAN_PRODUCT_HPP
#define LALA_CORE_CARTESIAN_PRODUCT_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "logic/logic.hpp"
#include "universes/primitive_upset.hpp"

namespace lala {

template<class... As>
class CartesianProduct;

namespace impl {
  template<class... As>
  CUDA constexpr auto index_sequence_of(const CartesianProduct<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template<class... As>
  CUDA constexpr auto index_sequence_of(const battery::tuple<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template <class A, class B>
  CUDA constexpr auto index_sequence_of(const A& a, const B& b) {
    static_assert(decltype(impl::index_sequence_of(a))::size() == decltype(impl::index_sequence_of(b))::size());
    return impl::index_sequence_of(a);
  }

  template<class... Os>
  CUDA constexpr typename CartesianProduct<Os...>::local_type make_cp(Os... os) {
    return typename CartesianProduct<Os...>::local_type(os...);
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
Concretization function: \f$ \gamma((a_1, \ldots, a_n)) = \bigcap_{i \leq n} \gamma_i(a_i) \f$. */
template<class... As>
class CartesianProduct {
public:
  template<size_t i>
  using type_of = typename battery::tuple_element<i, battery::tuple<As...>>::type;
  constexpr static size_t n = battery::tuple_size<battery::tuple<As...>>{};
  static_assert(n > 0, "CartesianProduct must not be empty.");
  using this_type = CartesianProduct<As...>;
  using local_type = CartesianProduct<typename As::local_type...>;
  using memory_type = typename type_of<0>::memory_type;

  template<class...Bs> friend class CartesianProduct;

  using value_type = battery::tuple<typename As::value_type...>;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = (... && As::sequential);
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = (... && As::preserve_bot);
  constexpr static const bool preserve_top = (... && As::preserve_top);
  constexpr static const bool preserve_join = (... && As::preserve_join);
  constexpr static const bool preserve_meet = false; // false in general, not sure if there are conditions the underlying universes could satisfy to make this true.
  constexpr static const bool injective_concretization = (... && As::injective_concretization);
  constexpr static const bool preserve_concrete_covers = (... && As::preserve_concrete_covers);
  constexpr static const bool complemented = false;
  constexpr static const char* name = "CartesianProduct";

private:
  battery::tuple<As...> val;

public:
  /** Initialize a Cartesian product to bottom using default constructors. */
  constexpr CartesianProduct() = default;
  CUDA constexpr CartesianProduct(const As&... as): val(battery::make_tuple(as...)) {}
  CUDA constexpr CartesianProduct(As&&... as): val(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA constexpr CartesianProduct(typename As::value_type... vs): val(battery::make_tuple(As(vs)...)) {}

  template<class... Bs>
  CUDA constexpr CartesianProduct(const CartesianProduct<Bs...>& other): val(other.val) {}

  template<class... Bs>
  CUDA constexpr CartesianProduct(CartesianProduct<Bs...>&& other): val(std::move(other.val)) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class... Bs>
  CUDA constexpr this_type& operator=(const CartesianProduct<Bs...>& other) {
    if constexpr(sequential) {
      val = other.val;
      return *this;
    }
    else {
      static_assert(sequential, "operator= seq (CartesianProduct).");
    }
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      val = other.val;
      return *this;
    }
    else {
      static_assert(sequential, "operator= seq (CartesianProduct).");
    }
  }

  /** Cartesian product initialized to \f$ (\bot_1, \ldots, \bot_n) \f$. */
  CUDA static constexpr local_type bot() {
    return local_type(As::bot()...);
  }

  /** Cartesian product initialized to \f$ (\top_1, \ldots, \top_n) \f$. */
  CUDA static constexpr local_type top() {
    return local_type(As::top()...);
  }

private:
  /** Interpret the formula in the component `i`. */
  template<IKind kind, bool diagnose, size_t i, class F, class Env, class... Bs>
  CUDA NI static bool interpret_one(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    return type_of<i>::template interpret<kind, diagnose>(f, env, k.template project<i>(), diagnostics);
  }

  template<IKind kind, bool diagnose, size_t i = 0, class F, class Env, class... Bs>
  CUDA NI static bool interpret_all(const F& f, CartesianProduct<Bs...>& k, bool empty, const Env& env, IDiagnostics& diagnostics) {
    if constexpr(i == n) {
      return !empty;
    }
    else {
      bool res = interpret_one<kind, diagnose, i>(f, env, k, diagnostics);
      return interpret_all<kind, diagnose, i+1>(f, k, empty && !res, env, diagnostics);
    }
  }

public:
  template<size_t i, bool diagnose = false, class F, class Env, class... Bs>
  CUDA static bool interpret_one_tell(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    return interpret_one<IKind::TELL, diagnose, i>(f, env, k, diagnostics);
  }

  template<size_t i, bool diagnose = false, class F, class Env, class... Bs>
  CUDA static bool interpret_one_ask(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    return interpret_one<IKind::ASK, diagnose, i>(f, env, k, diagnostics);
  }

  template<IKind kind, bool diagnose = false, class F, class Env, class... Bs>
  CUDA NI static bool interpret(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    CALL_WITH_ERROR_CONTEXT(
      "No component of this Cartesian product can interpret this formula.",
      (interpret_all<kind, diagnose>(f, k, true, env, diagnostics))
    );
  }

  /** Interpret the formula `f` in all sub-universes in which `f` is interpretable. */
  template<bool diagnose, class F, class Env, class... Bs>
  CUDA static bool interpret_tell(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    return interpret<IKind::TELL, diagnose>(f, env, k, diagnostics);
  }

  template<bool diagnose, class F, class Env, class... Bs>
  CUDA static bool interpret_ask(const F& f, const Env& env, CartesianProduct<Bs...>& k, IDiagnostics& diagnostics) {
    return interpret<IKind::ASK, diagnose>(f, env, k, diagnostics);
  }

private:
  template<size_t... I>
  CUDA constexpr value_type value_(std::index_sequence<I...>) const {
    return value_type(project<I>().value()...);
  }

  template<size_t... I>
  CUDA constexpr local::B is_top_(std::index_sequence<I...>) const {
    return (... || project<I>().is_top());
  }

  template<size_t... I>
  CUDA constexpr local::B is_bot_(std::index_sequence<I...>) const {
    return (... && project<I>().is_bot());
  }

public:
  /** You must use the lattice interface (tell methods) to modify the projected type, if you use assignment you violate the PCCP model. */
  template<size_t i>
  CUDA constexpr type_of<i>& project() {
    return battery::get<i>(val);
  }

  template<size_t i>
  CUDA constexpr const type_of<i>& project() const {
    return battery::get<i>(val);
  }

  CUDA constexpr value_type value() const {
    return value_(std::index_sequence_for<As...>{});
  }

  /** \return `true` if \f$ \exists{j \geq i},~\gamma(a_j) = \top^\flat \f$, `false` otherwise.
   * @parallel @order-preserving @increasing */
  CUDA constexpr local::B is_top() const {
    return is_top_(std::index_sequence_for<As...>{});
  }

  /** \return `true` if \f$ \forall{j \geq i},~\gamma(a_j) = \bot^\flat \f$, `false` otherwise.
   * @parallel @order-preserving @decreasing
   */
  CUDA constexpr local::B is_bot() const {
    return is_bot_(std::index_sequence_for<As...>{});
  }

private:
  template<size_t i = 0, class... Bs>
  CUDA constexpr bool join_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      bool has_changed = project<i>().join(other.template project<i>());
      has_changed |= join_<i+1>(other);
      return has_changed;
    }
    else {
      return false;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA constexpr bool meet_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      bool has_changed = project<i>().meet(other.template project<i>());
      has_changed |= meet_<i+1>(other);
      return has_changed;
    }
    else {
      return false;
    }
  }

  template<size_t i = 0>
  CUDA constexpr void meet_bot_() {
    if constexpr (i < n) {
      project<i>().meet_bot();
      meet_bot_<i+1>();
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA constexpr bool extract_(CartesianProduct<Bs...>& ua) {
    if constexpr (i < n) {
      bool is_under = project<i>().extract(ua.template project<i>());
      return is_under && extract_<i+1>(ua);
    }
    else {
      return true;
    }
  }

  template <size_t i = 0>
  CUDA constexpr void join_top_() {
    if constexpr(i < n) {
      project<i>().join_top();
      join_top_<i+1>();
    }
  }

public:
  CUDA constexpr void join_top() {
    join_top_();
  }

  template <class... Bs>
  CUDA constexpr bool join(const CartesianProduct<Bs...>& other) {
    return join_(other);
  }

  template<size_t i, class Ai>
  CUDA constexpr bool join(const Ai& a) {
    return project<i>().join(a);
  }

  CUDA constexpr void meet_bot() {
    meet_bot_();
  }

  template <class... Bs>
  CUDA constexpr bool meet(const CartesianProduct<Bs...>& other) {
    return meet_(other);
  }

  template<size_t i, class Ai>
  CUDA constexpr bool meet(const Ai& a) {
    return project<i>().meet(a);
  }

  /** For correctness, the parameter `ua` must be stored in a local memory. */
  template <class... Bs>
  CUDA constexpr bool extract(CartesianProduct<Bs...>& ua) const {
    return extract_(ua);
  }

// Implementation of the logical signature.

private:
  template<class A, size_t... I>
  CUDA constexpr void project(Sig fun, const A& a, std::index_sequence<I...>)
  {
    (project<I>().project(fun, a.template project<I>()),...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr void project(Sig fun, const A& a, const B& b, std::index_sequence<I...>)
  {
    (project<I>().project(fun, a.template project<I>(), b.template project<I>()),...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr void project_left(Sig fun, const A& a, const B& b, std::index_sequence<I...>)
  {
    (project<I>().project(fun, a.template project<I>(), b),...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr void project_right(Sig fun, const A& a, const B& b, std::index_sequence<I...>)
  {
    (project<I>().project(fun, a, b.template project<I>()),...);
  }

public:
  CUDA static constexpr bool is_trivial_fun(Sig fun) {
    return (... && As::is_trivial_fun(fun));
  }

  /** Given a product \f$ (x_1, \ldots, x_n) \f$, join in-place \f$ (fun(x_1), \ldots, fun(x_n)) \f$. */
  template<class... Bs>
  CUDA constexpr void project(Sig fun, const CartesianProduct<Bs...>& a) {
    project(fun, a, impl::index_sequence_of(a));
  }

  /** Given two product \f$ (x_1, \ldots, x_n) \f$ and \f$ (y_1, \ldots, y_n) \f$, join in-place \f$ (fun(x_1, y_1), \ldots, fun(x_n, y_n)) \f$.
      If either the left or right operand is not a product, join in-place \f$ (fun(x_1, c), \ldots, fun(x_n, c)) \f$ or  \f$ (fun(c, y_1), \ldots, fun(c, y_n)) \f$. */
  template<class... As2, class... Bs>
  CUDA constexpr void project(Sig fun, const CartesianProduct<As2...>& a, const CartesianProduct<Bs...>& b) {
    project(fun, a, b, impl::index_sequence_of(a, b));
  }

  template<class... As2, class B>
  CUDA constexpr void project(Sig fun, const CartesianProduct<As2...>& a, const B& b) {
    project_left(fun, a, b, impl::index_sequence_of(a));
  }

  template<class A, class... Bs>
  CUDA constexpr auto project(Sig fun, const A& a, const CartesianProduct<Bs...>& b) {
    project_right(fun, a, b, impl::index_sequence_of(b));
  }

private:
  template<size_t i, class Env, class Allocator = typename Env::allocator_type>
  CUDA NI TFormula<Allocator> deinterpret_(AVar x,
    typename TFormula<Allocator>::Sequence& seq, const Env& env, const Allocator& allocator) const
  {
    if constexpr(i < n) {
      auto f = project<i>().deinterpret(x, env, allocator);
      if(!f.is_true()) {
        seq.push_back(project<i>().deinterpret(x, env, allocator));
      }
      return deinterpret_<i+1, Env>(x, seq, env, allocator);
    }
    else {
      if(seq.size() == 1) {
        return std::move(seq[0]);
      }
      else {
        return TFormula<Allocator>::make_nary(
          AND,
          std::move(seq));
      }
    }
  }

public:
  template<class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(AVar x, const Env& env, const Allocator& allocator = Allocator()) const {
    typename TFormula<Allocator>::Sequence seq(allocator);
    return deinterpret_<0, Env>(x, seq, env, allocator);
  }

private:
  template<size_t i = 0>
  CUDA NI void print_() const {
    if constexpr(i < n) {
      project<i>().print();
      if constexpr(i < n - 1) {
        printf("\n");
        print_<i+1>();
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
CUDA constexpr const typename CartesianProduct<As...>::template type_of<i>&
project(const CartesianProduct<As...>& cp) {
  return cp.template project<i>();
}

template<size_t i, class... As>
CUDA constexpr typename CartesianProduct<As...>::template type_of<i>&
project(CartesianProduct<As...>& cp) {
  return cp.template project<i>();
}

// Lattice operators
namespace impl {
  template<class A, class B, size_t... I>
  CUDA constexpr auto fjoin_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(fjoin(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr auto fmeet_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(fmeet(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool eq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) == project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool neq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... || (project<I>(a) != project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool leq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) <= project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool lt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return leq_(a, b, std::index_sequence<I...>{}) && neq_(a, b, std::index_sequence<I...>{});
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool geq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) >= project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool gt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return geq_(a, b, std::index_sequence<I...>{}) && neq_(a, b, std::index_sequence<I...>{});
  }
}

/** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
template<class... As, class... Bs>
CUDA constexpr auto fjoin(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::fjoin_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
template<class... As, class... Bs>
CUDA constexpr auto fmeet(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::fmeet_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \leq (b_1, \ldots, b_n) \f$ holds when \f$ \forall{i \leq n},~a_i \leq_i b_i \f$. */
template<class... As, class... Bs>
CUDA constexpr bool operator<=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::leq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator<(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::lt_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator>=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::geq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator>(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::gt_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator==(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::eq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator!=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::neq_(a, b, impl::index_sequence_of(a, b));
}

namespace impl {
  template<size_t i = 0, class... As>
  void std_print(std::ostream &s, const CartesianProduct<As...> &cp) {
    if constexpr(i < CartesianProduct<As...>::n) {
      s << project<i>(cp);
      if constexpr(i < CartesianProduct<As...>::n - 1) {
        s << ", ";
        std_print<i+1>(s, cp);
      }
    }
  }
}

template<class A, class... As>
std::ostream& operator<<(std::ostream &s, const CartesianProduct<A, As...> &cp) {
// There is a weird compilation bug with `template<class... As>` where the compiler tries to instantiate << with an empty sequence of templates.
// Forcing at least one template solves the problem.
  s << "(";
  impl::std_print<0>(s, cp);
  s << ")";
  return s;
}

} // namespace lala

#endif
