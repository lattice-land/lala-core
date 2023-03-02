// Copyright 2022 Pierre Talbot

#ifndef PRE_DUAL_HPP
#define PRE_DUAL_HPP

namespace lala {

/** The dual of a pre-universe, called "chain predual", dualizes lattice and non-lattice operations when the underlying lattice is a chain.
    In addition to the dual lattice, we must reverse the approximation direction (UNDER to OVER, and OVER to UNDER) when interpreting a formula in `interpret` (that only works because `L` is a chain). */
template<class L>
struct ChainPreDual {
  static_assert(L::is_totally_ordered, "This dual construction only works over lattices that are totally ordered (chain).");

  using reverse_type = L;
  using value_type = typename L::value_type;
  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = L::preserve_top;
  constexpr static const bool preserve_top = L::preserve_bot;
  constexpr static const bool injective_concretization = L::injective_concretization;
  constexpr static const bool preserve_inner_covers = L::preserve_inner_covers;
  constexpr static const bool complemented = L::complemented;
  constexpr static const bool increasing = !L::increasing;
  constexpr static const char* name = L::dual_name;
  constexpr static const char* dual_name = L::name;
  constexpr static const bool is_arithmetic = L::is_arithmetic;
  CUDA constexpr static value_type zero() { return L::zero(); }
  CUDA constexpr static value_type one() { return L::one(); }

  template<class F>
  using iresult = typename L::iresult<F>;

  template<class F, class Sort, bool dualize = false>
  CUDA static iresult<F> interpret(const F& f, const Sort& sort, Approx appx) {
    return L::template interpret<F, Sort, !dualize>(f, sort, dapprox(appx));
  }

  /** We suppose `interpret_type` always maps to `bot` or an error. */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    auto r = L::interpret_type(f);
    if(r.has_value() && r.value() == L::bot()) {
      return std::move(r).map(bot());
    }
    return r;
  }

  CUDA static constexpr Sig sig_order() { return L::dual_sig_order(); }
  CUDA static constexpr Sig dual_sig_order() { return L::sig_order(); }
  CUDA static constexpr Sig sig_strict_order() { return L::dual_sig_strict_order(); }
  CUDA static constexpr Sig dual_sig_strict_order() { return L::sig_strict_order(); }
  CUDA static constexpr value_type bot() { return L::top(); }
  CUDA static constexpr value_type top() { return L::bot(); }
  CUDA static constexpr value_type join(value_type x, value_type y) { return L::meet(x, y); }
  CUDA static constexpr value_type meet(value_type x, value_type y) { return L::join(x, y); }
  CUDA static constexpr bool order(value_type x, value_type y) { return L::order(y, x); }
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return L::strict_order(y, x); }
  CUDA static constexpr bool has_unique_next(value_type x) { return L::has_unique_prev(x); }
  CUDA static constexpr bool has_unique_prev(value_type x) { return L::has_unique_next(x); }
  CUDA static value_type next(value_type i) { return L::prev(i); }
  CUDA static value_type prev(value_type i) { return L::next(i); }
  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) { return L::is_supported_fun(dapprox(appx), sig); }

  template<Approx appx, Sig sig, class... Args>
  CUDA static constexpr auto fun(Args... args) {
    return L::template fun<dapprox(appx), sig>(args...);
  }
};

} // namespace lala

#endif
