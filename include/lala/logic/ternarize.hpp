#ifndef LALA_CORE_TERNARIZE_HPP
#define LALA_CORE_TERNARIZE_HPP

#include "ast.hpp"

namespace lala {
namespace impl {

template <class F>
class Ternarizer
{
public:
  using allocator_type = battery::standard_allocator;

  /** A constraint in ternary form is either unary or of the form `x = (y <op> z)`. */
  static bool is_ternary_form(const F& f) {
    int vars = num_vars(f);
    return vars == 1 ||
      (vars == 3 && f.is_binary() && f.sig() == EQ &&
         ((f.seq(0).is_variable() && f.seq(1).is_binary() && is_ternary_op(f.seq(1).sig()) && f.seq(1).seq(0).is_variable() && f.seq(1).seq(1).is_variable())
       || (f.seq(1).is_variable() && f.seq(0).is_binary() && is_ternary_op(f.seq(0).sig()) && f.seq(0).seq(0).is_variable() && f.seq(0).seq(1).is_variable())));
  }

  static bool is_ternary_op(Sig sig) {
    return sig == MAX || sig == MIN || sig == EQ || sig == LEQ || (!is_logical(sig) && !is_predicate(sig));
  }

  Ternarizer():
    introduced_int_vars(0),
    introduced_bool_vars(0),
    introduced_constants(0)
  {}

private:
  battery::vector<F, allocator_type> conjunction;
  battery::vector<F, allocator_type> existentials;
  std::unordered_map<std::string, int> name2exists;
  unsigned int introduced_int_vars;
  unsigned int introduced_bool_vars;
  unsigned int introduced_constants;

  F introduce_var(const std::string& name, auto sort, bool constant) {
    auto var_name = LVar<allocator_type>(name.data());
    existentials.push_back(F::make_exists(UNTYPED, var_name, sort));
    assert(!name2exists.contains(name));
    name2exists[name] = existentials.size() - 1;
    if(constant) { introduced_constants++; }
    else if(sort.is_int()) { introduced_int_vars++; }
    else if(sort.is_bool()) { introduced_bool_vars++; }
    return F::make_lvar(UNTYPED, var_name);
  }

  F introduce_int_var() {
    std::string name = "__VAR_Z_" + std::to_string(introduced_int_vars);
    return introduce_var(name, Sort<allocator_type>(Sort<allocator_type>::Int), false);
  }

  F introduce_bool_var() {
    std::string name = "__VAR_B_" + std::to_string(introduced_bool_vars);
    return introduce_var(name, Sort<allocator_type>(Sort<allocator_type>::Bool), false);
  }

  F var_of(int x) const {
    return F::make_lvar(UNTYPED, battery::get<0>(existentials[x].exists()));
  }

  F ternarize_constant(const F& f) {
    assert(f.is(F::Z) || f.is(F::B));
    auto index = f.is(F::Z) ? f.z() : (int)f.b();
    std::string name = "__CONSTANT_" + (index < 0 ? std::string("m") : std::string("")) + std::to_string(abs(index));
    // if the constant is already a logical variable, we return it.
    if (name2exists.contains(name)) {
      return var_of(name2exists[name]);
    }
    auto var = introduce_var(name, f.sort().value(), true);
    conjunction.push_back(F::make_binary(var, EQ, f));
    return var;
  }

  /** Create the ternary formula `x = y <sig> z`. */
  F push_ternary(const F& x, const F& y, Sig sig, const F& z) {
    conjunction.push_back(
      F::make_binary(x, EQ, F::make_binary(y, sig, z)));
    return x;
  }

  F ternarize_unary(const F& f) {
    auto x = ternarize(f.seq(0));
    switch(f.sig()) {
      /** -x ~~> t = 0 - x */
      case NEG: return push_ternary(introduce_int_var(), ternarize(F::make_z(0)), SUB, x);
      /** |x| ~~> t1 = 0 - x /\ t2 = max(x, t1) */
      case ABS: return push_ternary(introduce_int_var(), x, MAX, ternarize(F::make_unary(NEG, x)));
      /** NOT x ~~> ternarize(x = 0) ~~> t = (x = 0) */
      case NOT: return ternarize(F::make_binary(x, EQ, F::make_z(0)));
      default: {
        printf("Unary operator %s not supported\n", string_of_sig(f.sig()));
        printf("In formula: "); f.print(); printf("\n");
        // assert(false);
        return f;
      }
    }
  }

  bool is_boolean(const F& f) {
    assert(f.is(F::LV));
    std::string varname(f.lv().data());
    assert(name2exists.contains(varname));
    return battery::get<1>(existentials[name2exists[varname]].exists()).is_bool();
  }

  /** Let `t` be a variable in a logical context, e.g. X OR Y.
   * If `t` is an integer, the semantics is that `t` is true whenever `t != 0`, and not only when `t == 1`.
   */
  F booleanize(const F& t, Sig sig) {
    if(is_logical(sig) && !is_boolean(t)) {
      return ternarize(F::make_binary(t, NEQ, F::make_z(0)));
    }
    return t;
  }

  F ternarize_binary(F f, bool toplevel) {
    /** We introduce a new temporary variable `t0`.
     * The type of `t0` is decided by the return type of the operator and whether we are at toplevel or not.
     */
    F t0;
    F t1;
    F t2;
    /** We first handle "almost ternarized" constraint.
     * If the symbol is already an equality with a variable on one side, we only need to ternarize the other half.
     * We set t0 to be the variable and proceeds. */
    if((f.seq(0).is_variable() && f.seq(1).is_binary()) || (f.seq(1).is_variable() && f.seq(0).is_binary()) && (f.sig() == EQUIV || f.sig() == EQ))  {
      int left = f.seq(1).is_variable();
      int right = f.seq(0).is_variable();
      t0 = f.seq(left);
      f = f.seq(right);
      toplevel = false;
    }
    /** We don't need to create t0 for these formulas at toplevel. */
    else if(toplevel && (f.sig() == NEQ || f.sig() == XOR || f.sig() == IMPLY || f.sig() == GT || f.sig() == LT)) {}
    else if(is_logical(f.sig()) || is_predicate(f.sig())
      || ((f.sig() == MIN || f.sig() == MAX) && is_boolean(t1) && is_boolean(t2)))
    {
      t0 = toplevel ? ternarize_constant(F::make_z(1)) : introduce_bool_var();
    }
    else {
      t0 = introduce_int_var();
      if(toplevel) {
        ternarize_binary(F::make_binary(t0, NEQ, F::make_z(0)), true);
      }
    }
    t1 = ternarize(f.seq(0));
    t1 = booleanize(t1, f.sig());
    t2 = ternarize(f.seq(1));
    t2 = booleanize(t2, f.sig());
    switch(f.sig()) {
      case AND:
      case MIN: return push_ternary(t0, t1, MIN, t2);
      case OR:
      case MAX: return push_ternary(t0, t1, MAX, t2);
      case EQ:
      case EQUIV: return push_ternary(t0, t1, EQ, t2);
      // x xor y ~~> t1 = not t2 /\ t1 = (x = y)
      case NEQ:
      case XOR: {
        if(toplevel) {
          return push_ternary(ternarize_constant(F::make_z(0)), t1, EQ, t2);
        }
        push_ternary(ternarize(F::make_unary(NOT, t0)), t1, EQ, t2);
        return t0;
      }
      case IMPLY: {
        t1 = ternarize(F::make_unary(NOT, t1));
        if(toplevel) {
          return push_ternary(ternarize_constant(F::make_z(1)), t1, MAX, t2);
        }
        return push_ternary(t0, t1, MAX, t2);
      }
      case LEQ: return push_ternary(t0, t1, LEQ, t2);
      // x >= y ~~> y <= x
      case GEQ: return push_ternary(t0, t2, LEQ, t1);
      // x > y ~~> !(x <= y)
      case GT: {
        if(toplevel) {
          return push_ternary(ternarize_constant(F::make_z(0)), t1, LEQ, t2);
        }
        push_ternary(ternarize(F::make_unary(NOT, t0)), t1, LEQ, t2);
        return t0;
      }
      // x < y ~~> y > x ~~> !(y <= x)
      case LT: {
        if(toplevel) {
          return push_ternary(ternarize_constant(F::make_z(0)), t2, LEQ, t1);
        }
        push_ternary(ternarize(F::make_unary(NOT, t0)), t2, LEQ, t1);
        return t0;
      }
      default: {
        return push_ternary(t0, t1, f.sig(), t2);
      }
    }
  }

  std::pair<F, F> binarize_middle(const F& f) {
    assert(f.is(F::Seq) && f.seq().size() > 2);
    battery::vector<F, allocator_type> left;
    battery::vector<F, allocator_type> right;
    int i;
    for(i = 0; i < f.seq().size() / 2; ++i) {
      left.push_back(f.seq(i));
    }
    for(; i < f.seq().size(); ++i) {
      right.push_back(f.seq(i));
    }
    return {
      ternarize(left.size() == 1 ? left.back() : F::make_nary(f.sig(), std::move(left))),
      ternarize(right.size() == 1 ? right.back() : F::make_nary(f.sig(), std::move(right)))};
  }

  F ternarize_nary(const F& f, bool toplevel) {
    if(is_associative(f.sig())) {
      auto [t1, t2] = binarize_middle(f);
      return ternarize(F::make_binary(t1, f.sig(), t2), toplevel);
    }
    else {
      auto tmp = ternarize(F::make_binary(f.seq(0), f.sig(), f.seq(1)));
      for (int i = 2; i < f.seq().size() - 1; i++) {
        tmp = ternarize(F::make_binary(tmp, f.sig(), f.seq(i)));
      }
      return ternarize(F::make_binary(tmp, f.sig(), f.seq(f.seq().size() - 1)), true);
    }
  }

  F ternarize(const F& f, bool toplevel = false) {
    if (f.is_variable()) {
      if(toplevel) {
        return ternarize(F::make_binary(f, NEQ, F::make_z(0)), true);
      }
      return f;
    }
    else if (f.is(F::Z) || f.is(F::B)) {
      if(toplevel) {
        return f.z() != 0 ? F::make_true() : F::make_false();
      }
      return ternarize_constant(f);
    }
    else if (f.is_unary()) {
      auto t = ternarize_unary(f);
      if(toplevel) {
        return ternarize(F::make_binary(t, NEQ, F::make_z(0)), true);
      }
      return t;
    }
    else if (f.is_binary()) {
      return ternarize_binary(f, toplevel);
    }
    else if (f.is(F::Seq) && f.seq().size() > 2) {
      return ternarize_nary(f, toplevel);
    }
    return f;
  }

public:
  void compute(const F& f) {
    if (f.is(F::Seq) && f.sig() == AND) {
      auto seq = f.seq();
      for (int i = 0; i < seq.size(); ++i) {
        compute(f.seq(i));
      }
    }
    else if(f.is(F::E)) {
      existentials.push_back(f);
      name2exists[std::string(battery::get<0>(f.exists()).data())] = existentials.size() - 1;
    }
    else if(!is_ternary_form(f)) {
      ternarize(f, true);
    }
    // Either it is unary, or already in ternary form.
    else {
      conjunction.push_back(f);
    }
  }

  F create() && {
    auto ternarized_formula = std::move(existentials);
    for (int i = 0; i < conjunction.size(); ++i) {
      ternarized_formula.push_back(std::move(conjunction[i]));
    }
    return F::make_nary(AND, std::move(ternarized_formula), UNTYPED, false);
  }
};

} // namespace impl

/**
 * Given a formula `f`, we transform it into a conjunction of formulas of this form:
 * 1. `x <op> c` where `c` is a constant.
 * 2. `x = (y <op> z)` where `<op>` is a binary operator, either arithmetic or a comparison (`=`, `<=`).
 * This ternary form is used by the lala-pc/PIR solver.
 */
template <class F>
F ternarize(const F& f) {
  impl::Ternarizer<F> ternarizer;
  ternarizer.compute(f);
  return std::move(ternarizer).create();
}

} // namespace lala

#endif // LALA_CORE_TERNARIZE_HPP