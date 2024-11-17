#ifndef LALA_CORE_TERNARIZE_HPP
#define LALA_CORE_TERNARIZE_HPP

#include "ast.hpp"
#include "env.hpp"

namespace lala {
namespace impl {

template <class F, class Env>
class Ternarizer
{
public:
  using allocator_type = battery::standard_allocator;

  /** A constraint in ternary form is either unary or of the form `x = (y <op> z)`. */
  static bool is_ternary_form(const F& f, bool ternarize_all = false) {
    int vars = num_vars(f);
    return (!ternarize_all && vars == 1) ||
      (vars == 3 && f.is_binary() && f.sig() == EQ &&
         ((f.seq(0).is_variable() && f.seq(1).is_binary() && is_ternary_op(f.seq(1).sig()) && f.seq(1).seq(0).is_variable() && f.seq(1).seq(1).is_variable())
       || (f.seq(1).is_variable() && f.seq(0).is_binary() && is_ternary_op(f.seq(0).sig()) && f.seq(0).seq(0).is_variable() && f.seq(0).seq(1).is_variable())));
  }

  static bool is_ternary_op(Sig sig) {
    return sig == MAX || sig == MIN || sig == EQ || sig == LEQ || (!is_logical(sig) && !is_predicate(sig));
  }

private:
  void introduce_existing_var(const std::string& varname) {
    if(varname.starts_with("__VAR_Z_")) {
      introduced_int_vars = std::max((unsigned int)std::stoul(varname.substr(8))+1, introduced_int_vars);
    }
    else if(varname.starts_with("__VAR_B_")) {
      introduced_bool_vars = std::max((unsigned int)std::stoul(varname.substr(8))+1, introduced_bool_vars);
    }
  }

public:
  /** The environment is helpful to recover the sort of the free variables. */
  Ternarizer(const Env& env, bool ternarize_all):
    introduced_int_vars(0),
    introduced_bool_vars(0),
    introduced_constants(0),
    env(env),
    ternarize_all(ternarize_all)
  {
    /** We skip all the temporary variables already created in the environment. */
    for(int i = 0; i < env.num_vars(); ++i) {
      introduce_existing_var(std::string(env[i].name.data()));
    }
  }

private:
  const Env& env;
  bool ternarize_all;
  battery::vector<F, allocator_type> conjunction;
  battery::vector<F, allocator_type> existentials;
  std::unordered_map<std::string, int> name2exists;
  unsigned int introduced_int_vars;
  unsigned int introduced_bool_vars;
  unsigned int introduced_constants;

  F introduce_var(const std::string& name, auto sort, bool constant) {
    auto var_name = LVar<allocator_type>(name.data());
    assert(!env.contains(name.data()));
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

  F ternarize_constant(const F& f) {
    assert(f.is(F::Z) || f.is(F::B));
    auto index = f.to_z();
    std::string name = "__CONSTANT_" + (index < 0 ? std::string("m") : std::string("")) + std::to_string(abs(index));
    // if the constant is already a logical variable, we return it.
    if (name2exists.contains(name) || env.contains(name.data())) {
      return F::make_lvar(UNTYPED, LVar<allocator_type>(name.data()));
    }
    auto var = introduce_var(name, f.sort().value(), true);
    conjunction.push_back(F::make_binary(var, EQ, f));
    return var;
  }

  bool is_constant_var(const F& x) const {
    if(x.is(F::LV)) {
      std::string varname(x.lv().data());
      return varname.starts_with("__CONSTANT_");
    }
    return false;
  }

  int value_of_constant(const F& x) const {
    assert(is_constant_var(x));
    std::string varname(x.lv().data());
    varname = varname.substr(11);
    varname[0] = varname[0] == 'm' ? '-' : varname[0];
    return std::stoi(varname);
  }

  /** We try to simplify the ternary constraint into a unary constraint in case of constant values. */
  bool try_simplify_push_ternary(const F& x, const F& y, Sig sig, const F& z) {
    /** We don't simplify if we need to ternarize everything. */
    if(ternarize_all) {
      return false;
    }
    /** We first seek to simply the ternary constraint in case of two constants. */
    int xc = is_constant_var(x);
    int yc = is_constant_var(y);
    int zc = is_constant_var(z);
    if(yc + zc == 2) {
      F y_sig_z = F::make_binary(F::make_z(value_of_constant(y)), sig, F::make_z(value_of_constant(z)));
      F simplified = F::make_binary(x, EQ, eval(y_sig_z));
      if(is_ternary_form(simplified)) {
        compute(simplified);
        return true;
      }
    }
    else if(xc + yc + zc == 2) {
      assert(xc == 1);
      F y_sig_z =
        (yc == 1)
        ? F::make_binary(F::make_z(value_of_constant(y)), sig, z)
        : F::make_binary(y, sig, F::make_z(value_of_constant(z)));
      int x_value = value_of_constant(x);
      if(x_value == 0) {
        auto r = negate(y_sig_z);
        F not_y_sig_z = r.has_value() ? *r : F::make_unary(NOT, y_sig_z);
        not_y_sig_z = eval(not_y_sig_z);
        if(is_ternary_form(not_y_sig_z)) {
          compute(not_y_sig_z);
          return true;
        }
      }
      else if(is_ternary_form(y_sig_z)) {
        compute(y_sig_z);
        return true;
      }
    }
    return false;
  }

  /** Create the ternary formula `x = y <sig> z`. */
  F push_ternary(const F& x, const F& y, Sig sig, const F& z) {
    if(try_simplify_push_ternary(x, y, sig, z)) {
      return x;
    }
    /** If the simplification was not possible, we add the ternary constraint. */
    conjunction.push_back(F::make_binary(x, EQ, F::make_binary(y, sig, z)));
    return x;
  }

  F ternarize_unary(const F& f, bool toplevel = false) {
    F x = ternarize(f.seq(0));
    switch(f.sig()) {
      /** -x ~~> t = 0 - x */
      case NEG: {
        F t = push_ternary(introduce_int_var(), ternarize(F::make_z(0)), SUB, x);
        if(toplevel) {
          return ternarize(F::make_binary(t, NEQ, F::make_z(0)), true);
        }
        return t;
      }
      /** |x| ~~> t1 = 0 - x /\ t2 <= max(x, t1) /\ t2 >= 0 /\ t2 >= x /\ t2 >= t1 */
      case ABS: {
        F t1 = ternarize(F::make_unary(NEG, x));
        F t2 = introduce_int_var();
        compute(F::make_binary(t2, GEQ, F::make_z(0)));
        compute(F::make_binary(t2, GEQ, t1));
        compute(F::make_binary(t2, GEQ, x));
        compute(F::make_binary(t2, LEQ, F::make_binary(x, MAX, t1)));
        if(toplevel) {
          return ternarize(F::make_binary(t2, NEQ, F::make_z(0)), true);
        }
        return t2;
      }
      /** NOT x ~~> ternarize(x = 0) ~~> t = (x = 0) */
      case NOT: return ternarize(F::make_binary(x, EQ, F::make_z(0)), toplevel);
      case MINIMIZE:
      case MAXIMIZE: {
        conjunction.push_back(F::make_unary(f.sig(), x));
        return x;
      }
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
    if(name2exists.contains(varname)) {
      return battery::get<1>(existentials[name2exists[varname]].exists()).is_bool();
    }
    else {
      auto var_opt = env.variable_of(varname.data());
      if(var_opt.has_value()) {
        return var_opt->get().sort.is_bool();
      }
    }
    assert(false); // undeclared variable.
    return false;
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
    bool almost_ternary = false;
    /** We first handle "almost ternarized" constraint.
     * If the symbol is already an equality with a variable on one side, we only need to ternarize the other half.
     * We set t0 to be the variable and proceeds. */
    if((((f.seq(0).is_variable() || f.seq(0).is_constant()) && f.seq(1).is_binary())
     ||((f.seq(1).is_variable() || f.seq(1).is_constant()) && f.seq(0).is_binary()))
     && (f.sig() == EQUIV || f.sig() == EQ))
    {
      int left = f.seq(0).is_binary();
      int right = f.seq(1).is_binary();
      t0 = ternarize(f.seq(left));
      f = f.seq(right);
      toplevel = false;
      almost_ternary = true;
    }
    t1 = ternarize(f.seq(0));
    t1 = booleanize(t1, f.sig());
    t2 = ternarize(f.seq(1));
    t2 = booleanize(t2, f.sig());
    if(!almost_ternary) {
    /** We don't need to create t0 for these formulas at toplevel. */
      if(toplevel && (f.sig() == NEQ || f.sig() == XOR || f.sig() == IMPLY || f.sig() == GT || f.sig() == LT)) {}
      else if(is_logical(f.sig()) || is_predicate(f.sig())
        || ((f.sig() == MIN || f.sig() == MAX) && is_boolean(t1) && is_boolean(t2)))
      {
        t0 = toplevel ? ternarize_constant(F::make_z(1)) : introduce_bool_var();
      }
      else {
        t0 = toplevel ? ternarize_constant(F::make_z(1)) : introduce_int_var();
      }
    }
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
      case IMPLY: return ternarize(F::make_binary(F::make_unary(NOT, t1), OR, t2), toplevel);
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
      F tmp = ternarize(F::make_binary(f.seq(0), f.sig(), f.seq(1)));
      for (int i = 2; i < f.seq().size() - 1; i++) {
        tmp = ternarize(F::make_binary(tmp, f.sig(), f.seq(i)));
      }
      tmp = ternarize(F::make_binary(tmp, f.sig(), f.seq(f.seq().size() - 1)), toplevel);
      return tmp;
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
        return f.to_z() != 0 ? F::make_true() : F::make_false();
      }
      return ternarize_constant(f);
    }
    else if (f.is(F::S)) {
      return f;
    }
    else if (f.is_unary()) {
      return ternarize_unary(f, toplevel);
    }
    else if (f.is_binary()) {
      return ternarize_binary(f, toplevel);
    }
    else if (f.is(F::Seq) && f.seq().size() > 2) {
      return ternarize_nary(f, toplevel);
    }
    printf("Unsupported formula: "); f.print(false); printf("\n");
    assert(false);
    return F::make_false();
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
      std::string varname(battery::get<0>(f.exists()).data());
      name2exists[varname] = existentials.size() - 1;
      /** If ternarize has been called before, some temporary variables __VAR_Z_* and __VAR_B_* might already have been created.
       * In that case, we need to update the counters to avoid conflicts.
       * This is not perfect, because it requires existential quantifier are in the beginning of the formula (before we introduce any variable).
       * For more robustness, we should rename the variables in the formula, if introduced_int_vars >= X in __VAR_Z_X.
       */
      introduce_existing_var(varname);
    }
    else if(!f.is(F::ESeq) && !is_ternary_form(f, ternarize_all)) {
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
    if(ternarized_formula.size() == 1) {
      return std::move(ternarized_formula[0]);
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
template <class F, class Env = VarEnv<battery::standard_allocator>>
F ternarize(const F& f, const Env& env = Env(), bool ternarize_all = false) {
  impl::Ternarizer<F, Env> ternarizer(env, ternarize_all);
  ternarizer.compute(f);
  return std::move(ternarizer).create();
}

} // namespace lala

#endif // LALA_CORE_TERNARIZE_HPP