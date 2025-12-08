#ifndef LALA_CORE_TERNARIZE_HPP
#define LALA_CORE_TERNARIZE_HPP

#include "ast.hpp"
#include "env.hpp"

namespace lala {

template <class F>
bool is_constant_var(const F& x) {
  if(x.is(F::LV)) {
    std::string varname(x.lv().data());
    return varname.starts_with("__CONSTANT_");
  }
  return false;
}

template <class F>
int value_of_constant(const F& x) {
  assert(is_constant_var(x));
  std::string varname(x.lv().data());
  varname = varname.substr(11);
  varname[0] = varname[0] == 'm' ? '-' : varname[0];
  return std::stoi(varname);
}

template <class F>
CUDA bool is_tnf(const F& f) {
  return f.is_binary() && f.seq(0).is_variable() &&
    (f.sig() == EQ || f.sig() == EQUIV) &&
    f.seq(1).is_binary() && f.seq(1).seq(0).is_variable() &&
    f.seq(1).seq(1).is_variable();
}

namespace impl {

template <class F, class Env>
class Ternarizer
{
public:
  using allocator_type = battery::standard_allocator;

  /** A constraint is in extended ternary form if it is either unary (without NEQ, IN) or of the form `x = (y <op> z)`. */
  static bool is_extended_ternary_form(const F& f) {
    int vars = num_vars(f);
    return (vars == 1 && (f.is(F::E) || (f.is_binary() && f.sig() != NEQ && f.sig() != IN && (f.seq(0).is_variable() || f.seq(1).is_variable()) && (f.seq(0).is_constant() || f.seq(1).is_constant())))) ||
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
    else if(varname.starts_with("__VAR_R_")) { // added by Yi-Nung
      introduced_real_vars = std::max((unsigned int)std::stoul(varname.substr(8))+1, introduced_real_vars);
    }
  }

public:
  /** The environment is helpful to recover the sort of the free variables. */
  Ternarizer(const Env& env):
    introduced_int_vars(0),
    introduced_bool_vars(0),
    introduced_real_vars(0), // added by Yi-Nung
    introduced_constants(0),
    env(env)
  {
    /** We skip all the temporary variables already created in the environment. */
    for(int i = 0; i < env.num_vars(); ++i) {
      introduce_existing_var(std::string(env[i].name.data()));
    }
  }

private:
  const Env& env;
  battery::vector<F, allocator_type> conjunction;
  battery::vector<F, allocator_type> existentials;
  std::unordered_map<std::string, int> name2exists;
  unsigned int introduced_int_vars;
  unsigned int introduced_bool_vars;
  unsigned int introduced_real_vars; // added by Yi-Nung
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
    else if(sort.is_real()) { introduced_real_vars++; } // added by Yi-Nung
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

  // added by Yi-Nung
  F introduce_real_var() {
    std::string name = "__VAR_R_" + std::to_string(introduced_real_vars);
    return introduce_var(name, Sort<allocator_type>(Sort<allocator_type>::Real), false);
  }

public:
  F ternarize_constant(const F& f) {
    assert(f.is(F::Z) || f.is(F::B) || f.is(F::R));
    if (f.is(F::Z) || f.is(F::B)) {
      auto index = f.to_z(); 
      std::string name = "__CONSTANT_" + (index < 0 ? std::string("m") : std::string("")) + std::to_string(std::abs(index));
      // if the constant is already a logical variable, we return it.
      if (name2exists.contains(name) || env.contains(name.data())) {
        return F::make_lvar(UNTYPED, LVar<allocator_type>(name.data()));
      }
      auto var = introduce_var(name, f.sort().value(), true);
      conjunction.push_back(F::make_binary(var, EQ, f));
      return var;
    }
    else {
      // added by Yi-Nung
      auto index = f.r();
      double lb = std::get<0>(index);
      double ub = std::get<1>(index);
      std::cout << "lb = " << lb << " ub = " << ub << std::endl;
      std::cout << "abs(lb) = " << std::abs(lb) << " abs(ub) = " << std::abs(ub) << std::endl;
      std::string name = "__CONSTANT_" + (lb < 0 ? std::string("m") : std::string("")) + std::to_string(std::abs(lb)) + "a" + (ub < 0 ? std::string("m") : std::string("")) + std::to_string(std::abs(ub));
      std::cout << "name = " << name << std::endl;
      // if the constant is already a logical variable, we return it.
      if (name2exists.contains(name) || env.contains(name.data())) {
        return F::make_lvar(UNTYPED, LVar<allocator_type>(name.data()));
      }
      auto var = introduce_var(name, f.sort().value(), true);
      conjunction.push_back(F::make_binary(var, EQ, f));
      return var;
    }
    
  }

private:
  /** Create a unary formula if the ternary formula can be simplified. */
  bool simplify_to_unary(F x, F y, Sig sig, F z) {
    /** Unary constraint of the form `1 <=> x <= 5`, `0 <=> x <= 5` or `1 <=> x == 5`   */
    if(is_constant_var(x) && (is_constant_var(y) || is_constant_var(z)) &&
      (sig == LEQ || (sig == EQ && value_of_constant(x) == 1)))
    {
      auto yv = is_constant_var(y) ? F::make_z(value_of_constant(y)) : y;
      auto zv = is_constant_var(z) ? F::make_z(value_of_constant(z)) : z;
      if(value_of_constant(x) == 0) {
        conjunction.push_back(F::make_binary(yv, GT, zv));
      }
      else {
        conjunction.push_back(F::make_binary(yv, sig, zv));
      }
      return true;
    }
    return false;
  }

  /** Create the ternary formula `x = y <sig> z`. */
  F push_ternary(F x, F y, Sig sig, F z) {
    if(simplify_to_unary(x,y,sig,z)) {
      return x;
    }
    if(sig == SUB) {
      /** We simplify x = y - z into y = x + z. */
      conjunction.push_back(F::make_binary(y, EQ, F::make_binary(x, ADD, z)));
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
      /** |x| ~~> t1 = 0 - x /\ t2 = max(x, t1) /\ t2 >= 0 */
      case ABS: {
        F t1 = ternarize(F::make_unary(NEG, x));
        F t2 = introduce_int_var();
        compute(F::make_binary(t2, EQ, F::make_binary(x, MAX, t1)));
        compute(F::make_binary(t2, GEQ, F::make_z(0)));
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
        printf("%% Detected during ternarization (during preprocessing): Unary operator %s not supported\n", string_of_sig(f.sig()));
        printf("%% In formula: "); f.print(); printf("\n");
        exit(EXIT_FAILURE);
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

  // added by Yi-Nung, i thought it will be used later, but not sure.
  bool is_int(const F& f) {
    assert(f.is(F::LV));
    std::string varname(f.lv().data());
    if(name2exists.contains(varname)) {
      return battery::get<1>(existentials[name2exists[varname]].exists()).is_int();
    }
    else {
      auto var_opt = env.variable_of(varname.data());
      if(var_opt.has_value()) {
        return var_opt->get().sort.is_int();
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
    if(f.sig() == IN && f.is_binary() && f.seq(0).is_variable() && f.seq(1).is(F::S)) {
      if(toplevel) {
        compute(decompose_in_constraint(f));
        /** The decomposition of x in S does not capture the approximation x >= min(S) /\ x <= max(S).
         * Therefore, we still give this unary constraint to the interval store in order to over-approximate it.
         * We avoid adding the over-approximation if the decomposition is also a unary constraint. */
        if(f.seq(1).s().size() > 1) {
          conjunction.push_back(f);
        }
        return F::make_true(); /* unused anyways. */
      }
      else {
        return ternarize(decompose_in_constraint(f), toplevel);
      }
    }
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
      /** If a IN constraint appears on the right side, we decompose it here and immediately call ternarize again. */
      if(f.seq(right).sig() == IN) {
        return ternarize(F::make_binary(f.seq(left), EQ, decompose_in_constraint(f.seq(right))), toplevel);
      }
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
      // marked by Yi-Nung, just for testing nnv project.
      // else {
      //   t0 = toplevel ? ternarize_constant(F::make_z(1)) : introduce_int_var();
      // }
      else { // added by Yi-Nung
        t0 = toplevel ? ternarize_constant(F::make_z(1)) : introduce_real_var();
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
    else if (f.is(F::R)) {
      // added by Yi-Nung
      if (toplevel){
        auto fitv = f.to_r();
        return std::get<0>(fitv) != 0 && std::get<1>(fitv) != 0 ? F::make_true() : F::make_false();
      }
      return ternarize_constant(f);
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
    printf("%% Detected during ternarization (during preprocessing): unsupported formula.");
    f.print();
    printf("\n");
    exit(EXIT_FAILURE);
  }

public:
  void compute(const F& f) {
    if (f.is(F::Seq) && f.sig() == AND) {
      const auto& seq = f.seq();
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
       * This is not perfect, because it requires existential quantifier to be in the beginning of the formula (before we introduce any variable).
       * For more robustness, we should rename the variables in the formula, if introduced_int_vars >= X in __VAR_Z_X.
       */
      introduce_existing_var(varname);
    }
    else if(!f.is(F::ESeq) && !is_extended_ternary_form(f)) {
      ternarize(f, true);
    }
    // Either it is unary, an extended formula or already in ternary form.
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
F ternarize(const F& f, const Env& env = Env(), const std::vector<int>& constants = {}) {
  impl::Ternarizer<F, Env> ternarizer(env);
  for(int c : constants) {
    ternarizer.ternarize_constant(F::make_z(c));
  }
  ternarizer.compute(f);
  return std::move(ternarizer).create();
}

} // namespace lala

#endif // LALA_CORE_TERNARIZE_HPP