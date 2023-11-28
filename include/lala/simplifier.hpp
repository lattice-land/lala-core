// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_SIMPLIFIER_HPP
#define LALA_CORE_SIMPLIFIER_HPP

#include "logic/logic.hpp"
#include "universes/primitive_upset.hpp"
#include "abstract_deps.hpp"
#include "battery/dynamic_bitset.hpp"

namespace lala {

/** This abstract domain works at the level of logical formulas.
 * It refines the formula by performing a number of simplifications w.r.t. an underlying abstract domain including:
 *  1. Removing assigned variables.
 *  2. Removing unused variables.
 *  3. Removing entailed formulas.
 *  4. Removing variable equality by tracking equivalence classes.
 *
 * The simplified formula can be obtained by calling `deinterpret()`.
 * Given a solution to the simplified formula, the extended model (with the variables deleted) can be obtained by calling `representative()` to obtain the representative variable of each equivalence class.
 */
template<class A, class Allocator>
class Simplifier {
public:
  using allocator_type = Allocator;
  using sub_type = A;
  using sub_allocator_type = typename sub_type::allocator_type;
  using universe_type = typename sub_type::universe_type;
  using memory_type = typename universe_type::memory_type;
  using this_type = Simplifier<sub_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const char* name = "Simplifier";

  template<class A2, class Alloc2>
  friend class Simplifier;

  using formula_sequence = battery::vector<TFormula<allocator_type>, allocator_type>;

private:
  AType atype;
  abstract_ptr<sub_type> sub;
  // We keep a copy of the variable environment in which the formula has been initially interpreted.
  // This is necessary to project the variables and ask constraints in the subdomain during refinement.
  VarEnv<allocator_type> env;
  // Read-only conjunctive formula, where each is treated independently.
  formula_sequence formulas;
  // Write-only (accessed in only 1 thread because this is not a parallel lattice entity) conjunctive formula, the main operation is a map between formulas and simplified_formulas.
  formula_sequence simplified_formulas;
  // eliminated_variables[i] is `true` when the variable `i` can be removed because it is assigned to a constant.
  battery::dynamic_bitset<memory_type, allocator_type> eliminated_variables;
  // eliminated_formulas[i] is `true` when the formula `i` is entailed.
  battery::dynamic_bitset<memory_type, allocator_type> eliminated_formulas;
  // `equivalence_classes[i]` contains the index of the representative variable in the equivalence class of the variable `i`.
  battery::vector<ZDec<int, memory_type>, allocator_type> equivalence_classes;
  // `constants[i]` contains the universe value of the representative variables `i`, aggregated by join on the values of all variables in the equivalence class.
  battery::vector<universe_type, allocator_type> constants;

public:
  CUDA Simplifier(AType atype
    , abstract_ptr<sub_type> sub
    , const allocator_type& alloc = allocator_type())
   : atype(atype), sub(sub), env(alloc)
   , formulas(alloc), simplified_formulas(alloc)
   , eliminated_variables(alloc), eliminated_formulas(alloc)
   , equivalence_classes(alloc), constants(alloc)
  {}

  CUDA Simplifier(this_type&& other)
    : atype(other.atype), sub(std::move(other.sub)), env(other.env)
    , formulas(std::move(other.formulas)), simplified_formulas(std::move(other.simplified_formulas))
    , eliminated_variables(std::move(other.eliminated_variables)), eliminated_formulas(std::move(other.eliminated_formulas))
    , equivalence_classes(std::move(other.equivalence_classes)), constants(std::move(other.constants))
  {}

  struct light_copy_tag {};

  // This return a light copy of `other`, basically just keeping the equivalence classes and the environment to be able to print solutions and call `representative`.
  template<class A2, class Alloc2>
  CUDA Simplifier(const Simplifier<A2, Alloc2>& other, light_copy_tag tag, abstract_ptr<sub_type> sub, const allocator_type& alloc = allocator_type())
   : atype(other.atype)
   , sub(sub)
   , env(other.env, alloc)
   , equivalence_classes(other.equivalence_classes, alloc)
   , constants(other.constants, alloc)
  {}

  CUDA allocator_type get_allocator() const {
    return formulas.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA size_t vars() const {
    return equivalence_classes.size();
  }

  struct tell_type {
    int num_vars;
    formula_sequence formulas;
    VarEnv<allocator_type> env;
    tell_type(const allocator_type& alloc = allocator_type())
      : num_vars(0), formulas(alloc), env(alloc)
    {}
  };

  template<class F> using iresult = IResult<tell_type, F>;
  template<class F> using iresult_tell = iresult<F>;

private:
  template <class F, class Env>
  CUDA NI void interpret_in_impl(const F& f, Env& env, iresult<F>& res) const {
    if(res.is_error()) {
      return;
    }
    if(f.is(F::E)) {
      auto avar = env.interpret(f.map_atype(aty()));
      if(avar.has_value()) {
        res.value().num_vars++;
      }
      else {
        res = std::move(avar).template map_error<tell_type>();
      }
    }
    else if(f.is(F::Seq) && f.sig() == AND) {
      const auto& seq = f.seq();
      for(int i = 0; i < seq.size(); ++i) {
        interpret_in_impl(seq[i], env, res);
      }
    }
    else {
      res.value().formulas.push_back(f);
    }
  }

public:
  template <class F, class Env>
  CUDA NI iresult_tell<F> interpret_tell_in(const F& f, Env& env) const {
    auto snap = env.snapshot();
    iresult_tell<F> res{get_allocator()};
    interpret_in_impl(f, env, res);
    if(!res.has_value()) {
      env.restore(snap);
    }
    else {
      res.value().env = env;
    }
    return std::move(res);
  }

  CUDA this_type& tell(tell_type&& t) {
    env = std::move(t.env);
    eliminated_variables.resize(t.num_vars);
    eliminated_formulas.resize(t.formulas.size());
    constants.resize(t.num_vars);
    equivalence_classes.resize(t.num_vars);
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      equivalence_classes[i].tell(local::ZDec(i));
    }
    formulas = std::move(t.formulas);
    simplified_formulas.resize(formulas.size());
    return *this;
  }

private:
  // Return the abstract variable of the subdomain from the abstract variable `x` of this domain.
  // In the environment, all variables should have been interpreted by the sub-domain, and we assume avars[0] contains the sub abstract variable.
  CUDA AVar to_sub_var(AVar x) const {
    return env[x].avars[0];
  }

  CUDA AVar to_sub_var(int vid) const {
    return to_sub_var(AVar{aty(), vid});
  }

  // `f` must be a formula from `formulas`.
  CUDA AVar var_of(const TFormula<allocator_type>& f) const {
    using F = TFormula<allocator_type>;
    if(f.is(F::LV)) {
      assert(env.variable_of(f.lv()).has_value());
      assert(env.variable_of(f.lv())->avar_of(aty()).has_value());
      return env.variable_of(f.lv())->avar_of(aty()).value();
    }
    else {
      assert(f.is(F::V));
      assert(env[f.v()].avar_of(aty()).has_value());
      return env[f.v()].avar_of(aty()).value();
    }
  }

public:
  /** Return the representative variable of the equivalence class of the variable with the name `vname`. */
  CUDA const LVar<allocator_type>& representative(const LVar<allocator_type>& vname) const {
    int rep = equivalence_classes[env.variable_of(vname)->avar_of(aty())->vid()];
    return env.name_of(AVar{aty(), rep});
  }

  /** Project the abstract universe of `vname` taking into account simplifications (representative variable and constant). */
  template <class B, class Env>
  CUDA const universe_type& project(const LVar<allocator_type>& vname, const Env& benv, const B& b) const {
    int rep = equivalence_classes[env.variable_of(vname)->avar_of(aty())->vid()];
    const auto& rep_name = env.name_of(AVar{aty(), rep});
    const auto& variable = benv.variable_of(rep_name);
    if(variable.has_value()) {
      return b.project(variable->avars[0]);
    }
    else {
      return constants[rep];
    }
  }

  template <class Env>
  CUDA const Sort<allocator_type>& sort_of(const LVar<allocator_type>& vname, const Env& benv) const {
    const auto& variable = benv.variable_of(representative(vname));
    if(variable.has_value()) {
      return variable->sort;
    }
    else {
      return env.variable_of(vname)->sort;
    }
  }

private:
  template <class Mem>
  CUDA void eliminate(battery::dynamic_bitset<memory_type, allocator_type>& mask, size_t i, BInc<Mem>& has_changed) {
    if(!mask.test(i)) {
      mask.set(i, true);
      has_changed.tell_top();
    }
  }

  // We eliminate the representative of the variable `i` if it is a singleton.
  template <class Mem>
  CUDA void vrefine(size_t i, BInc<Mem>& has_changed) {
    const auto& u = sub->project(to_sub_var(i));
    size_t j = equivalence_classes[i];
    constants[j].tell(u, has_changed);
    if(constants[j].lb().value() == constants[j].ub().value()) {
      eliminate(eliminated_variables, j, has_changed);
    }
  }

  template <class Mem>
  CUDA void crefine(size_t i, BInc<Mem>& has_changed) {
    using F = TFormula<allocator_type>;
    // Eliminate constraint of the form x = y, and add x,y in the same equivalence class.
    if(is_var_equality(formulas[i])) {
      AVar x = var_of(formulas[i].seq(0));
      AVar y = var_of(formulas[i].seq(1));
      equivalence_classes[x.vid()].tell(local::ZDec(equivalence_classes[y.vid()]), has_changed);
      equivalence_classes[y.vid()].tell(local::ZDec(equivalence_classes[x.vid()]), has_changed);
      eliminate(eliminated_formulas, i, has_changed);
    }
    else {
      // Eliminate entailed formulas.
      auto r = sub->interpret_ask_in(formulas[i], env);
      if(r.has_value()) {
        if(sub->ask(r.value())) {
          eliminate(eliminated_formulas, i, has_changed);
          return;
        }
      }
      // Replace assigned variables by constants.
      // Note that since everything is in a fixed point loop, both the constant and the equivalence class might be updated later on.
      // This is one of the reasons we cannot update `formulas` in-place: we would not be able to update the constant a second time (since the variable would be eliminated).
      auto f = formulas[i].map([&](const F& f) {
        if(f.is_variable()) {
          AVar x = var_of(f);
          if(eliminated_variables.test(x.vid())) {
            return constants[x.vid()].template deinterpret<F>();
          }
          else if(equivalence_classes[x.vid()] != x.vid()) {
            return F::make_lvar(UNTYPED, env.name_of(AVar{aty(), equivalence_classes[x.vid()]}));
          }
          return f.map_atype(UNTYPED);
        }
        return f;
      });
      if(f != simplified_formulas[i]) {
        simplified_formulas[i] = f;
        has_changed.tell_top();
      }
    }
  }

public:
  /** We have one refinement operator per variable and one per constraint in the interpreted formula. */
  CUDA size_t num_refinements() const {
    return constants.size() + formulas.size();
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    assert(i < num_refinements());
    if(i < constants.size()) {
      vrefine(i, has_changed);
    }
    else {
      crefine(i - constants.size(), has_changed);
    }
  }

  CUDA int num_eliminated_variables() const {
    int keep = 0;
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      if(equivalence_classes[i] == i && !eliminated_variables.test(i)) {
        ++keep;
      }
    }
    return equivalence_classes.size() - keep;
  }

  CUDA int num_eliminated_formulas() const {
    return eliminated_formulas.count();
  }

  CUDA NI TFormula<allocator_type> deinterpret() {
    using F = TFormula<allocator_type>;
    typename F::Sequence seq{get_allocator()};

    // for(int i = 0; i < equivalence_classes.size(); ++i) {
    //   printf("equivalence_classes[%d] = %d (%s)\n", i, equivalence_classes[i].value(), env[AVar(aty(), i)].name.data());
    // }

    // for(int i = 0; i < equivalence_classes.size(); ++i) {
    //   printf("eliminated_variables[%d] = %d\n", i, eliminated_variables.test(i));
    // }

    // for(int i = 0; i < formulas.size(); ++i) {
    //   printf("eliminated_formulas[%d] = %d\n", i, eliminated_formulas.test(i));
    //   printf("before: "); formulas[i].print(); printf("\n");
    //   printf("after: "); simplified_formulas[i].print(); printf("\n");
    // }

    // A representative variable is eliminated if all variables in its equivalence class must be eliminated.
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      eliminated_variables.set(equivalence_classes[i], eliminated_variables.test(equivalence_classes[i]) && eliminated_variables.test(i));
    }

    // for(int i = 0; i < equivalence_classes.size(); ++i) {
    //   printf("eliminated_variables'[%d] = %d\n", i, eliminated_variables.test(i));
    // }

    // Deinterpret the existential quantifiers (only one per equivalence classes), and the domain of each variable.
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      if(equivalence_classes[i] == i && !eliminated_variables.test(i)) {
        const auto& x = env[AVar{aty(), i}];
        seq.push_back(F::make_exists(UNTYPED, x.name, x.sort));
        auto domain_constraint = constants[i].deinterpret(AVar(aty(), i), env);
        map_avar_to_lvar(domain_constraint, env, true);
        seq.push_back(domain_constraint);
      }
    }

    // Deinterpret the simplified formulas.
    for(int i = 0; i < simplified_formulas.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        seq.push_back(simplified_formulas[i]);
      }
    }
    if(seq.size() == 0) {
      return F::make_true();
    }
    else {
      return F::make_nary(AND, std::move(seq));
    }
  }
};

} // namespace lala

#endif
