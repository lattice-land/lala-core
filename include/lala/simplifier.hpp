// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_SIMPLIFIER_HPP
#define LALA_CORE_SIMPLIFIER_HPP

#include "logic/logic.hpp"
#include "universes/arith_bound.hpp"
#include "abstract_deps.hpp"
#include "battery/dynamic_bitset.hpp"

namespace lala {

/** This abstract domain works at the level of logical formulas.
 * It deduces the formula by performing a number of simplifications w.r.t. an underlying abstract domain including:
 *  1. Removing assigned variables.
 *  2. Removing unused variables.
 *  3. Removing entailed formulas.
 *  4. Removing variable equality by tracking equivalence classes.
 *
 * The simplified formula can be obtained by calling `deinterpret()`.
 * Given a solution to the simplified formula, the value of the variables deleted can be obtained by calling `print_variable()`.
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
  constexpr static const bool sequential = universe_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  // Note that I did not define the concretization function formally yet... This is not yet an fully fledged abstract domain, we need to work more on it!
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const char* name = "Simplifier";

  template<class A2, class Alloc2>
  friend class Simplifier;

  using formula_sequence = battery::vector<TFormula<allocator_type>, allocator_type>;

private:
  AType atype;
  AType store_aty;
  abstract_ptr<sub_type> sub;
  // We keep a copy of the variable environment in which the formula has been initially interpreted.
  // This is necessary to project the variables and ask constraints in the subdomain during deduction.
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
  battery::vector<ZUB<int, memory_type>, allocator_type> equivalence_classes;
  // `constants[i]` contains the universe value of the representative variables `i`, aggregated by meet on the values of all variables in the equivalence class.
  battery::vector<universe_type, allocator_type> constants;

public:
  CUDA Simplifier(AType atype
    , AType store_aty
    , abstract_ptr<sub_type> sub
    , const allocator_type& alloc = allocator_type())
   : atype(atype), store_aty(store_aty), sub(sub), env(alloc)
   , formulas(alloc), simplified_formulas(alloc)
   , eliminated_variables(alloc), eliminated_formulas(alloc)
   , equivalence_classes(alloc), constants(alloc)
  {}

  CUDA Simplifier(this_type&& other)
    : atype(other.atype), store_aty(other.store_aty), sub(std::move(other.sub)), env(other.env)
    , formulas(std::move(other.formulas)), simplified_formulas(std::move(other.simplified_formulas))
    , eliminated_variables(std::move(other.eliminated_variables)), eliminated_formulas(std::move(other.eliminated_formulas))
    , equivalence_classes(std::move(other.equivalence_classes)), constants(std::move(other.constants))
  {}

  struct light_copy_tag {};

  // This return a light copy of `other`, basically just keeping the equivalence classes and the environment to be able to print solutions and call `representative`.
  template<class A2, class Alloc2>
  CUDA Simplifier(const Simplifier<A2, Alloc2>& other, light_copy_tag tag, abstract_ptr<sub_type> sub, const allocator_type& alloc = allocator_type())
   : atype(other.atype)
   , store_aty(other.store_aty)
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

  /** @parallel @order-preserving @increasing  */
  CUDA local::B is_bot() const {
    return sub->is_bot();
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA size_t vars() const {
    return equivalence_classes.size();
  }

  template <class Alloc>
  struct tell_type {
    int num_vars;
    formula_sequence formulas;
    VarEnv<Alloc>* env;
    tell_type(const Alloc& alloc = Alloc())
      : num_vars(0), formulas(alloc), env(nullptr)
    {}
  };

public:
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    if(f.is(F::E)) {
      tell.num_vars++;
      tell.env = &env;
    }
    else {
      tell.formulas.push_back(f);
      tell.env = &env;
    }
    return true;
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class Alloc2>
  CUDA bool interpret(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    return interpret_tell<diagnose>(f, env, tell, diagnostics);
  }

  CUDA void initialize(int num_vars, int num_cons) {
    eliminated_variables.resize(num_vars);
    eliminated_variables.reset();
    eliminated_formulas.resize(num_cons);
    eliminated_formulas.reset();
    constants.resize(num_vars);
    for(int i = 0; i < constants.size(); ++i) {
      constants[i].join_top();
    }
    equivalence_classes.resize(num_vars);
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      equivalence_classes[i] = i;
    }
  }

public:
  /** @sequential */
  template <class Alloc2>
  CUDA bool deduce(tell_type<Alloc2>&& t) {
    if(t.env != nullptr) { // could be nullptr if the interpreted formula is true.
      env = *(t.env);
      initialize(t.num_vars, t.formulas.size());
      formulas = std::move(t.formulas);
      simplified_formulas.resize(formulas.size());
      return true;
    }
    return false;
  }

  // `f` must be a formula from `formulas`.
  CUDA AVar var_of(const TFormula<allocator_type>& f) const {
    using F = TFormula<allocator_type>;
    if(f.is(F::LV)) {
      assert(env.variable_of(f.lv()).has_value());
      assert(env.variable_of(f.lv())->get().avar_of(store_aty).has_value());
      return env.variable_of(f.lv())->get().avar_of(store_aty).value();
    }
    else {
      assert(f.is(F::V));
      assert(env[f.v()].avar_of(store_aty).has_value());
      return env[f.v()].avar_of(store_aty).value();
    }
  }

public:
  /** Print the abstract universe of `vname` taking into account simplifications (representative variable and constant).
  */
  template <class Alloc, class Abs, class Env>
  CUDA void print_variable(const LVar<Alloc>& vname, const Env& benv, const Abs& b) const {
    assert(env.variable_of(vname).has_value());
    const auto& local_var = env.variable_of(vname)->get();
    assert(local_var.avar_of(store_aty).has_value());
    int rep = equivalence_classes[local_var.avar_of(store_aty)->vid()];
    const auto& rep_name = env.name_of(AVar{store_aty, rep});
    auto benv_variable = benv.variable_of(rep_name);
    if(benv_variable.has_value()) {
      benv_variable->get().sort.print_value(b.project(benv_variable->get().avars[0]));
    }
    else {
      local_var.sort.print_value(constants[rep]);
    }
  }

private:
  /** \return `true` if mask[i] was changed. */
  CUDA local::B eliminate(battery::dynamic_bitset<memory_type, allocator_type>& mask, size_t i) {
    if(!mask.test(i)) {
      mask.set(i, true);
      return true;
    }
    return false;
  }

  // We eliminate the representative of the variable `i` if it is a singleton.
  CUDA local::B vdeduce(int i) {
    const auto& u = sub->project(AVar{store_aty, i});
    size_t j = equivalence_classes[i];
    local::B has_changed = constants[j].meet(u);
    if(!constants[j].is_bot() && constants[j].lb().value() == constants[j].ub().value()) {
      has_changed |= eliminate(eliminated_variables, j);
    }
    return has_changed;
  }

private:
  CUDA local::B add_equivalence(AVar x, AVar y) {
    local::ZUB min_vid(equivalence_classes[x.vid()]);
    min_vid.meet(equivalence_classes[y.vid()]);
    // We must update representative element of x and y, to avoid disconnecting previous representative elements.
    local::B has_changed = false;
    has_changed |= equivalence_classes[equivalence_classes[x.vid()]].meet(min_vid);
    has_changed |= equivalence_classes[equivalence_classes[y.vid()]].meet(min_vid);
    has_changed |= equivalence_classes[x.vid()].meet(min_vid);
    has_changed |= equivalence_classes[y.vid()].meet(min_vid);
    return has_changed;
  }

public:
  CUDA local::B cons_deduce(int i) {
    using F = TFormula<allocator_type>;
    local::B has_changed = false;
    // Eliminate constraint of the form x = y, and add x,y in the same equivalence class.
    if(is_var_equality(formulas[i])) {
      add_equivalence(var_of(formulas[i].seq(0)), var_of(formulas[i].seq(1)));
      has_changed |= eliminate(eliminated_formulas, i);
      return has_changed;
    }
    else {
      // Eliminate entailed formulas.
      IDiagnostics diagnostics;
      typename sub_type::template ask_type<allocator_type> ask;
#ifdef _MSC_VER // Avoid MSVC compiler bug. See https://stackoverflow.com/questions/77144003/use-of-template-keyword-before-dependent-template-name
      if(sub->interpret_ask(formulas[i], env, ask, diagnostics))
#else
      if(sub->template interpret_ask(formulas[i], env, ask, diagnostics))
#endif
      {
        if(sub->ask(ask)) {
          return eliminate(eliminated_formulas, i);
        }
      }
      // Replace assigned variables by constants.
      // Note that since everything is in a fixed point loop, both the constant and the equivalence class might be updated later on.
      // This is one of the reasons we cannot update `formulas` in-place: we would not be able to update the constant a second time (since the variable would be eliminated).
      auto f = formulas[i].map([&](const F& f, const F& parent) {
        if(f.is_variable()) {
          AVar x = var_of(f);
          if(eliminated_variables.test(x.vid())) {
            auto k = constants[x.vid()].template deinterpret<F>();
            if(env[x].sort.is_bool() && k.is(F::Z) && parent.is_logical()) {
              return k.z() == 0 ? F::make_false() : F::make_true();
            }
            return std::move(k);
          }
          else if(equivalence_classes[x.vid()] != x.vid()) {
            return F::make_lvar(UNTYPED, env.name_of(AVar{store_aty, equivalence_classes[x.vid()]}));
          }
          return f.map_atype(UNTYPED);
        }
        return f;
      });
      f = eval(f);
      if(f.is_true()) {
        return eliminate(eliminated_formulas, i);
      }
      if(f != simplified_formulas[i]) {
        simplified_formulas[i] = f;
        return true;
      }
      return false;
    }
  }

  /** We have one deduction operator per variable and one per constraint in the interpreted formula. */
  CUDA size_t num_deductions() const {
    return constants.size() + formulas.size();
  }

  CUDA local::B deduce(size_t i) {
    assert(i < num_deductions());
    if(i < constants.size()) {
      return vdeduce(i);
    }
    else {
      return cons_deduce(i - constants.size());
    }
  }

  template <class Env>
  CUDA void init_env(const Env& env) {
    this->env = env;
  }

private:
  CUDA void normalize_equivalence_classes() {
    /** Normalize equivalence classes */
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      /** The representative element is always the one with the lowest index, and we know that equivalence_classes[i] <= i. Therefore `equivalence_classes[equivalence_classes[rep]]` must be the representative element. */
      constants[equivalence_classes[equivalence_classes[i]]].meet(constants[equivalence_classes[i]]);
      equivalence_classes[i] = equivalence_classes[equivalence_classes[i]];
    }
  }

  /** I-CSE algorithm.
   * For each pair of TNF constraints `x <=> y op z` and `x' <=> y' op' z'`, whenever `[y'] = [y]`, `op = op'` and `[z] = [z']`, we add the equivalence `x = x'` and eliminate the second constraint.
   * Note that [x] represents the equivalence class of `x`.
   * To avoid an algorithm running in O(n^2), we use a hash map to detect syntactical equivalence between `y op z` and `y' op z'`.
   * Further, for commutative operators, we redefine the equality function.
   *
   * This algorithm is applied until a fixpoint is reached.
   * \return The number of formulas eliminated.
   */
  template <class Seq>
  CUDA int i_cse(const Seq& tnf, size_t& eliminated_constraints, size_t& fixpoint_iterations) {
    auto hash = [](const std::tuple<int,Sig,int> &right_tnf){
      return static_cast<size_t>(std::get<0>(right_tnf))
           * static_cast<size_t>(std::get<1>(right_tnf))
           * static_cast<size_t>(std::get<2>(right_tnf));
    };
    // This equality function also checks for commutative operators (in which case the hash will also be the same).
    auto equal = [](const std::tuple<int,Sig,int> &l, const std::tuple<int,Sig,int> &r){
      if(std::get<1>(l) == std::get<1>(r)) {
        return (std::get<0>(l) == std::get<0>(r) && std::get<2>(l) == std::get<2>(r))
         || (is_commutative(std::get<1>(l)) && std::get<0>(l) == std::get<2>(r) && std::get<2>(l) == std::get<0>(r));
      }
      return false;
    };
    std::unordered_map<std::tuple<int,Sig,int>, int, decltype(hash), decltype(equal)> cs(tnf.size(), hash, equal);
    bool has_changed = true;
    while(has_changed) {
      ++fixpoint_iterations;
      has_changed = false;
      cs.clear();
      for(int i = 0; i < tnf.size(); ++i) {
        if(!eliminated_formulas.test(i)) {
          int x = equivalence_classes[var_of(tnf[i].seq(0)).vid()];
          int y = equivalence_classes[var_of(tnf[i].seq(1).seq(0)).vid()];
          int z = equivalence_classes[var_of(tnf[i].seq(1).seq(1)).vid()];
          Sig op = tnf[i].seq(1).sig();
          auto p = cs.insert(std::make_pair(std::make_tuple(y, op, z), x));
          if(!p.second) { // `p.second` is false if we detect a collision.
            AVar x2 = AVar{store_aty, p.first->second};
            add_equivalence(AVar{store_aty, x}, x2);
            eliminate(eliminated_formulas, i);
            eliminated_constraints++;
            has_changed = true;
          }
        }
      }
      normalize_equivalence_classes();
    }
    return eliminated_constraints;
  }

public:
  /** We simplify the TNF formula by computing equivalence classes and removing useless variables.
   * `tnf` is a conjunction of formulas in TNF, existential quantifiers and unary constraints.
   *
   * precondition: `init_env` must have been called.
  */
  template <class F>
  CUDA F simplify_tnf(const F& tnf_f, size_t& eliminated_equality_constraints, size_t& eliminated_constraints_by_icse, size_t& icse_fixpoint_iterations) {
    assert(tnf_f.is(F::Seq) && tnf_f.sig() == AND);
    const auto& tnf = tnf_f.seq();
    initialize(sub->vars(), tnf.size());
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      constants[equivalence_classes[i]].meet(sub->project(AVar{store_aty, i}));
    }
    /** Compute equivalence classes by detecting equality constraints. */
    /** We also eliminate all the existential quantifier and unary constraints,
     * those will be re-generated from the underlying store later. */
    for(int i = 0; i < tnf.size(); ++i) {
      if(!is_tnf(tnf[i])) {
        eliminate(eliminated_formulas, i);
      }
      /** 1 <=> x = y */
      else if(tnf[i].seq(1).sig() == EQ || tnf[i].seq(1).sig() == EQUIV) {
        AVar x = var_of(tnf[i].seq(0));
        auto val = constants[equivalence_classes[x.vid()]];
        if(val.lb() == val.ub() && val.lb() == 1)
        {
          add_equivalence(var_of(tnf[i].seq(1).seq(0)), var_of(tnf[i].seq(1).seq(1)));
          eliminate(eliminated_formulas, i);
          eliminated_equality_constraints++;
        }
      }
    }
    normalize_equivalence_classes();
    i_cse(tnf, eliminated_constraints_by_icse, icse_fixpoint_iterations);
    /** Keep only the variables that are representative and occur in at least one TNF constraint. */
    eliminated_variables.set();
    for(int i = 0; i < tnf.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        AVar x = var_of(tnf[i].seq(0));
        AVar y = var_of(tnf[i].seq(1).seq(0));
        AVar z = var_of(tnf[i].seq(1).seq(1));
        eliminated_variables.set(equivalence_classes[x.vid()], false);
        eliminated_variables.set(equivalence_classes[y.vid()], false);
        eliminated_variables.set(equivalence_classes[z.vid()], false);
      }
    }
    /** Update the constant values of eliminated variables.
     * This is useful later when printing the solutions.
     */
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      constants[equivalence_classes[i]].meet(sub->project(AVar{store_aty, i}));
    }
    /** Simplify the formulas. */
    typename F::Sequence seq(tnf.get_allocator());
    deinterpret_vars(seq);
    deinterpret_constraints(seq, tnf, true);
    return seq.size() == 0 ? F::make_true() : F::make_nary(AND, std::move(seq));
  }

private:
  template <class F>
  void substitute_var(F& f) const {
    if(f.is_variable()) {
      AVar x = var_of(f);
      if(equivalence_classes[x.vid()] != x.vid()) {
        x = AVar{store_aty, equivalence_classes[x.vid()]};
        f = F::make_lvar(store_aty, env.name_of(x));
      }
      if(eliminated_variables.test(x.vid())) {
        auto k = constants[x.vid()].template deinterpret<F>();
        if(env[x].sort.is_bool() && k.is(F::Z)) {
          f = k.z() == 0 ? F::make_false() : F::make_true();
        }
        else {
          f = std::move(k);
        }
      }
    }
  }

public:
  /** Given any formula (not necessarily in TNF), we substitute each variable with its representative variable or a constant if it got eliminated. */
  template <class F>
  CUDA void substitute(F& f) const {
    f.inplace_map([this](F& leaf, const F&) { substitute_var(leaf); });
  }

  CUDA size_t num_vars_after_elimination() const {
    size_t keep = 0;
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      if(equivalence_classes[i] == i && !eliminated_variables.test(i)) {
        ++keep;
      }
    }
    return keep;
  }

  CUDA size_t num_eliminated_variables() const {
    return equivalence_classes.size() - num_vars_after_elimination();
  }

private:
  // Deinterpret the existential quantifiers (only one per equivalence classes), and the domain of each variable.
  template <class Seq>
  CUDA void deinterpret_vars(Seq& seq) {
    using F = TFormula<allocator_type>;
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      if(equivalence_classes[i] == i && !eliminated_variables.test(i)) {
        const auto& x = env[AVar{store_aty, i}];
        seq.push_back(F::make_exists(UNTYPED, x.name, x.sort));
        auto domain_constraint = constants[i].deinterpret(AVar{store_aty, i}, env, get_allocator());
        map_avar_to_lvar(domain_constraint, env, true);
        seq.push_back(domain_constraint);
      }
    }
  }

  template <class Seq1, class Seq2>
  CUDA void deinterpret_constraints(Seq1& seq, const Seq2& formulas, bool enable_substitute = false) {
    // Deinterpret the simplified formulas.
    for(int i = 0; i < formulas.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        seq.push_back(formulas[i]);
        if(enable_substitute) {
          substitute(seq.back());
        }
      }
    }
  }

public:
  // A representative variable is eliminated if all variables in its equivalence class must be eliminated.
  CUDA void eliminate_variables() {
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      eliminated_variables.set(equivalence_classes[i], eliminated_variables.test(equivalence_classes[i]) && eliminated_variables.test(i));
    }
  }

  CUDA NI TFormula<allocator_type> deinterpret() {
    using F = TFormula<allocator_type>;
    typename F::Sequence seq(get_allocator());
    if(is_bot()) {
      return F::make_false();
    }
    deinterpret_vars(seq);
    deinterpret_constraints(seq, simplified_formulas);
    return seq.size() == 0 ? F::make_true() : F::make_nary(AND, std::move(seq));
  }
};

} // namespace lala

#endif
