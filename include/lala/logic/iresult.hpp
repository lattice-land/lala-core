// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_IDIAGNOSTICS_HPP
#define LALA_CORE_IDIAGNOSTICS_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "ast.hpp"

namespace lala {

/** `IDiagnostics` is used in abstract domains to diagnose why a formula cannot be interpreted (error) or if it was interpreted by under- or over-approximation (warnings).
    If the abstract domain cannot interpret the formula, it must explain why.
    This is similar to compilation errors in compiler. */
template<class F, class Allocator = typename F::allocator_type>
class IDiagnostics {
public:
  using allocator_type = typename F::allocator_type;
  using this_type = IDiagnostics<F>;

private:
  battery::string<allocator_type> ad_name;
  battery::string<allocator_type> description;
  F uninterpretable_formula;
  AType aty;
  battery::vector<IDiagnostics<F>, allocator_type> suberrors;
  bool fatal;

  CUDA void print_indent(int indent) const {
    for(int i = 0; i < indent; ++i) {
      printf(" ");
    }
  }

  CUDA void print_line(const char* line, int indent) const {
    print_indent(indent);
    printf("%s", line);
  }

public:
  // If fatal is false, it is considered as a warning.
  CUDA NI IDiagnostics(battery::string<allocator_type> ad_name,
    battery::string<allocator_type> description,
    F uninterpretable_formula,
    AType aty = UNTYPED)
   : ad_name(std::move(ad_name)),
     description(std::move(description)),
     uninterpretable_formula(std::move(uninterpretable_formula)),
     aty(aty),
     fatal(fatal)
  {}

  CUDA NI this_type& add_suberror(IDiagnostics<F>&& suberror) {
    suberrors.push_back(suberror);
    return *this;
  }

  CUDA NI void print(int indent = 0) const {
    if(fatal) {
      print_line("[error] ", indent);
    }
    else {
      print_line("[warning] ", indent);
    }
    printf("Uninterpretable formula.\n");
    print_indent(indent);
    printf("  Abstract domain: %s\n", ad_name.data());
    print_line("  Abstract type: ", indent);
    if(aty == UNTYPED) {
      printf("untyped\n");
    }
    else {
      printf("%d\n", aty);
    }
    print_line("  Formula: ", indent);
    uninterpretable_formula.print(true);
    printf("\n");
    print_indent(indent);
    printf("  Description: %s\n", description.data());
    for(int i = 0; i < suberrors.size(); ++i) {
      suberrors[i].print(indent + 2);
      printf("\n");
    }
  }

  CUDA bool is_fatal() const { return fatal; }
};

#define RETURN_INTERPRETATION_ERROR(msg) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(true, name, (msg), f)); \
  } \
  return false;

#define RETURN_INTERPRETATION_WARNING(msg) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(false, name, (msg), f)); \
  } \
  return true;

}

#endif
