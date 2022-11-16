// Copyright 2021 Pierre Talbot

#ifndef IRESULT_HPP
#define IRESULT_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "logic/ast.hpp"

namespace lala {

/** The representation of an error (or warning) obtained when interpreting a formula in an abstract universe or domain. */
template<class F>
class IError {
public:
  using allocator_type = typename F::allocator_type;
  using this_type = IError<F>;

private:
  battery::string<allocator_type> ad_name;
  battery::string<allocator_type> description;
  F uninterpretable_formula;
  AType aty;
  battery::vector<IError<F>, allocator_type> suberrors;
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
  CUDA IError(bool fatal, battery::string<allocator_type> ad_name,
    battery::string<allocator_type> description,
    F uninterpretable_formula,
    AType aty = UNTYPED)
   : ad_name(std::move(ad_name)),
     description(std::move(description)),
     uninterpretable_formula(std::move(uninterpretable_formula)),
     aty(aty),
     fatal(fatal)
  {}

  CUDA this_type& add_suberror(IError<F>&& suberror) {
    suberrors.push_back(suberror);
    return *this;
  }

  CUDA void print(int indent = 0) const {
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

/** This class is used in abstract domains to represent the result of an interpretation.
    If the abstract domain cannot interpret the formula, it must explain why.
    This is similar to compilation errors in compiler. */
template <class T, class F>
class IResult {
public:
  using allocator_type = typename F::allocator_type;
  using error_type = IError<F>;
  using this_type = IResult<T, F>;

private:
  using warnings_type = battery::vector<error_type, allocator_type>;

  using result_type = battery::variant<
    T,
    error_type>;

  result_type result;
  warnings_type warnings;

  template <class U>
  CUDA static result_type map_result(battery::variant<U, error_type>&& other) {
    if(other.index() == 0) {
      return result_type::template create<0>(T(std::move(battery::get<0>(other))));
    }
    else {
      return result_type::template create<1>(std::move(battery::get<1>(other)));
    }
  }

public:
  template <class U, class F2>
  friend class IResult;

  CUDA IResult(T&& data):
    result(result_type::template create<0>(std::move(data))) {}

  CUDA IResult(T&& data, error_type&& warning):
    result(result_type::template create<0>(std::move(data)))
  {
    assert(!warning.is_fatal());
    push_warning(std::move(warning));
  }

  CUDA this_type& push_warning(error_type&& warning) {
    warnings.push_back(std::move(warning));
    return *this;
  }

  CUDA IResult(error_type&& error):
    result(result_type::template create<1>(std::move(error))) {}

  template<class U>
  CUDA IResult(IResult<U, F>&& map): result(map_result(std::move(map.result))),
    warnings(std::move(map.warnings)) {}

  CUDA this_type& operator=(this_type&& other) {
    result = std::move(other.result);
    warnings = std::move(other.warnings);
    return *this;
  }

  CUDA bool is_ok() const {
    return result.index() == 0;
  }

  CUDA bool has_value() const {
    return is_ok();
  }

  CUDA const T& value() const {
    return battery::get<0>(result);
  }

  template<class U>
  CUDA IResult<U, F> map(U&& data2) && {
    auto r = IResult<U, F>(std::move(data2));
    r.warnings = std::move(warnings);
    return std::move(r);
  }

  template<class U>
  CUDA this_type& join_warnings(IResult<U, F>&& other) {
    for(int i = 0; i < other.warnings.size(); ++i) {
      warnings.push_back(std::move(other.warnings[i]));
    }
    return *this;
  }

  template<class U>
  CUDA this_type& join_errors(IResult<U, F>&& other) {
    error().add_suberror(std::move(other.error()));
    return join_warnings(other);
  }

  CUDA T& value() {
    return battery::get<0>(result);
  }

  CUDA const error_type& error() const {
    return battery::get<1>(result);
  }

  CUDA error_type& error() {
    return battery::get<1>(result);
  }

  CUDA bool has_warning() const {
    return warnings.size() > 0;
  }

  CUDA void print_diagnostics() const {
    if(is_ok()) {
      printf("successfully interpreted\n");
    }
    else {
      error().print();
    }
    printf("\n");
    for(int i = 0; i < warnings.size(); ++i) {
      warnings[i].print();
    }
  }
};

}

#endif
