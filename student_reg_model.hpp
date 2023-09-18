// Code generated by stanc v2.32.1
#include <stan/model/model_header.hpp>
namespace student_reg_model_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 19> locations_array__ =
  {" (found before start of program)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 9, column 4 to column 15)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 10, column 4 to column 15)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 11, column 4 to column 15)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 12, column 4 to column 15)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 16, column 4 to column 25)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 17, column 4 to column 25)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 18, column 4 to column 25)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 19, column 4 to column 25)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 22, column 4 to column 70)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 2, column 4 to column 19)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 3, column 11 to column 12)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 3, column 4 to column 17)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 4, column 11 to column 12)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 4, column 4 to column 17)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 5, column 11 to column 12)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 5, column 4 to column 17)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 6, column 11 to column 12)",
  " (in '/Users/nicolabranchini/PycharmProjects/ais-heavy-tails/student_reg_model.stan', line 6, column 4 to column 16)"};
class student_reg_model_model final : public model_base_crtp<student_reg_model_model> {
 private:
  int N;
  Eigen::Matrix<double,-1,1> x1_data__;
  Eigen::Matrix<double,-1,1> x2_data__;
  Eigen::Matrix<double,-1,1> x3_data__;
  Eigen::Matrix<double,-1,1> y_data__;
  Eigen::Map<Eigen::Matrix<double,-1,1>> x1{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,1>> x2{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,1>> x3{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,1>> y{nullptr, 0};
 public:
  ~student_reg_model_model() {}
  student_reg_model_model(stan::io::var_context& context__, unsigned int
                          random_seed__ = 0, std::ostream*
                          pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "student_reg_model_model_namespace::student_reg_model_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 10;
      context__.validate_dims("data initialization", "N", "int",
        std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      current_statement__ = 10;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 10;
      stan::math::check_greater_or_equal(function__, "N", N, 0);
      current_statement__ = 11;
      stan::math::validate_non_negative_index("x1", "N", N);
      current_statement__ = 12;
      context__.validate_dims("data initialization", "x1", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      x1_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                    std::numeric_limits<double>::quiet_NaN());
      new (&x1) Eigen::Map<Eigen::Matrix<double,-1,1>>(x1_data__.data(), N);
      {
        std::vector<local_scalar_t__> x1_flat__;
        current_statement__ = 12;
        x1_flat__ = context__.vals_r("x1");
        current_statement__ = 12;
        pos__ = 1;
        current_statement__ = 12;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 12;
          stan::model::assign(x1, x1_flat__[(pos__ - 1)],
            "assigning variable x1", stan::model::index_uni(sym1__));
          current_statement__ = 12;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 13;
      stan::math::validate_non_negative_index("x2", "N", N);
      current_statement__ = 14;
      context__.validate_dims("data initialization", "x2", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      x2_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                    std::numeric_limits<double>::quiet_NaN());
      new (&x2) Eigen::Map<Eigen::Matrix<double,-1,1>>(x2_data__.data(), N);
      {
        std::vector<local_scalar_t__> x2_flat__;
        current_statement__ = 14;
        x2_flat__ = context__.vals_r("x2");
        current_statement__ = 14;
        pos__ = 1;
        current_statement__ = 14;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 14;
          stan::model::assign(x2, x2_flat__[(pos__ - 1)],
            "assigning variable x2", stan::model::index_uni(sym1__));
          current_statement__ = 14;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 15;
      stan::math::validate_non_negative_index("x3", "N", N);
      current_statement__ = 16;
      context__.validate_dims("data initialization", "x3", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      x3_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                    std::numeric_limits<double>::quiet_NaN());
      new (&x3) Eigen::Map<Eigen::Matrix<double,-1,1>>(x3_data__.data(), N);
      {
        std::vector<local_scalar_t__> x3_flat__;
        current_statement__ = 16;
        x3_flat__ = context__.vals_r("x3");
        current_statement__ = 16;
        pos__ = 1;
        current_statement__ = 16;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 16;
          stan::model::assign(x3, x3_flat__[(pos__ - 1)],
            "assigning variable x3", stan::model::index_uni(sym1__));
          current_statement__ = 16;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 17;
      stan::math::validate_non_negative_index("y", "N", N);
      current_statement__ = 18;
      context__.validate_dims("data initialization", "y", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      y_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                   std::numeric_limits<double>::quiet_NaN());
      new (&y) Eigen::Map<Eigen::Matrix<double,-1,1>>(y_data__.data(), N);
      {
        std::vector<local_scalar_t__> y_flat__;
        current_statement__ = 18;
        y_flat__ = context__.vals_r("y");
        current_statement__ = 18;
        pos__ = 1;
        current_statement__ = 18;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 18;
          stan::model::assign(y, y_flat__[(pos__ - 1)],
            "assigning variable y", stan::model::index_uni(sym1__));
          current_statement__ = 18;
          pos__ = (pos__ + 1);
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1 + 1 + 1;
  }
  inline std::string model_name() const final {
    return "student_reg_model_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.32.1",
             "stancflags = --include-paths=."};
  }
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "student_reg_model_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 1;
      alpha = in__.template read<local_scalar_t__>();
      local_scalar_t__ beta1 = DUMMY_VAR__;
      current_statement__ = 2;
      beta1 = in__.template read<local_scalar_t__>();
      local_scalar_t__ beta2 = DUMMY_VAR__;
      current_statement__ = 3;
      beta2 = in__.template read<local_scalar_t__>();
      local_scalar_t__ beta3 = DUMMY_VAR__;
      current_statement__ = 4;
      beta3 = in__.template read<local_scalar_t__>();
      {
        current_statement__ = 5;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(alpha, 0, 1));
        current_statement__ = 6;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(beta1, 0, 1));
        current_statement__ = 7;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(beta2, 0, 1));
        current_statement__ = 8;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(beta3, 0, 1));
        current_statement__ = 9;
        lp_accum__.add(stan::math::student_t_lpdf<propto__>(y, 5,
                         stan::math::add(
                           stan::math::add(
                             stan::math::add(alpha,
                               stan::math::multiply(beta1, x1)),
                             stan::math::multiply(beta2, x2)),
                           stan::math::multiply(beta3, x3)), 1));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    static constexpr const char* function__ =
      "student_reg_model_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      double alpha = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      alpha = in__.template read<local_scalar_t__>();
      double beta1 = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      beta1 = in__.template read<local_scalar_t__>();
      double beta2 = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      beta2 = in__.template read<local_scalar_t__>();
      double beta3 = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 4;
      beta3 = in__.template read<local_scalar_t__>();
      out__.write(alpha);
      out__.write(beta1);
      out__.write(beta2);
      out__.write(beta3);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 1;
      alpha = in__.read<local_scalar_t__>();
      out__.write(alpha);
      local_scalar_t__ beta1 = DUMMY_VAR__;
      current_statement__ = 2;
      beta1 = in__.read<local_scalar_t__>();
      out__.write(beta1);
      local_scalar_t__ beta2 = DUMMY_VAR__;
      current_statement__ = 3;
      beta2 = in__.read<local_scalar_t__>();
      out__.write(beta2);
      local_scalar_t__ beta3 = DUMMY_VAR__;
      current_statement__ = 4;
      beta3 = in__.read<local_scalar_t__>();
      out__.write(beta3);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "alpha", "double",
        std::vector<size_t>{});
      current_statement__ = 2;
      context__.validate_dims("parameter initialization", "beta1", "double",
        std::vector<size_t>{});
      current_statement__ = 3;
      context__.validate_dims("parameter initialization", "beta2", "double",
        std::vector<size_t>{});
      current_statement__ = 4;
      context__.validate_dims("parameter initialization", "beta3", "double",
        std::vector<size_t>{});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 1;
      alpha = context__.vals_r("alpha")[(1 - 1)];
      out__.write(alpha);
      local_scalar_t__ beta1 = DUMMY_VAR__;
      current_statement__ = 2;
      beta1 = context__.vals_r("beta1")[(1 - 1)];
      out__.write(beta1);
      local_scalar_t__ beta2 = DUMMY_VAR__;
      current_statement__ = 3;
      beta2 = context__.vals_r("beta2")[(1 - 1)];
      out__.write(beta2);
      local_scalar_t__ beta3 = DUMMY_VAR__;
      current_statement__ = 4;
      beta3 = context__.vals_r("beta3")[(1 - 1)];
      out__.write(beta3);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"alpha", "beta1", "beta2", "beta3"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
                std::vector<size_t>{}, std::vector<size_t>{},
                std::vector<size_t>{}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta1");
    param_names__.emplace_back(std::string() + "beta2");
    param_names__.emplace_back(std::string() + "beta3");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta1");
    param_names__.emplace_back(std::string() + "beta2");
    param_names__.emplace_back(std::string() + "beta3");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta1\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta3\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta1\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta3\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((1 + 1) + 1) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((1 + 1) + 1) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = student_reg_model_model_namespace::student_reg_model_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return student_reg_model_model_namespace::profiles__;
}
#endif