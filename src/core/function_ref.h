#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace axon {

template <class Sig> class FunctionRef;

// Non-owning reference to a callable. Two words, trivially copyable, never allocates.
// The referenced callable must outlive the FunctionRef (fine for synchronous parallel_for).
template <class R, class... Args> class FunctionRef<R(Args...)> {
public:
  FunctionRef() = delete;

  // Function objects (lambdas, functors) are referenced by address.
  template <class F>
    requires(std::is_class_v<std::remove_cvref_t<F>> &&
             !std::is_same_v<std::remove_cvref_t<F>, FunctionRef> &&
             std::is_invocable_r_v<R, F&, Args...>)
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  FunctionRef(F&& f) noexcept : trampoline_(&invoke_object<std::remove_reference_t<F>>) {
    storage_.object = const_cast<void*>(static_cast<const void*>(std::addressof(f)));
  }

  // Plain functions are stored as a function pointer (a void* cannot hold one portably).
  template <class R2, class... Args2>
    requires(std::is_invocable_r_v<R, R2 (*)(Args2...), Args...>)
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  FunctionRef(R2 (*f)(Args2...)) noexcept : trampoline_(&invoke_function<R2, Args2...>) {
    storage_.function = reinterpret_cast<void (*)()>(f);
  }

  R operator()(Args... args) const { return trampoline_(storage_, std::forward<Args>(args)...); }

private:
  union Storage {
    void* object;
    void (*function)();
  };

  template <class F> static R invoke_object(Storage storage, Args... args) {
    return (*static_cast<F*>(storage.object))(std::forward<Args>(args)...);
  }

  template <class R2, class... Args2> static R invoke_function(Storage storage, Args... args) {
    auto* f = reinterpret_cast<R2 (*)(Args2...)>(storage.function);
    return f(std::forward<Args>(args)...);
  }

  Storage storage_{};
  R (*trampoline_)(Storage, Args...);
};

}  // namespace axon
