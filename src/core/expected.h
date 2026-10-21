#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace axon {

enum class ErrorCode : std::uint8_t {
  Ok = 0,
  Io,
  Parse,
  Unsupported,
  InvalidArgument,
  OutOfMemory,
  Internal,
};

constexpr const char* error_code_name(ErrorCode code) {
  switch (code) {
  case ErrorCode::Ok:
    return "ok";
  case ErrorCode::Io:
    return "io";
  case ErrorCode::Parse:
    return "parse";
  case ErrorCode::Unsupported:
    return "unsupported";
  case ErrorCode::InvalidArgument:
    return "invalid_argument";
  case ErrorCode::OutOfMemory:
    return "out_of_memory";
  case ErrorCode::Internal:
    return "internal";
  }
  return "unknown";
}

// A recoverable, user-facing failure. Messages state what was expected and what was found.
struct Error {
  ErrorCode code = ErrorCode::Internal;
  std::string message;
};

inline Error make_error(ErrorCode code, std::string message) {
  return Error{.code = code, .message = std::move(message)};
}

// Minimal result type: either a T or an Error. Used for I/O and parsing paths where
// exceptions are not wanted. Deliberately small; no monadic operators.
template <class T> class [[nodiscard]] Expected {
  static_assert(!std::is_reference_v<T>, "Expected<T&> is not supported");

public:
  using value_type = T;

  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  Expected(T value) : has_value_(true) {
    new (static_cast<void*>(&storage_.value)) T(std::move(value));
  }
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  Expected(Error error) : has_value_(false) { new (&storage_.error) Error(std::move(error)); }

  Expected(const Expected& other) : has_value_(other.has_value_) {
    if (has_value_) {
      new (&storage_.value) T(other.storage_.value);
    } else {
      new (&storage_.error) Error(other.storage_.error);
    }
  }

  Expected(Expected&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
      : has_value_(other.has_value_) {
    if (has_value_) {
      new (&storage_.value) T(std::move(other.storage_.value));
    } else {
      new (&storage_.error) Error(std::move(other.storage_.error));
    }
  }

  Expected& operator=(const Expected& other) {
    if (this != &other) {
      Expected tmp(other);
      swap_with(tmp);
    }
    return *this;
  }

  Expected& operator=(Expected&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
    if (this != &other) {
      destroy();
      has_value_ = other.has_value_;
      if (has_value_) {
        new (&storage_.value) T(std::move(other.storage_.value));
      } else {
        new (&storage_.error) Error(std::move(other.storage_.error));
      }
    }
    return *this;
  }

  ~Expected() { destroy(); }

  [[nodiscard]] bool has_value() const noexcept { return has_value_; }
  explicit operator bool() const noexcept { return has_value_; }

  T& value() & noexcept { return storage_.value; }
  const T& value() const& noexcept { return storage_.value; }
  T&& value() && noexcept { return std::move(storage_.value); }

  T& operator*() & noexcept { return storage_.value; }
  const T& operator*() const& noexcept { return storage_.value; }
  T&& operator*() && noexcept { return std::move(storage_.value); }
  T* operator->() noexcept { return &storage_.value; }
  const T* operator->() const noexcept { return &storage_.value; }

  const Error& error() const& noexcept { return storage_.error; }
  Error& error() & noexcept { return storage_.error; }
  Error&& error() && noexcept { return std::move(storage_.error); }

  template <class U> T value_or(U&& fallback) const& {
    return has_value_ ? storage_.value : static_cast<T>(std::forward<U>(fallback));
  }

private:
  union Storage {
    Storage() {}   // NOLINT(modernize-use-equals-default): union members are managed manually
    ~Storage() {}  // NOLINT(modernize-use-equals-default)
    T value;
    Error error;
  };

  void destroy() noexcept {
    if (has_value_) {
      storage_.value.~T();
    } else {
      storage_.error.~Error();
    }
  }

  void swap_with(Expected& other) {
    Expected tmp(std::move(other));
    // The moved-from object is destroyed and rebuilt in place below, which is the point.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    other.destroy();
    other.has_value_ = has_value_;
    if (has_value_) {
      new (&other.storage_.value) T(std::move(storage_.value));
    } else {
      new (&other.storage_.error) Error(std::move(storage_.error));
    }
    destroy();
    has_value_ = tmp.has_value_;
    if (has_value_) {
      new (&storage_.value) T(std::move(tmp.storage_.value));
    } else {
      new (&storage_.error) Error(std::move(tmp.storage_.error));
    }
  }

  Storage storage_;
  bool has_value_;
};

// Expected<void>: success carries nothing, failure carries an Error.
template <> class [[nodiscard]] Expected<void> {
public:
  using value_type = void;

  Expected() noexcept = default;
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  Expected(Error error) : error_(std::move(error)), has_value_(false) {}

  [[nodiscard]] bool has_value() const noexcept { return has_value_; }
  explicit operator bool() const noexcept { return has_value_; }

  const Error& error() const& noexcept { return error_; }
  Error& error() & noexcept { return error_; }
  Error&& error() && noexcept { return std::move(error_); }

private:
  Error error_;
  bool has_value_ = true;
};

}  // namespace axon
