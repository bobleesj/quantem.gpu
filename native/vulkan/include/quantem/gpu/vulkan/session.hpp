#pragma once

#include "quantem/gpu/vulkan/c_api.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace quantem::gpu::vulkan {

class StatusError : public std::runtime_error {
public:
  StatusError(const qgpu_status status, const qgpu_error &error)
      : std::runtime_error(error.message), status_(status),
        native_code_(error.native_code) {}

  [[nodiscard]] qgpu_status status() const noexcept { return status_; }
  [[nodiscard]] int native_code() const noexcept { return native_code_; }

private:
  qgpu_status status_;
  int native_code_;
};

inline void throw_on_error(const qgpu_status status, const qgpu_error &error) {
  if (status != QGPU_STATUS_OK)
    throw StatusError(status, error);
}

class Result {
public:
  explicit Result(qgpu_vulkan_result *value = nullptr) : value_(value) {}
  Result(const Result &) = delete;
  Result &operator=(const Result &) = delete;
  Result(Result &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}
  Result &operator=(Result &&other) noexcept {
    if (this != &other) {
      qgpu_vulkan_result_release_v1(&value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }
  ~Result() { qgpu_vulkan_result_release_v1(&value_); }

  [[nodiscard]] qgpu_vulkan_result_view view() const {
    qgpu_vulkan_result_view result{};
    result.struct_size = sizeof(result);
    qgpu_error error{};
    error.struct_size = sizeof(error);
    throw_on_error(qgpu_vulkan_result_view_v1(value_, &result, &error), error);
    return result;
  }

private:
  qgpu_vulkan_result *value_;
};

class Session {
public:
  explicit Session(const qgpu_vulkan_open_request &request) {
    qgpu_error error{};
    error.struct_size = sizeof(error);
    throw_on_error(qgpu_vulkan_open_v1(&request, &value_, &error), error);
  }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}
  Session &operator=(Session &&other) noexcept {
    if (this != &other) {
      qgpu_vulkan_close_v1(&value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }
  ~Session() { qgpu_vulkan_close_v1(&value_); }

  [[nodiscard]] Result
  request_products(const qgpu_vulkan_product_request &request) {
    qgpu_vulkan_result *result = nullptr;
    qgpu_error error{};
    error.struct_size = sizeof(error);
    throw_on_error(
        qgpu_vulkan_request_products_v1(value_, &request, &result, &error),
        error);
    return Result(result);
  }

  void cancel_through(const std::uint64_t generation) {
    qgpu_error error{};
    error.struct_size = sizeof(error);
    throw_on_error(
        qgpu_vulkan_cancel_through_generation_v1(value_, generation, &error),
        error);
  }

  [[nodiscard]] bool poll_event(qgpu_vulkan_event *event) {
    if (event == nullptr)
      throw std::invalid_argument("event is null");
    event->struct_size = sizeof(*event);
    qgpu_error error{};
    error.struct_size = sizeof(error);
    const qgpu_status status = qgpu_vulkan_poll_event_v1(value_, event, &error);
    if (status == QGPU_STATUS_NO_EVENT)
      return false;
    throw_on_error(status, error);
    return true;
  }

private:
  qgpu_vulkan_session *value_ = nullptr;
};

} // namespace quantem::gpu::vulkan
