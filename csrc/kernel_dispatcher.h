#pragma once
#include <torch/extension.h>
#include <torch/types.h>
#include <cutlass/numeric_types.h>


namespace flash_attention {

template <typename T>
struct TypeWrapper { using type = T; };

template <typename T, T val>
struct ValueWrapper { static constexpr T value = val; };


inline void fallback_no_impl() {
    TORCH_CHECK(false, "No suitable implementation for flash attention kernel.");
}

template <typename NextDispatcher>
inline void type_dispatcher(torch::ScalarType type, NextDispatcher&& next_dispatcher) {
    switch (type)
    {
    case torch::kHalf:
        next_dispatcher(TypeWrapper<cutlass::half_t>{});
        break;
    case torch::kBFloat16:
        next_dispatcher(TypeWrapper<cutlass::bfloat16_t>{});
        break;
    default:
        fallback_no_impl();
        break;
    }
}

template <typename NextDispatcher>
inline void bool_dispatcher(bool flag, NextDispatcher&& next_dispatcher) {
    if (flag) {
        next_dispatcher(ValueWrapper<bool, true>{});
    } else {
        next_dispatcher(ValueWrapper<bool, false>{});
    }
}

template <typename NextDispatcher>
inline void headdim_dispatcher(int64_t headdim, NextDispatcher&& next_dispatcher) {
    if (headdim <= 128 ) {
        next_dispatcher(ValueWrapper<int64_t, 128>{});
    } else {
        fallback_no_impl();
    }
}

} // namespace flash_attention
