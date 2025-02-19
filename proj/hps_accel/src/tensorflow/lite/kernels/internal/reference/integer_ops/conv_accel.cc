/*
 * Copyright 2021 The CFU-Playground Authors
 * Copyright 2019 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv_accel.h"
#include "blocks.h"

using hps_accel::Vector16;
using hps_accel::multiply_accumulate;

namespace tflite {
namespace reference_integer_ops {

void ConvPerChannel4x4(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  TFLITE_DCHECK_EQ(params.dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(params.dilation_height_factor, 1);
  TFLITE_DCHECK_EQ(params.padding_type, PaddingType::kValid);
  TFLITE_DCHECK_EQ(params.padding_values.width, 0);
  TFLITE_DCHECK_EQ(params.padding_values.height, 0);
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  TFLITE_DCHECK(input_depth == 1 || input_depth % 4 == 0);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  TFLITE_DCHECK_EQ(filter_shape.Dims(1), 4);
  const int filter_height = 4;
  TFLITE_DCHECK_EQ(filter_shape.Dims(2), 4);
  const int filter_width = 4;

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  hps_accel::LoadFilter(input_depth, output_depth, filter_data);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = out_y * stride_height;
      // Check bounds for input buffer. This assumes "valid" padding type.
      TFLITE_DCHECK_LE(in_y_origin + filter_height, input_height);
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_width;
        // Check bounds for input buffer. This assumes "valid" padding type.
        TFLITE_DCHECK_LE(in_x_origin + filter_width, input_width);
        const int8_t *current_input_data = input_data +
            Offset(input_shape, batch, in_y_origin, in_x_origin, 0);
        hps_accel::LoadInput(input_width, input_depth, current_input_data);
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = 0;
          for (int i = 0; i < filter_height * filter_width * input_depth / 16; ++i) {
            Vector16 input = hps_accel::GetInput();
            Vector16 filter = hps_accel::GetFilter();
            acc += multiply_accumulate(input, filter, input_offset);
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
