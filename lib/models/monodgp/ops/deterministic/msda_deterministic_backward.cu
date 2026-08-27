#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kReduceThreads = 32;

inline int ceil_log2_u64(uint64_t value) {
  int bits = 0;
  uint64_t capacity = 1;
  while (capacity < value) {
    capacity <<= 1;
    ++bits;
  }
  return std::max(bits, 1);
}

template <typename scalar_t>
__global__ void generate_destination_pairs_kernel(
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ sources,
    int64_t num_records,
    int batch_begin,
    int spatial_size,
    int num_query,
    int num_heads,
    int num_levels,
    int num_points,
    uint32_t invalid_key,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const scalar_t* __restrict__ sampling_locations) {
  const int64_t record = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (record >= num_records) {
    return;
  }

  int64_t cursor = record;
  const int corner = cursor & 3;
  cursor >>= 2;
  const int point = cursor % num_points;
  cursor /= num_points;
  const int level = cursor % num_levels;
  cursor /= num_levels;
  const int head = cursor % num_heads;
  cursor /= num_heads;
  const int query = cursor % num_query;
  const int local_batch = cursor / num_query;
  const int global_batch = batch_begin + local_batch;

  const int height = static_cast<int>(spatial_shapes[level * 2]);
  const int width = static_cast<int>(spatial_shapes[level * 2 + 1]);
  const int level_start = static_cast<int>(level_start_index[level]);
  const int64_t sampling_index =
      (((((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + head)
          * num_levels + level) * num_points + point) * 2);
  const scalar_t x = sampling_locations[sampling_index] * static_cast<scalar_t>(width)
      - static_cast<scalar_t>(0.5);
  const scalar_t y = sampling_locations[sampling_index + 1] * static_cast<scalar_t>(height)
      - static_cast<scalar_t>(0.5);

  if (!(x > static_cast<scalar_t>(-1) && x < static_cast<scalar_t>(width)
        && y > static_cast<scalar_t>(-1) && y < static_cast<scalar_t>(height))) {
    keys[record] = invalid_key;
    sources[record] = static_cast<uint32_t>(record);
    return;
  }

  const int x0 = static_cast<int>(floor(static_cast<double>(x)));
  const int y0 = static_cast<int>(floor(static_cast<double>(y)));
  const int xx = x0 + ((corner == 1 || corner == 3) ? 1 : 0);
  const int yy = y0 + ((corner >= 2) ? 1 : 0);
  if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
    keys[record] = invalid_key;
    sources[record] = static_cast<uint32_t>(record);
    return;
  }

  const int spatial_index = level_start + yy * width + xx;
  const uint32_t destination = static_cast<uint32_t>(
      (static_cast<uint64_t>(local_batch) * spatial_size + spatial_index) * num_heads + head);
  keys[record] = destination;
  sources[record] = static_cast<uint32_t>(record);
}

__device__ __forceinline__ int64_t lower_bound_key(
    const uint32_t* __restrict__ sorted_keys,
    int64_t count,
    uint32_t target) {
  int64_t first = 0;
  int64_t length = count;
  while (length > 0) {
    const int64_t half = length >> 1;
    const int64_t middle = first + half;
    if (sorted_keys[middle] < target) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first;
}

template <typename scalar_t>
__global__ void reduce_sorted_grad_value_kernel(
    const uint32_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ sorted_sources,
    int64_t num_records,
    int batch_begin,
    int chunk_batch,
    int spatial_size,
    int num_query,
    int num_heads,
    int channels,
    int num_levels,
    int num_points,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const scalar_t* __restrict__ sampling_locations,
    const scalar_t* __restrict__ attention_weights,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_value) {
  const uint32_t destination = blockIdx.x;
  __shared__ int64_t shared_begin;
  __shared__ int64_t shared_end;
  if (threadIdx.x == 0) {
    shared_begin = lower_bound_key(sorted_keys, num_records, destination);
    shared_end = lower_bound_key(sorted_keys, num_records, destination + 1);
  }
  __syncthreads();
  const int64_t begin = shared_begin;
  const int64_t end = shared_end;
  if (begin == end) {
    return;
  }

  uint32_t destination_cursor = destination;
  const int head = destination_cursor % num_heads;
  destination_cursor /= num_heads;
  const int spatial_index = destination_cursor % spatial_size;
  const int local_batch = destination_cursor / spatial_size;
  if (local_batch >= chunk_batch) {
    return;
  }
  const int global_batch = batch_begin + local_batch;
  uint32_t first_record = sorted_sources[begin];
  uint32_t first_cursor = first_record >> 2;
  first_cursor /= num_points;
  const int level = first_cursor % num_levels;
  const int height = static_cast<int>(spatial_shapes[level * 2]);
  const int width = static_cast<int>(spatial_shapes[level * 2 + 1]);
  const int level_start = static_cast<int>(level_start_index[level]);
  const int pixel = spatial_index - level_start;

  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    scalar_t accumulator = static_cast<scalar_t>(0);
    for (int64_t position = begin; position < end; ++position) {
      int64_t record = static_cast<int64_t>(sorted_sources[position]);
      const int corner = record & 3;
      record >>= 2;
      const int point = record % num_points;
      record /= num_points;
      const int record_level = record % num_levels;
      record /= num_levels;
      const int record_head = record % num_heads;
      record /= num_heads;
      const int query = record % num_query;

      const int64_t sampling_index =
          (((((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + record_head)
              * num_levels + record_level) * num_points + point) * 2);
      const scalar_t x = sampling_locations[sampling_index] * static_cast<scalar_t>(width)
          - static_cast<scalar_t>(0.5);
      const scalar_t y = sampling_locations[sampling_index + 1] * static_cast<scalar_t>(height)
          - static_cast<scalar_t>(0.5);
      const scalar_t dx = x - static_cast<scalar_t>(floor(static_cast<double>(x)));
      const scalar_t dy = y - static_cast<scalar_t>(floor(static_cast<double>(y)));

      scalar_t bilinear_weight;
      if (corner == 0) {
        bilinear_weight = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
      } else if (corner == 1) {
        bilinear_weight = dx * (static_cast<scalar_t>(1) - dy);
      } else if (corner == 2) {
        bilinear_weight = (static_cast<scalar_t>(1) - dx) * dy;
      } else {
        bilinear_weight = dx * dy;
      }

      const int64_t attention_index =
          ((((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + record_head)
             * num_levels + record_level) * num_points + point);
      const int64_t grad_output_index =
          (((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + record_head)
           * channels + channel);
      accumulator += grad_output[grad_output_index]
          * attention_weights[attention_index] * bilinear_weight;
    }

    const int64_t output_index =
        (((static_cast<int64_t>(global_batch) * spatial_size + spatial_index) * num_heads + head)
         * channels + channel);
    grad_value[output_index] = accumulator;
  }
}

template <typename scalar_t>
__global__ void deterministic_query_grads_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const scalar_t* __restrict__ sampling_locations,
    const scalar_t* __restrict__ attention_weights,
    const scalar_t* __restrict__ grad_output,
    int batch,
    int spatial_size,
    int num_query,
    int num_heads,
    int channels,
    int num_levels,
    int num_points,
    scalar_t* __restrict__ grad_sampling_locations,
    scalar_t* __restrict__ grad_attention_weights) {
  constexpr int kWarpSize = 32;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t warp_index =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t total_warps = static_cast<int64_t>(batch) * num_query * num_heads;
  if (warp_index >= total_warps) {
    return;
  }

  int64_t cursor = warp_index;
  const int head = cursor % num_heads;
  cursor /= num_heads;
  const int query = cursor % num_query;
  const int batch_index = cursor / num_query;

  for (int level = 0; level < num_levels; ++level) {
    const int height = static_cast<int>(spatial_shapes[level * 2]);
    const int width = static_cast<int>(spatial_shapes[level * 2 + 1]);
    const int level_start = static_cast<int>(level_start_index[level]);
    const int64_t level_value_base =
        (batch_index * static_cast<int64_t>(spatial_size) + level_start)
        * num_heads * channels;

    for (int point = 0; point < num_points; ++point) {
      const int64_t attention_index =
          ((((batch_index * static_cast<int64_t>(num_query) + query) * num_heads + head)
             * num_levels + level) * num_points + point);
      const int64_t sampling_index = attention_index * 2;
      const scalar_t x = sampling_locations[sampling_index] * static_cast<scalar_t>(width)
          - static_cast<scalar_t>(0.5);
      const scalar_t y = sampling_locations[sampling_index + 1] * static_cast<scalar_t>(height)
          - static_cast<scalar_t>(0.5);

      scalar_t grad_attention = static_cast<scalar_t>(0);
      scalar_t grad_x = static_cast<scalar_t>(0);
      scalar_t grad_y = static_cast<scalar_t>(0);
      if (x > static_cast<scalar_t>(-1) && x < static_cast<scalar_t>(width)
          && y > static_cast<scalar_t>(-1) && y < static_cast<scalar_t>(height)) {
        const int x0 = static_cast<int>(floor(static_cast<double>(x)));
        const int y0 = static_cast<int>(floor(static_cast<double>(y)));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const scalar_t dx = x - static_cast<scalar_t>(x0);
        const scalar_t dy = y - static_cast<scalar_t>(y0);
        const scalar_t one_minus_dx = static_cast<scalar_t>(1) - dx;
        const scalar_t one_minus_dy = static_cast<scalar_t>(1) - dy;
        const scalar_t attention = attention_weights[attention_index];

        for (int channel = lane; channel < channels; channel += kWarpSize) {
          scalar_t v00 = static_cast<scalar_t>(0);
          scalar_t v01 = static_cast<scalar_t>(0);
          scalar_t v10 = static_cast<scalar_t>(0);
          scalar_t v11 = static_cast<scalar_t>(0);
          if (y0 >= 0 && x0 >= 0) {
            v00 = value[level_value_base
                + (static_cast<int64_t>(y0) * width + x0) * num_heads * channels
                + head * channels + channel];
          }
          if (y0 >= 0 && x1 < width) {
            v01 = value[level_value_base
                + (static_cast<int64_t>(y0) * width + x1) * num_heads * channels
                + head * channels + channel];
          }
          if (y1 < height && x0 >= 0) {
            v10 = value[level_value_base
                + (static_cast<int64_t>(y1) * width + x0) * num_heads * channels
                + head * channels + channel];
          }
          if (y1 < height && x1 < width) {
            v11 = value[level_value_base
                + (static_cast<int64_t>(y1) * width + x1) * num_heads * channels
                + head * channels + channel];
          }

          const scalar_t sampled =
              v00 * one_minus_dx * one_minus_dy
              + v01 * dx * one_minus_dy
              + v10 * one_minus_dx * dy
              + v11 * dx * dy;
          const scalar_t d_sampled_dx =
              (v01 - v00) * one_minus_dy + (v11 - v10) * dy;
          const scalar_t d_sampled_dy =
              (v10 - v00) * one_minus_dx + (v11 - v01) * dx;
          const int64_t grad_output_index =
              ((batch_index * static_cast<int64_t>(num_query) + query) * num_heads + head)
              * channels + channel;
          const scalar_t top_grad = grad_output[grad_output_index];
          grad_attention += top_grad * sampled;
          grad_x += top_grad * attention * d_sampled_dx;
          grad_y += top_grad * attention * d_sampled_dy;
        }
      }

      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        grad_attention += __shfl_down_sync(0xffffffff, grad_attention, offset);
        grad_x += __shfl_down_sync(0xffffffff, grad_x, offset);
        grad_y += __shfl_down_sync(0xffffffff, grad_y, offset);
      }
      if (lane == 0) {
        grad_attention_weights[attention_index] = grad_attention;
        grad_sampling_locations[sampling_index] = grad_x * static_cast<scalar_t>(width);
        grad_sampling_locations[sampling_index + 1] = grad_y * static_cast<scalar_t>(height);
      }
    }
  }
}

template <typename scalar_t>
__global__ void generate_level_pairs_kernel(
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ sources,
    scalar_t* __restrict__ bilinear_weights,
    scalar_t* __restrict__ cached_attention,
    uint32_t* __restrict__ grad_output_bases,
    int64_t num_records,
    int batch_begin,
    int num_query,
    int num_heads,
    int channels,
    int num_levels,
    int num_points,
    int level,
    int height,
    int width,
    uint32_t invalid_key,
    const scalar_t* __restrict__ sampling_locations,
    const scalar_t* __restrict__ attention_weights) {
  const int64_t source = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (source >= num_records) {
    return;
  }
  int64_t cursor = source;
  const int corner = cursor & 3;
  cursor >>= 2;
  const int point = cursor % num_points;
  cursor /= num_points;
  const int head = cursor % num_heads;
  cursor /= num_heads;
  const int query = cursor % num_query;
  const int local_batch = cursor / num_query;
  const int global_batch = batch_begin + local_batch;
  const int64_t sampling_index =
      (((((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + head)
          * num_levels + level) * num_points + point) * 2);
  const scalar_t x = sampling_locations[sampling_index] * static_cast<scalar_t>(width)
      - static_cast<scalar_t>(0.5);
  const scalar_t y = sampling_locations[sampling_index + 1] * static_cast<scalar_t>(height)
      - static_cast<scalar_t>(0.5);
  uint32_t destination = invalid_key;
  scalar_t bilinear_weight = static_cast<scalar_t>(0);
  if (x > static_cast<scalar_t>(-1) && x < static_cast<scalar_t>(width)
      && y > static_cast<scalar_t>(-1) && y < static_cast<scalar_t>(height)) {
    const int x0 = static_cast<int>(floor(static_cast<double>(x)));
    const int y0 = static_cast<int>(floor(static_cast<double>(y)));
    const int xx = x0 + ((corner == 1 || corner == 3) ? 1 : 0);
    const int yy = y0 + ((corner >= 2) ? 1 : 0);
    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
      destination = static_cast<uint32_t>(
          (static_cast<uint64_t>(local_batch) * height * width + yy * width + xx)
          * num_heads + head);
      const scalar_t dx = x - static_cast<scalar_t>(floor(static_cast<double>(x)));
      const scalar_t dy = y - static_cast<scalar_t>(floor(static_cast<double>(y)));
      if (corner == 0) {
        bilinear_weight = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
      } else if (corner == 1) {
        bilinear_weight = dx * (static_cast<scalar_t>(1) - dy);
      } else if (corner == 2) {
        bilinear_weight = (static_cast<scalar_t>(1) - dx) * dy;
      } else {
        bilinear_weight = dx * dy;
      }
    }
  }
  keys[source] = destination;
  sources[source] = static_cast<uint32_t>(source);
  bilinear_weights[source] = bilinear_weight;
  const int64_t attention_index =
      ((((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + head)
         * num_levels + level) * num_points + point);
  cached_attention[source] = attention_weights[attention_index];
  grad_output_bases[source] = static_cast<uint32_t>(
      ((static_cast<int64_t>(global_batch) * num_query + query) * num_heads + head) * channels);
}

template <typename scalar_t>
__global__ void reduce_level_pairs_kernel(
    const uint32_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ sorted_sources,
    int64_t num_records,
    int batch_begin,
    int spatial_size,
    int num_query,
    int num_heads,
    int channels,
    int num_levels,
    int num_points,
    int level,
    int level_start,
    int height,
    int width,
    const scalar_t* __restrict__ bilinear_weights,
    const scalar_t* __restrict__ cached_attention,
    const uint32_t* __restrict__ grad_output_bases,
  const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_value) {
  const uint32_t destination = blockIdx.x;
  __shared__ int64_t shared_bounds[2];
  if (threadIdx.x == 0) {
    shared_bounds[0] = lower_bound_key(sorted_keys, num_records, destination);
    shared_bounds[1] = lower_bound_key(sorted_keys, num_records, destination + 1);
  }
  __syncthreads();
  const int64_t begin = shared_bounds[0];
  const int64_t end = shared_bounds[1];
  if (begin == end) {
    return;
  }
  uint32_t destination_cursor = destination;
  const int head = destination_cursor % num_heads;
  destination_cursor /= num_heads;
  const int pixel = destination_cursor % (height * width);
  const int local_batch = destination_cursor / (height * width);
  const int global_batch = batch_begin + local_batch;
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    scalar_t accumulator = static_cast<scalar_t>(0);
    for (int64_t position = begin; position < end; ++position) {
      const uint32_t source_index = sorted_sources[position];
      const int64_t grad_output_index =
          static_cast<int64_t>(grad_output_bases[source_index]) + channel;
      accumulator += grad_output[grad_output_index]
          * cached_attention[source_index] * bilinear_weights[source_index];
    }
    const int64_t output_index =
        (((static_cast<int64_t>(global_batch) * spatial_size + level_start + pixel) * num_heads + head)
         * channels + channel);
    grad_value[output_index] = accumulator;
  }
}

}  // namespace

torch::Tensor deterministic_msda_grad_value_cuda(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_locations,
    const torch::Tensor& attention_weights,
    const torch::Tensor& grad_output,
    int64_t batch_step) {
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be CUDA");
  TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be CUDA");
  TORCH_CHECK(sampling_locations.is_cuda(), "sampling_locations must be CUDA");
  TORCH_CHECK(attention_weights.is_cuda(), "attention_weights must be CUDA");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(sampling_locations.is_contiguous(), "sampling_locations must be contiguous");
  TORCH_CHECK(attention_weights.is_contiguous(), "attention_weights must be contiguous");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  TORCH_CHECK(value.scalar_type() == torch::kFloat || value.scalar_type() == torch::kDouble,
              "only float32/float64 are supported");
  TORCH_CHECK(value.scalar_type() == sampling_locations.scalar_type()
              && value.scalar_type() == attention_weights.scalar_type()
              && value.scalar_type() == grad_output.scalar_type(), "dtype mismatch");
  TORCH_CHECK(batch_step > 0, "batch_step must be positive");

  const c10::cuda::CUDAGuard device_guard(value.device());
  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_query = sampling_locations.size(1);
  const int num_levels = sampling_locations.size(3);
  const int num_points = sampling_locations.size(4);
  const int actual_batch_step = std::min<int64_t>(batch, batch_step);

  auto shapes_cpu = spatial_shapes.to(torch::kCPU);
  auto starts_cpu = level_start_index.to(torch::kCPU);
  const int64_t* shapes = shapes_cpu.data_ptr<int64_t>();
  const int64_t* starts = starts_cpu.data_ptr<int64_t>();
  auto grad_value = torch::zeros_like(value);

  const int64_t max_records = static_cast<int64_t>(actual_batch_step) * num_query
      * num_heads * num_points * 4;
  TORCH_CHECK(max_records <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              "record count exceeds CUB int limit");
  int64_t max_level_pixels = 0;
  for (int level = 0; level < num_levels; ++level) {
    max_level_pixels = std::max(max_level_pixels, shapes[level * 2] * shapes[level * 2 + 1]);
  }
  const uint64_t max_destinations = static_cast<uint64_t>(actual_batch_step)
      * max_level_pixels * num_heads;
  TORCH_CHECK(max_destinations < static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
              "destination count exceeds uint32 range");
  auto key_options = value.options().dtype(torch::kInt);
  auto keys_in = torch::empty({max_records}, key_options);
  auto keys_out = torch::empty({max_records}, key_options);
  auto sources_in = torch::empty({max_records}, key_options);
  auto sources_out = torch::empty({max_records}, key_options);
  auto bilinear_weights = torch::empty({max_records}, value.options());
  auto cached_attention = torch::empty({max_records}, value.options());
  auto grad_output_bases = torch::empty({max_records}, key_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(value.get_device());
  size_t temp_storage_bytes = 0;
  const int end_bit = ceil_log2_u64(max_destinations + 1);
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      reinterpret_cast<uint32_t*>(keys_in.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(keys_out.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(sources_in.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(sources_out.data_ptr<int32_t>()),
      static_cast<int>(max_records),
      0,
      end_bit,
      stream));
  auto temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)}, value.options().dtype(torch::kUInt8));

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "deterministic_msda_grad_value_cuda", [&] {
    for (int level = 0; level < num_levels; ++level) {
      const int height = static_cast<int>(shapes[level * 2]);
      const int width = static_cast<int>(shapes[level * 2 + 1]);
      const int level_start = static_cast<int>(starts[level]);
      for (int batch_begin = 0; batch_begin < batch; batch_begin += actual_batch_step) {
        const int chunk_batch = std::min(actual_batch_step, batch - batch_begin);
        const int64_t num_records = static_cast<int64_t>(chunk_batch) * num_query
            * num_heads * num_points * 4;
        const uint32_t num_destinations = static_cast<uint32_t>(
            static_cast<uint64_t>(chunk_batch) * height * width * num_heads);
        const uint32_t invalid_key = num_destinations;
        const int level_end_bit = ceil_log2_u64(static_cast<uint64_t>(num_destinations) + 1);
        const int key_blocks = static_cast<int>((num_records + kThreads - 1) / kThreads);
        generate_level_pairs_kernel<scalar_t><<<key_blocks, kThreads, 0, stream>>>(
            reinterpret_cast<uint32_t*>(keys_in.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(sources_in.data_ptr<int32_t>()),
            bilinear_weights.data_ptr<scalar_t>(),
            cached_attention.data_ptr<scalar_t>(),
            reinterpret_cast<uint32_t*>(grad_output_bases.data_ptr<int32_t>()),
            num_records,
            batch_begin,
            num_query,
            num_heads,
            channels,
            num_levels,
            num_points,
            level,
            height,
            width,
            invalid_key,
            sampling_locations.data_ptr<scalar_t>(),
            attention_weights.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr<uint8_t>(),
            temp_storage_bytes,
            reinterpret_cast<uint32_t*>(keys_in.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(keys_out.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(sources_in.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(sources_out.data_ptr<int32_t>()),
            static_cast<int>(num_records),
            0,
            level_end_bit,
            stream));
        reduce_level_pairs_kernel<scalar_t><<<num_destinations, kReduceThreads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(keys_out.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(sources_out.data_ptr<int32_t>()),
            num_records,
            batch_begin,
            spatial_size,
            num_query,
            num_heads,
            channels,
            num_levels,
            num_points,
            level,
            level_start,
            height,
            width,
            bilinear_weights.data_ptr<scalar_t>(),
            cached_attention.data_ptr<scalar_t>(),
            reinterpret_cast<const uint32_t*>(grad_output_bases.data_ptr<int32_t>()),
            grad_output.data_ptr<scalar_t>(),
            grad_value.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
  });

  return grad_value;
}

std::vector<torch::Tensor> deterministic_msda_query_grads_cuda(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_locations,
    const torch::Tensor& attention_weights,
    const torch::Tensor& grad_output) {
  TORCH_CHECK(value.is_cuda() && spatial_shapes.is_cuda() && level_start_index.is_cuda(),
              "value and shape tensors must be CUDA");
  TORCH_CHECK(sampling_locations.is_cuda() && attention_weights.is_cuda() && grad_output.is_cuda(),
              "sampling, attention, and grad_output must be CUDA");
  TORCH_CHECK(value.is_contiguous() && spatial_shapes.is_contiguous()
              && level_start_index.is_contiguous() && sampling_locations.is_contiguous()
              && attention_weights.is_contiguous() && grad_output.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(value.scalar_type() == torch::kFloat || value.scalar_type() == torch::kDouble,
              "only float32/float64 are supported");
  TORCH_CHECK(value.scalar_type() == sampling_locations.scalar_type()
              && value.scalar_type() == attention_weights.scalar_type()
              && value.scalar_type() == grad_output.scalar_type(), "dtype mismatch");

  const c10::cuda::CUDAGuard device_guard(value.device());
  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_query = sampling_locations.size(1);
  const int num_levels = sampling_locations.size(3);
  const int num_points = sampling_locations.size(4);
  auto grad_sampling_locations = torch::zeros_like(sampling_locations);
  auto grad_attention_weights = torch::zeros_like(attention_weights);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(value.get_device());
  constexpr int warps_per_block = kThreads / 32;
  const int64_t total_warps = static_cast<int64_t>(batch) * num_query * num_heads;
  const int blocks = static_cast<int>((total_warps + warps_per_block - 1) / warps_per_block);

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "deterministic_msda_query_grads_cuda", [&] {
    deterministic_query_grads_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        value.data_ptr<scalar_t>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        batch,
        spatial_size,
        num_query,
        num_heads,
        channels,
        num_levels,
        num_points,
        grad_sampling_locations.data_ptr<scalar_t>(),
        grad_attention_weights.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {grad_sampling_locations, grad_attention_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "deterministic_msda_grad_value",
      &deterministic_msda_grad_value_cuda,
      "Deterministic MSDeformAttn grad_value (CUDA)",
      pybind11::arg("value"),
      pybind11::arg("spatial_shapes"),
      pybind11::arg("level_start_index"),
      pybind11::arg("sampling_locations"),
      pybind11::arg("attention_weights"),
      pybind11::arg("grad_output"),
      pybind11::arg("batch_step") = 1);
  module.def(
      "deterministic_msda_query_grads",
      &deterministic_msda_query_grads_cuda,
      "Deterministic MSDeformAttn sampling-location and attention gradients (CUDA)",
      pybind11::arg("value"),
      pybind11::arg("spatial_shapes"),
      pybind11::arg("level_start_index"),
      pybind11::arg("sampling_locations"),
      pybind11::arg("attention_weights"),
      pybind11::arg("grad_output"));
}
