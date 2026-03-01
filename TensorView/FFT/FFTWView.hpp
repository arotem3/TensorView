#pragma once
#include "TensorView/Access/StandardPattern.hpp"
#include "TensorView/Layouts/StridedLayout.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TView.hpp"

#ifdef TENSOR_USE_FFTW

namespace tensor
{
   /**
    * @brief A structure representing a view compatible with FFTW's guru64 interface.
    * Any tensor type convertible to a strided raw view can be adapted to this layout.
    */
   template <typename InType, typename OutType>
   class FFTWGuruLayout
   {
   public:
      using input_type = InType;
      using output_type = OutType;

      int rank;
      std::unique_ptr<fftw_iodim64[]> dims;
      int howmany_rank;
      std::unique_ptr<fftw_iodim64[]> howmany_dims;
      input_type *in;
      output_type *out;
   };

   namespace details
   {
      template <index_t NumDims, typename Layout, typename IndexSet>
      auto computeStrides(const AccessPattern<Layout, IndexSet> &pattern)
      {
         auto strided = makeStridedLayoutFrom<NumDims>(std::get<0>(unpackAccessPattern(pattern)));

         std::array<ptrdiff_t, NumDims> strides;
         for (index_t d = 0; d < NumDims; ++d)
            strides[d] = static_cast<ptrdiff_t>(strided.dimensions[d].stride);

         return strides;
      }
   } // namespace details

   template <typename TensorLikeIn, typename TensorLikeOut, std::ranges::random_access_range IntSequence>
   auto makeFFTWArrayView(TensorLikeIn &&in, TensorLikeOut &&out, IntSequence &&dims)
   {
      using InTensorType = std::remove_cvref_t<TensorLikeIn>;
      using OutTensorType = std::remove_cvref_t<TensorLikeOut>;
      using InTraits = details::TensorTraits<InTensorType>;
      using OutTraits = details::TensorTraits<OutTensorType>;

#ifdef TENSOR_USE_CUDA
      static_assert(InTraits::memorySpace() != MemorySpace::Device,
                    "makeFFTWArrayView does not support device memory for input tensor.");
      static_assert(OutTraits::memorySpace() != MemorySpace::Device,
                    "makeFFTWArrayView does not support device memory for output tensor.");
#endif

      constexpr index_t num_dims = InTensorType::numDims();
      static_assert(num_dims == OutTensorType::numDims(),
                    "Input and output tensors must have the same number of dimensions.");

      using InType = typename InTraits::value_type;
      using OutType = typename OutTraits::value_type;

      using GuruLayoutType = FFTWGuruLayout<InType, OutType>;
      using ResultType = std::optional<GuruLayoutType>;

      if (dims.size() == 0)
         return ResultType{std::nullopt};

      // Validate dims: non-empty, unique, sorted, in range
      TENSOR_CHECK(
          dims.size() <= num_dims && dims[0] < num_dims &&
              std::adjacent_find(dims.begin(), dims.end(), std::greater_equal<>()) == dims.end(),
          printf("dims must be a unique sorted sequence of dimension indices in [0, %jd) with no more than %jd "
                 "elements.\n",
                 static_cast<uintmax_t>(num_dims), static_cast<uintmax_t>(num_dims)));

      // Compute strides for input and output tensors
      const auto &in_shape = in.shape();
      const auto &out_shape = out.shape();
      auto in_strides = details::computeStrides<num_dims>(in_shape);
      auto out_strides = details::computeStrides<num_dims>(out_shape);

      // Setup transform dimensions
      GuruLayoutType guru;
      guru.rank = static_cast<int>(dims.size());
      guru.dims.reset(new fftw_iodim64[guru.rank]);

      bool tdims[num_dims] = {false};
      for (int i = 0; i < guru.rank; ++i)
      {
         const index_t dim = dims[i];
         guru.dims[i] = {static_cast<ptrdiff_t>(in_shape.shape(dim)), in_strides[dim], out_strides[dim]};
         tdims[dim] = true;
      }

      // Setup howmany dimensions (non-transform dimensions)
      std::vector<index_t> remaining_dims;
      remaining_dims.reserve(num_dims - dims.size());
      for (index_t d = 0; d < num_dims; ++d)
         if (!tdims[d])
            remaining_dims.push_back(d);

      if (remaining_dims.empty())
      {
         guru.howmany_rank = 1;
         guru.howmany_dims.reset(new fftw_iodim64[1]);
         guru.howmany_dims[0] = {1, 0, 0};
      }
      else
      {
         guru.howmany_rank = static_cast<int>(remaining_dims.size());
         guru.howmany_dims.reset(new fftw_iodim64[guru.howmany_rank]);
         for (int i = 0; i < guru.howmany_rank; ++i)
         {
            const index_t dim = remaining_dims[i];
            guru.howmany_dims[i] = {static_cast<ptrdiff_t>(in_shape.shape(dim)), in_strides[dim], out_strides[dim]};
         }
      }

      guru.in = InTraits::container(in).data() + in_shape.offset();
      guru.out = OutTraits::container(out).data() + out_shape.offset();

      return ResultType{std::move(guru)};
   }

   template <typename TensorLikeIn, typename TensorLikeOut, IndexLike... Dims>
   auto makeFFTWArrayView(TensorLikeIn &&in, TensorLikeOut &&out, Dims... dims)
   {
      return makeFFTWArrayView(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out),
                               std::array<index_t, sizeof...(Dims)>{static_cast<index_t>(dims)...});
   }
} // namespace tensor

#endif
