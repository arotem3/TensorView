#pragma once
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

   template <typename TensorLikeIn, typename TensorLikeOut, std::ranges::random_access_range IntSequence>
   auto makeFFTWArrayView(TensorLikeIn &&in, TensorLikeOut &&out, IntSequence &&dims)
   {
      using InTraits = details::TensorTraits<std::remove_cvref_t<TensorLikeIn>>;
      using OutTraits = details::TensorTraits<std::remove_cvref_t<TensorLikeOut>>;

#ifdef TENSOR_USE_CUDA
      static_assert(InTraits::memorySpace() != MemorySpace::Device,
                    "makeFFTWArrayView does not support device memory for input tensor.");
      static_assert(OutTraits::memorySpace() != MemorySpace::Device,
                    "makeFFTWArrayView does not support device memory for output tensor.");
#endif

      constexpr index_t num_dims = InTraits::shape_traits::numDims();
      static_assert(num_dims == OutTraits::shape_traits::numDims(),
                    "Input and output tensors must have the same number of dimensions.");

      using InType = typename InTraits::value_type;
      using OutType = typename OutTraits::value_type;

      using GuruLayoutType = FFTWGuruLayout<InType, OutType>;
      using ResultType = std::optional<GuruLayoutType>;

      if (dims.size() == 0)
         return ResultType{std::nullopt};

      TENSOR_CHECK(
          [&]()
          {
             // check valid number of dims
             if (dims.size() > num_dims)
                return false;

             // check sorted range
             if (dims[0] >= num_dims)
                return false;

             for (index_t i = 1; i < dims.size(); ++i)
             {
                if (dims[i] <= dims[i - 1] || dims[i] >= num_dims)
                   return false;
             }

             return true;
          }(),
          printf("dims must be a unique sorted sequence of dimension indices in [0, %jd) with no more than %jd "
                 "elements.\n",
                 static_cast<uintmax_t>(num_dims), static_cast<uintmax_t>(num_dims)));

      using InStrided = RawSubView<InType, num_dims, InTraits::container_traits::memorySpace()>;
      using OutStrided = RawSubView<OutType, num_dims, OutTraits::container_traits::memorySpace()>;

      InStrided in_view = in;
      OutStrided out_view = out;

      auto in_layout = in_view.shape();
      auto out_layout = out_view.shape();

      bool tdims[num_dims] = {false};

      GuruLayoutType guru;
      guru.rank = static_cast<int>(dims.size());
      guru.dims.reset(new fftw_iodim64[guru.rank]);
      for (int i = 0; i < guru.rank; ++i)
      {
         const index_t dim = dims[i];
         guru.dims[i].n = static_cast<ptrdiff_t>(in_layout.shape(dim));
         guru.dims[i].is = static_cast<ptrdiff_t>(in_layout.stride(dim));
         guru.dims[i].os = static_cast<ptrdiff_t>(out_layout.stride(dim));

         tdims[dim] = true;
      }

      std::vector<index_t> remaining_dims;
      for (index_t i = 0; i < num_dims; ++i)
      {
         if (!tdims[i])
            remaining_dims.push_back(i);
      }

      if (remaining_dims.size() == 0)
      {
         guru.howmany_rank = 1;
         guru.howmany_dims.reset(new fftw_iodim64[1]);
         guru.howmany_dims[0].n = 1;
         guru.howmany_dims[0].is = 0;
         guru.howmany_dims[0].os = 0;
      }
      else
      {
         guru.howmany_rank = static_cast<int>(remaining_dims.size());
         guru.howmany_dims.reset(new fftw_iodim64[guru.howmany_rank]);

         for (int i = 0; i < guru.howmany_rank; ++i)
         {
            const index_t dim = remaining_dims[i];
            guru.howmany_dims[i].n = static_cast<ptrdiff_t>(in_layout.shape(dim));
            guru.howmany_dims[i].is = static_cast<ptrdiff_t>(in_layout.stride(dim));
            guru.howmany_dims[i].os = static_cast<ptrdiff_t>(out_layout.stride(dim));
         }
      }

      guru.in = InTraits::container(in).data() + in_layout.offset();
      guru.out = OutTraits::container(out).data() + out_layout.offset();

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