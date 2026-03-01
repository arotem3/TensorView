#pragma once
#include "TensorView/FFT/FFTWView.hpp"
#include "TensorView/Macros.hpp"

#ifdef TENSOR_USE_FFTW

namespace tensor::details
{
   template <typename TensorLikeIn, typename TensorLikeOut, std::ranges::random_access_range IntSequence>
   inline void fftnImpl(TensorLikeIn &&in, TensorLikeOut &&out, IntSequence &&dims, bool forward)
   {
      if (dims.size() == 0)
         return;

      auto guru_opt = makeFFTWArrayView(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out), dims);
      using LayoutType = decltype(guru_opt)::value_type;
      using input_type = typename LayoutType::input_type;
      using output_type = typename LayoutType::output_type;

      static_assert(std::is_same_v<input_type, fftw_complex> || std::is_same_v<input_type, std::complex<double>>,
                    "Input tensor value type must be fftw_complex or std::complex<double>.");
      static_assert(std::is_same_v<output_type, fftw_complex> || std::is_same_v<output_type, std::complex<double>>,
                    "Output tensor value type must be fftw_complex or std::complex<double>.");

      TENSOR_CHECK(guru_opt, printf("Failed to create FFTW guru layout view.\n"));
      auto &guru = *guru_opt;

      int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

      fftw_plan plan = fftw_plan_guru64_dft(guru.rank, guru.dims.get(), guru.howmany_rank, guru.howmany_dims.get(),
                                            reinterpret_cast<fftw_complex *>(guru.in),
                                            reinterpret_cast<fftw_complex *>(guru.out), sign, FFTW_ESTIMATE);

      TENSOR_CHECK(plan, printf("Failed to create FFTW plan.\n"));

      fftw_execute(plan);
      fftw_destroy_plan(plan);

      double nrm = 1.0;
      for (index_t i = 0; i < dims.size(); ++i)
      {
         nrm *= static_cast<double>(in.shape(dims[i]));
      }
      nrm = 1.0 / std::sqrt(nrm);

      for (auto &val : out)
      {
         val *= nrm;
      }
   }
} // namespace tensor::details

namespace tensor
{
   /**
    * @brief Computes the n-dimensional FFT of the input tensor-like object and stores the result in the output
    * tensor-like object. The dimensions over which to compute the FFT are specified by the dims sequence.
    * The output is normalized by 1/sqrt(N) where N is the product of the sizes of the transformed dimensions. The
    * resulting transform is unitary.
    */
   template <typename TensorLikeIn, typename TensorLikeOut, std::ranges::random_access_range IntSequence>
   inline void fftn(TensorLikeIn &&in, TensorLikeOut &&out, IntSequence &&dims)
   {
      details::fftnImpl(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out),
                        std::forward<IntSequence>(dims), true);
   }

   /**
    * @brief Computes the n-dimensional FFT of the input tensor-like object and stores the result in the output
    * tensor-like object. The dimensions over which to compute the FFT are specified by the dims sequence.
    * The output is normalized by 1/sqrt(N) where N is the product of the sizes of the transformed dimensions. The
    * resulting transform is unitary.
    */
   template <typename TensorLikeIn, typename TensorLikeOut, IndexLike... TransformDims>
   inline void fftn(TensorLikeIn &&in, TensorLikeOut &&out, TransformDims... dims)
   {
      details::fftnImpl(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out),
                        std::array<index_t, sizeof...(TransformDims)>{static_cast<index_t>(dims)...}, true);
   }

   /**
    * @brief Computes the n-dimensional inverse FFT of the input tensor-like object and stores the result in the output
    * tensor-like object. The dimensions over which to compute the inverse FFT are specified by the dims sequence.
    * The output is normalized by 1/sqrt(N) where N is the product of the sizes of the transformed dimensions. The
    * resulting transform is unitary.
    */
   template <typename TensorLikeIn, typename TensorLikeOut, std::ranges::random_access_range IntSequence>
   inline void ifftn(TensorLikeIn &&in, TensorLikeOut &&out, IntSequence &&dims)
   {
      details::fftnImpl(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out),
                        std::forward<IntSequence>(dims), false);
   }

   /**
    * @brief Computes the n-dimensional inverse FFT of the input tensor-like object and stores the result in the output
    * tensor-like object. The dimensions over which to compute the inverse FFT are specified by the dims sequence.
    * The output is normalized by 1/sqrt(N) where N is the product of the sizes of the transformed dimensions. The
    * resulting transform is unitary.
    */
   template <typename TensorLikeIn, typename TensorLikeOut, IndexLike... TransformDims>
   inline void ifftn(TensorLikeIn &&in, TensorLikeOut &&out, TransformDims... dims)
   {
      details::fftnImpl(std::forward<TensorLikeIn>(in), std::forward<TensorLikeOut>(out),
                        std::array<index_t, sizeof...(TransformDims)>{static_cast<index_t>(dims)...}, false);
   }
} // namespace tensor

#endif
