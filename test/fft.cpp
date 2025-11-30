#include "test.hpp"
using namespace tensor;

#ifdef TENSOR_USE_FFTW
int test_simple1d()
{
   int n_failed = 0;

   Vector<std::complex<double>> in(16);
   Vector<std::complex<double>> out(16);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0); // transform along dim 0

   double nrm_in = 0.0;
   for (const auto &val : in)
      nrm_in += std::norm(val);
   nrm_in = std::sqrt(nrm_in);

   double nrm_out = 0.0;
   for (const auto &val : out)
      nrm_out += std::norm(val);
   nrm_out = std::sqrt(nrm_out);

   const double rel_error = std::abs(nrm_in - nrm_out) / nrm_in;

   if (rel_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 1D FFT not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 1D FFT is unitary!" << std::endl;
   }

   Vector<std::complex<double>> inv(16);
   ifftn(out, inv, 0); // inverse transform along dim 0

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      if (error > max_error)
         max_error = error;
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 1D Inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 1D Inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int test_many1d()
{
   int n_failed = 0;

   CTensor<std::complex<double>, 2> in(4, 16);
   CTensor<std::complex<double>, 2> out(4, 16);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 1); // transform along dim 1

   bool correct_norms = true;
   for (index_t i = 0; i < in.shape(0); i++)
   {
      auto in_row = in(i, All{});
      auto out_row = out(i, All{});

      double nrm_in = 0.0;
      for (const auto &val : in_row)
         nrm_in += std::norm(val);
      nrm_in = std::sqrt(nrm_in);

      double nrm_out = 0.0;
      for (const auto &val : out_row)
         nrm_out += std::norm(val);
      nrm_out = std::sqrt(nrm_out);

      const double rel_error = std::abs(nrm_in - nrm_out) / nrm_in;

      if (rel_error > 1e-12)
         correct_norms = false;
   }

   if (!correct_norms)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Many 1D FFTs not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Many 1D FFTs are unitary!" << std::endl;
   }

   FTensor<std::complex<double>, 2> inv(4, 16); // use Fortran order for variety
   ifftn(out, inv, 1);                          // inverse transform along dim 1

   double error = 0.0;
   for (index_t i = 0; i < in.shape(0); i++)
   {
      for (index_t j = 0; j < in.shape(1); j++)
      {
         error = std::max(error, std::abs(in(i, j) - inv(i, j)));
      }
   }

   if (error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Many 1D Inverse FFTs failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Many 1D Inverse FFTs passed!" << std::endl;
   }

   return n_failed;
}

int test_many2d_strided()
{
   int n_failed = 0;

   Tensor<std::complex<double>, 4> base(4, 8, 2, 16);
   auto in = base(All{}, Span{0, 8, 2}, 0, Span{4, 16});
   auto out = makeTensorLike(in);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0, 2); // transform along dims 0 and 2

   // check unitarity
   bool correct_norms = true;
   for (index_t i = 0; i < in.shape(1); i++)
   {
      auto in_slice = in(All{}, i, All{});
      double nrm_in = 0.0;
      for (const auto &val : in_slice)
         nrm_in += std::norm(val);
      nrm_in = std::sqrt(nrm_in);

      auto out_slice = out(All{}, i, All{});
      double nrm_out = 0.0;
      for (const auto &val : out_slice)
         nrm_out += std::norm(val);
      nrm_out = std::sqrt(nrm_out);

      const double rel_error = std::abs(nrm_in - nrm_out) / nrm_in;

      if (rel_error > 1e-12)
         correct_norms = false;
   }

   if (!correct_norms)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Many 2D strided FFTs not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Many 2D strided FFTs are unitary!" << std::endl;
   }

   // compare against contiguous copy
   CTensor<std::complex<double>, 3> in_copy = in;
   CTensor<std::complex<double>, 3> out_copy = makeTensorLike(in_copy);
   fftn(in_copy, out_copy, 0, 2);

   double max_error = 0.0;
   for (index_t i = 0; i < in_copy.size(); i++)
   {
      const double error = std::abs(out[i] - out_copy[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Many 2D strided FFTs differ from contiguous FFT!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Many 2D strided FFTs match contiguous FFT!" << std::endl;
   }

   // check inverse
   auto inv = makeTensorLike(in);
   ifftn(out, inv, 0, 2);

   max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Many 2D inverse FFTs failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Many 2D inverse FFTs passed!" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += test_simple1d();
   n_failed += test_many1d();
   n_failed += test_many2d_strided();

   PRINT_RESULT(n_failed);

   return n_failed;
}

#else
int main()
{
   return 0;
}
#endif
