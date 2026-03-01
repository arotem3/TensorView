#include "test.hpp"
using namespace tensor;

#ifdef TENSOR_USE_FFTW
int testSimple1d()
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

int testMany1d()
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

int testManyStrided2d()
{
   int n_failed = 0;

   Tensor<std::complex<double>, 4> base(4, 8, 2, 16);
   auto in = base(All{}, Range{0, 8, 2}, 0, Range{4, 16});
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

int testFullDimensionTransform()
{
   int n_failed = 0;

   // Test 2D FFT transforming all dimensions
   Matrix<std::complex<double>> in(8, 12);
   Matrix<std::complex<double>> out(8, 12);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0, 1); // transform all dimensions

   // Check unitarity
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
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Full 2D FFT not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Full 2D FFT is unitary!" << std::endl;
   }

   // Check inverse
   Matrix<std::complex<double>> inv(8, 12);
   ifftn(out, inv, 0, 1);

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Full 2D inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Full 2D inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int test3dFFT()
{
   int n_failed = 0;

   // Test 3D FFT
   Cube<std::complex<double>> in(4, 6, 8);
   Cube<std::complex<double>> out(4, 6, 8);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0, 1, 2); // transform all 3 dimensions

   // Check unitarity
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
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 3D FFT not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 3D FFT is unitary!" << std::endl;
   }

   // Check inverse
   Cube<std::complex<double>> inv(4, 6, 8);
   ifftn(out, inv, 0, 1, 2);

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 3D inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 3D inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int testNonPowerOfTwo()
{
   int n_failed = 0;

   // Test with non-power-of-2 sizes
   Vector<std::complex<double>> in(15);
   Vector<std::complex<double>> out(15);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0);

   // Check unitarity
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
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Non-power-of-2 FFT not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Non-power-of-2 FFT is unitary!" << std::endl;
   }

   // Check inverse
   Vector<std::complex<double>> inv(15);
   ifftn(out, inv, 0);

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Non-power-of-2 inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Non-power-of-2 inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int testSmallTensor()
{
   int n_failed = 0;

   // Test with very small tensor
   Vector<std::complex<double>> in(2);
   Vector<std::complex<double>> out(2);

   in[0] = std::complex<double>(1.0, 0.0);
   in[1] = std::complex<double>(0.0, 1.0);

   fftn(in, out, 0);

   // Check inverse
   Vector<std::complex<double>> inv(2);
   ifftn(out, inv, 0);

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Small tensor (size 2) inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Small tensor (size 2) inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int testDCSignal()
{
   int n_failed = 0;

   // Test DC signal (constant value)
   Vector<std::complex<double>> in(16);
   Vector<std::complex<double>> out(16);

   for (auto &val : in)
      val = std::complex<double>(1.0, 0.0);

   fftn(in, out, 0);

   // For a DC signal, FFT should be zero everywhere except first element
   // First element should be N (scaled by normalization factor 1/sqrt(N))
   const double expected_dc = std::sqrt(16.0);
   const double dc_error = std::abs(out[0] - std::complex<double>(expected_dc, 0.0));

   double max_other_error = 0.0;
   for (index_t i = 1; i < out.size(); i++)
   {
      max_other_error = std::max(max_other_error, std::abs(out[i]));
   }

   if (dc_error > 1e-12 || max_other_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " DC signal FFT incorrect!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " DC signal FFT correct!" << std::endl;
   }

   return n_failed;
}

int testFContiguous2D()
{
   int n_failed = 0;

   // Test F-contiguous 2D FFT
   FTensor<std::complex<double>, 2> in(6, 8);
   FTensor<std::complex<double>, 2> out(6, 8);

   for (auto &val : in)
      val = std::complex<double>(rand() / double(RAND_MAX), rand() / double(RAND_MAX));

   fftn(in, out, 0, 1);

   // Check unitarity
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
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " F-contiguous 2D FFT not unitary!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " F-contiguous 2D FFT is unitary!" << std::endl;
   }

   // Check inverse
   FTensor<std::complex<double>, 2> inv(6, 8);
   ifftn(out, inv, 0, 1);

   double max_error = 0.0;
   for (index_t i = 0; i < in.size(); i++)
   {
      const double error = std::abs(in[i] - inv[i]);
      max_error = std::max(max_error, error);
   }

   if (max_error > 1e-12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " F-contiguous 2D inverse FFT failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " F-contiguous 2D inverse FFT passed!" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += testSimple1d();
   n_failed += testMany1d();
   n_failed += testManyStrided2d();
   n_failed += testFullDimensionTransform();
   n_failed += test3dFFT();
   n_failed += testNonPowerOfTwo();
   n_failed += testSmallTensor();
   n_failed += testDCSignal();
   n_failed += testFContiguous2D();

   PRINT_RESULT(n_failed);

   return n_failed;
}

#else
int main()
{
   return 0;
}
#endif
