#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

double f(int v)
{
    return 2.0 * v + 1.0;
}

double g(double v)
{
    return 1.0 / (v * v + 1.0);
}

struct functor
{
    double operator()(int v) const
    {
        return f(v);
    }
};

int main()
{
    Tensor<int, 2> x(2, 3);
    Tensor<double, 2> y(2, 3);
    Tensor<double, 2> z(2, 3);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            x(i, j) = 3 * i + j;
            y(i, j) = f(x(i, j));
            z(i, j) = g(y(i, j));
        }
    }

    auto fx_lambda = transform([](int v) { return f(v); }, x);
    auto fx_pointer = transform(f, x);
    auto fx_object = transform(functor(), x);

    std::cout << "DEBUG: Type information for transformed tensors:" << std::endl;
    std::cout << typeid(fx_lambda).name() << std::endl;
    std::cout << typeid(fx_pointer).name() << std::endl;
    std::cout << typeid(fx_object).name() << std::endl;

    static_assert(std::is_same_v<typename decltype(fx_lambda)::value_type, double>, "expected double for value_type");
    static_assert(std::is_same_v<typename decltype(fx_pointer)::value_type, double>, "expected double for value_type");
    static_assert(std::is_same_v<typename decltype(fx_object)::value_type, double>, "expected double for value_type");

    int fails = 0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (y(i, j) != fx_lambda(i, j))
            {
                std::cout << "transform() with lambda: Mismatch at (" << i << ", " << j << "): "
                          << y(i, j) << " != " << fx_lambda(i, j) << std::endl;
                ++fails;
            }
            if (y(i, j) != fx_pointer(i, j))
            {
                std::cout << "transform() with function pointer: Mismatch at (" << i << ", " << j << "): "
                          << y(i, j) << " != " << fx_pointer(i, j) << std::endl;
                ++fails;
            }
            if (y(i, j) != fx_object(i, j))
            {
                std::cout << "transform() with object: Mismatch at (" << i << ", " << j << "): "
                          << y(i, j) << " != " << fx_object(i, j) << std::endl;
                ++fails;
            }
        }
    }

    // composition test
    auto gfy_lambda = transform([] (double v) { return g(v); }, fx_lambda);
    auto gfy_pointer = transform(g, fx_pointer);

    std::cout << typeid(gfy_lambda).name() << std::endl;
    std::cout << typeid(gfy_pointer).name() << std::endl;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (z(i, j) != gfy_lambda(i, j))
            {
                std::cout << "transform() with lambda composition: Mismatch at (" << i << ", " << j << "): "
                          << z(i, j) << " != " << gfy_lambda(i, j) << std::endl;
                ++fails;
            }
            if (z(i, j) != gfy_pointer(i, j))
            {
                std::cout << "transform() with function pointer composition: Mismatch at (" << i << ", " << j << "): "
                          << z(i, j) << " != " << gfy_pointer(i, j) << std::endl;
                ++fails;
            }
        }
    }

    if (fails == 0)
    {
        std::cout << "Transform test passed successfully!" << std::endl;
    }

    return fails;
}