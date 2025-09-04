// https://oi-wiki.org/math/numerical/newton/

double sqrt_newton(double n) {
  constexpr static double eps = 1E-15;
  double x = 1;
  while (true) {
    double nx = (x + n / x) / 2;
    if (abs(x - nx) < eps) break;
    x = nx;
  }
  return x;
}