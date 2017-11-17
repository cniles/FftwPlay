#include <iostream>
#include <fftw3.h>
#include <math.h>

// calculates e^[val * I], storing the result into complex value result
void compute_exp_using_euler_identity(double val, fftw_complex* result) {
    result[0][0] = cos(val);
    result[0][1] = sin(val);
}

// multiplies two complex numbers
void complex_mul(double* a, double* b, double* result) {
    result[0] = a[0] * b[0] - a[1] * b[1];
    result[1] = a[0] * b[1] + a[1] * b[0];
}

// performs the inverse DFT as described by eq 7.49, 7.50, p198 Hanna & Rowland
void inverse_transform(int t, fftw_complex *out, int n, fftw_complex* result) {
    result[0][0] = 0;
    result[0][1] = 0;
    for (int j = 0; j < n; ++j) {
        double *coef = out[j];
        double expon = (2 *M_PI * j / n) * t;
        fftw_complex z;
        compute_exp_using_euler_identity(expon, &z);
        fftw_complex zz;
        complex_mul(coef, z, zz);
        result[0][0] += zz[0];
        result[0][1] += zz[1];
    }
}

int main(int argc, char** argv) {
    int n = 100;

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

    p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_PATIENT);

    for (int i = 0; i < n; ++i) {
        in[i][0] = sin(i/(double)n * 2*M_PI);
        in[i][1] = 0.0;
    }

    fftw_execute(p);

    for (int i = 0; i < n; ++i) {
        double magnitude = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
        fftw_complex inv;
        inverse_transform(i, out, n, &inv);

        // in[i] ~= inv / n (ffwt produces un-normalized DFT, hence scaling by n^-1)
        std::cout << in[i][0] << "  ->  " << out[i][0] << " + " << out[i][1] << "i" << " -> " << inv[0] / n << " + " << inv[1] / n << "i" << std::endl;
    }

    // be good
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}
