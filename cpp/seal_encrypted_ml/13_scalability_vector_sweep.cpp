// 13_scalability_vector_sweep.cpp
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "examples.h"

using namespace std;
using namespace seal;

static void make_random_vector(size_t n, vector<double> &v, double lo, double hi, uint32_t seed)
{
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(lo, hi);
    v.resize(n);
    for (size_t i = 0; i < n; i++)
        v[i] = dist(rng);
}

static double plain_dot(const vector<double> &x, const vector<double> &w, double b)
{
    double z = b;
    for (size_t i = 0; i < x.size(); i++)
        z += x[i] * w[i];
    return z;
}

static double ms_since(
    const chrono::high_resolution_clock::time_point &t0, const chrono::high_resolution_clock::time_point &t1)
{
    return chrono::duration<double, milli>(t1 - t0).count();
}

void example_scalability_vector_sweep()
{
    print_example_banner("Example: Scalability (vector size sweep) - CKKS dot product");

    // ---- CKKS parameters ----
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // Depth for: multiply_plain + rescale + rotates/adds + add_plain
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

    const double scale = pow(2.0, 40);

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    // ---- Keys / tools ----
    auto t_key0 = chrono::high_resolution_clock::now();

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();

    PublicKey public_key;
    keygen.create_public_key(public_key);

    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    auto t_key1 = chrono::high_resolution_clock::now();
    double keygen_ms = ms_since(t_key0, t_key1);

    // ---- Sweep settings ----
    // Must be <= CKKS slots = poly_modulus_degree/2 = 4096
    vector<size_t> sizes = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

    cout << "RESULTS (vector size sweep)\n";
    cout << string(100, '-') << "\n";

    cout << left << setw(8) << "n" << setw(14) << "plain_ms" << setw(16) << "encrypted_ms" << setw(14) << "abs_err_z"
         << setw(18) << "dec_z(slot0)" << setw(18) << "plain_z"
         << "\n";

    cout << string(100, '-') << "\n";

    for (size_t n : sizes)
    {
        // Random but repeatable vectors
        vector<double> x, w;
        make_random_vector(n, x, -1.0, 1.0, 1234);
        make_random_vector(n, w, -1.0, 1.0, 5678);
        double b = 0.05;

        // ---- Plain ----
        auto t_p0 = chrono::high_resolution_clock::now();
        double z_plain = plain_dot(x, w, b);
        auto t_p1 = chrono::high_resolution_clock::now();
        double plain_ms = ms_since(t_p0, t_p1);

        // ---- Encrypted dot product ----
        auto t_e0 = chrono::high_resolution_clock::now();

        Plaintext x_plain;
        encoder.encode(x, scale, x_plain);

        Ciphertext x_encrypted;
        encryptor.encrypt(x_plain, x_encrypted);

        // weights at SAME scale as x so multiply_plain has scale ~ 2^80 then rescale -> 2^40
        Plaintext w_plain;
        encoder.encode(w, scale, w_plain);

        Ciphertext prod;
        evaluator.multiply_plain(x_encrypted, w_plain, prod);
        evaluator.rescale_to_next_inplace(prod); // scale ~ 2^40 (approximately)

        // (Optional but helps consistency)
        prod.scale() = scale;

        // Sum slots -> dot product in slot 0
        Ciphertext sum = prod;
        for (size_t step = 1; step < n; step <<= 1)
        {
            Ciphertext rotated;
            evaluator.rotate_vector(sum, static_cast<int>(step), gal_keys, rotated);
            evaluator.add_inplace(sum, rotated);
        }

        // Add bias: must match parms_id + scale
        Plaintext b_plain;
        encoder.encode(b, sum.scale(), b_plain);
        evaluator.mod_switch_to_inplace(b_plain, sum.parms_id());
        b_plain.scale() = sum.scale();
        evaluator.add_plain_inplace(sum, b_plain);

        // Decrypt
        Plaintext dec;
        decryptor.decrypt(sum, dec);

        vector<double> out;
        encoder.decode(dec, out);
        double z_he = out[0];

        auto t_e1 = chrono::high_resolution_clock::now();
        double enc_ms = ms_since(t_e0, t_e1);

        double abs_err = fabs(z_he - z_plain);

        cout << left << setw(8) << n << setw(14) << plain_ms << setw(16) << enc_ms << setw(14) << abs_err << setw(18)
             << z_he << setw(18) << z_plain << "\n";
    }

    cout << string(100, '-') << "\n";
    cout << "KeyGen time (ms): " << keygen_ms << "\n";
    cout << "(Note: n must be <= 4096 for poly_modulus_degree=8192)\n";
}
