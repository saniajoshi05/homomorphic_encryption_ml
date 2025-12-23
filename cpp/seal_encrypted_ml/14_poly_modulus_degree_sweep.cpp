// 14_poly_modulus_degree_sweep.cpp
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
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

struct OneRun
{
    double keygen_ms = 0.0;
    double enc_ms = 0.0;
    double abs_err = 0.0;
    double z_plain = 0.0;
    double z_he = 0.0;
    size_t n = 0;
};

static size_t sum_bits(const vector<int> &v)
{
    size_t s = 0;
    for (int b : v)
        s += static_cast<size_t>(b);
    return s;
}

static int log2_scale_bits(double scale)
{
    return static_cast<int>(round(log2(scale)));
}

// Choose coeff modulus chain + scale depending on degree (must be VALID!)
static void choose_ckks_parms(size_t deg, vector<int> &mod_bits, double &scale)
{
    if (deg == 4096)
    {
        // Valid small chain for deg=4096 that supports ONE rescale
        // Total bits = 100 (usually safe at 128-bit security for 4096)
        mod_bits = { 40, 30, 30 };
        scale = pow(2.0, 30);
    }
    else
    {
        // Standard stable chain for 8192 and 16384
        mod_bits = { 60, 40, 40, 60 };
        scale = pow(2.0, 40);
    }
}

static OneRun run_once(size_t deg, size_t vec_len)
{
    OneRun r;
    r.n = vec_len;

    vector<int> mod_bits;
    double scale = 0.0;
    choose_ckks_parms(deg, mod_bits, scale);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(deg);
    parms.set_coeff_modulus(CoeffModulus::Create(deg, mod_bits));

    SEALContext context(parms);
    if (!context.parameters_set())
        throw logic_error("encryption parameters are not set correctly");

    // ---- Keys/tools ----
    auto t_key0 = chrono::high_resolution_clock::now();

    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();

    PublicKey pk;
    keygen.create_public_key(pk);

    GaloisKeys galk;
    keygen.create_galois_keys(galk);

    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);
    CKKSEncoder encoder(context);

    auto t_key1 = chrono::high_resolution_clock::now();
    r.keygen_ms = ms_since(t_key0, t_key1);

    // ---- Data (repeatable random) ----
    vector<double> x, w;
    make_random_vector(vec_len, x, -1.0, 1.0, 1234);
    make_random_vector(vec_len, w, -1.0, 1.0, 5678);

    // Bias
    double b = 0.05;

    // Keep consistent input scaling across degrees
    const double input_scale = 0.1;
    for (size_t i = 0; i < vec_len; i++)
    {
        x[i] *= input_scale;
        w[i] *= input_scale;
    }

    r.z_plain = plain_dot(x, w, b);

    // ---- Encrypted dot-product (standard path: multiply_plain + rescale) ----
    auto t_e0 = chrono::high_resolution_clock::now();

    Plaintext x_plain;
    encoder.encode(x, scale, x_plain);

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    Plaintext w_plain;
    encoder.encode(w, scale, w_plain);

    Ciphertext prod;
    evaluator.multiply_plain(x_encrypted, w_plain, prod);
    evaluator.rescale_to_next_inplace(prod);
    prod.scale() = scale;

    // Sum slots -> dot product in slot 0
    Ciphertext sum = prod;
    for (size_t step = 1; step < vec_len; step <<= 1)
    {
        Ciphertext rotated;
        evaluator.rotate_vector(sum, static_cast<int>(step), galk, rotated);
        evaluator.add_inplace(sum, rotated);
    }

    // Add bias (must match parms_id and scale)
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

    r.z_he = out[0];
    r.abs_err = fabs(r.z_he - r.z_plain);

    auto t_e1 = chrono::high_resolution_clock::now();
    r.enc_ms = ms_since(t_e0, t_e1);

    return r;
}

void example_poly_modulus_degree_sweep()
{
    print_example_banner("Example: Poly modulus degree sweep (CKKS dot product)");

    cout << "We sweep poly_modulus_degree.\n";
    cout << "Using n=256 for deg=4096 (smaller chain/noise headroom), and n=2048 for deg=8192/16384.\n";
    cout << "(n must be <= slots = poly_modulus_degree/2)\n\n";

    cout << "RESULTS\n";
    cout << "----------------------------------------------------------------------------------------------------------"
            "----\n";
    cout << "poly_deg  slots    n      coeff_bits   scale_bits   keygen_ms   encrypted_ms   abs_err_z      "
            "dec_z(slot0)      plain_z\n";
    cout << "----------------------------------------------------------------------------------------------------------"
            "----\n";

    vector<size_t> degrees = { 4096, 8192, 16384 };

    for (size_t deg : degrees)
    {
        try
        {
            size_t vec_len = (deg == 4096) ? 256 : 2048;

            vector<int> mod_bits;
            double scale = 0.0;
            choose_ckks_parms(deg, mod_bits, scale);

            size_t slots = deg / 2;
            size_t coeff_bits = sum_bits(mod_bits);
            int scale_bits = log2_scale_bits(scale);

            OneRun r = run_once(deg, vec_len);

            cout << deg << "      " << slots << "     " << r.n << "     " << coeff_bits << "        " << scale_bits
                 << "         " << r.keygen_ms << "     " << r.enc_ms << "        " << r.abs_err << "   " << r.z_he
                 << "         " << r.z_plain << "\n";
        }
        catch (const exception &e)
        {
            cout << deg << "      [EXCEPTION] " << e.what() << "\n";
        }
    }

    cout << "----------------------------------------------------------------------------------------------------------"
            "----\n";
    cout << "(Note: deg=4096 uses a smaller modulus chain and smaller scale; we also use smaller n to avoid noise "
            "exhaustion.)\n";
}
