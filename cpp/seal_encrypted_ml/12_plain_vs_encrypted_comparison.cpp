// 12_plain_vs_encrypted_comparison.cpp
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "examples.h"

using namespace std;
using namespace seal;

static double plain_dot(const vector<double> &x, const vector<double> &w, double b)
{
    double z = b;
    for (size_t i = 0; i < x.size(); i++)
        z += x[i] * w[i];
    return z;
}

void example_plain_vs_encrypted_comparison()
{
    print_example_banner("Example: Plain vs Encrypted Linear Inference (CKKS dot-product)");

    // --- CKKS parameters (stable + valid for 8192) ---
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // Total 200 bits: OK for 8192 at 128-bit security in SEAL
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

    SEALContext context(parms);
    if (!context.parameters_set())
    {
        throw logic_error("encryption parameters are not set correctly");
    }

    print_parameters(context);
    cout << endl;

    // --- Tools / Keys ---
    auto t_key0 = chrono::high_resolution_clock::now();

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();

    PublicKey public_key;
    keygen.create_public_key(public_key);

    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    auto t_key1 = chrono::high_resolution_clock::now();
    double keygen_ms = chrono::duration<double, milli>(t_key1 - t_key0).count();

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    const double scale = pow(2.0, 40);

    // --- Toy input ---
    vector<double> x = { 0.2, -0.4, 0.6, 0.1, -0.3 };
    vector<double> w = { 1.1, -0.7, 0.9, 0.4, -1.0 };
    double b = 0.05;

    // --- Plain reference ---
    auto t_plain0 = chrono::high_resolution_clock::now();
    double z_plain = plain_dot(x, w, b);
    auto t_plain1 = chrono::high_resolution_clock::now();

    double plain_ms = chrono::duration<double, milli>(t_plain1 - t_plain0).count();

    cout << fixed << setprecision(6);
    cout << "Plain z = w*x + b: " << z_plain << "\n\n";

    // --- Encrypted computation ---
    auto t_he0 = chrono::high_resolution_clock::now();

    // Encode + encrypt x
    Plaintext x_plain;
    encoder.encode(x, scale, x_plain);

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    // Encode weights (same scale to keep multiply_plain scale ~ 2^40 * 2^40 => rescale needed)
    Plaintext w_plain;
    encoder.encode(w, scale, w_plain);

    // prod = x * w (slotwise)
    Ciphertext prod;
    evaluator.multiply_plain(x_encrypted, w_plain, prod);
    evaluator.rescale_to_next_inplace(prod);
    prod.scale() = scale;

    // sum slots -> dot product in slot 0
    Ciphertext z_encrypted = prod;
    int n = static_cast<int>(x.size());
    for (int step = 1; step < n; step <<= 1)
    {
        Ciphertext rotated;
        evaluator.rotate_vector(z_encrypted, step, gal_keys, rotated);
        evaluator.add_inplace(z_encrypted, rotated);
    }

    // add bias (match parms_id + scale)
    Plaintext b_plain;
    encoder.encode(b, z_encrypted.scale(), b_plain);
    evaluator.mod_switch_to_inplace(b_plain, z_encrypted.parms_id());
    b_plain.scale() = z_encrypted.scale();
    evaluator.add_plain_inplace(z_encrypted, b_plain);

    // decrypt + decode
    Plaintext z_dec_plain;
    decryptor.decrypt(z_encrypted, z_dec_plain);

    vector<double> z_out;
    encoder.decode(z_dec_plain, z_out);

    auto t_he1 = chrono::high_resolution_clock::now();
    double he_ms = chrono::duration<double, milli>(t_he1 - t_he0).count();

    double z_he = z_out[0];
    double abs_err = fabs(z_he - z_plain);

    // --- Print comparison ---
    cout << "RESULTS\n";
    cout << "------------------------------------------------------------\n";
    cout << "KeyGen time (ms):              " << keygen_ms << "\n";
    cout << "Plain compute time (ms):       " << plain_ms << "\n";
    cout << "Encrypted compute time (ms):   " << he_ms << "\n";
    cout << "------------------------------------------------------------\n";
    cout << "Encrypted z (slot 0):          " << z_he << "\n";
    cout << "Abs error |z_he - z_plain|:    " << abs_err << "\n";
    cout << "------------------------------------------------------------\n";
    cout << "(This compares plain vs CKKS-encrypted dot product only.)\n";
}
