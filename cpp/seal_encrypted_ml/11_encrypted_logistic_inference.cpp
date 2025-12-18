// 11_encrypted_logistic_inference.cpp
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "examples.h"

using namespace std;
using namespace seal;

static inline void modswitch_plain_to_match(const Ciphertext &ct, Plaintext &pt, Evaluator &evaluator)
{
    if (pt.parms_id() != ct.parms_id())
    {
        evaluator.mod_switch_to_inplace(pt, ct.parms_id());
    }
}

struct RunResult
{
    double y_dec = 0.0;
    double abs_err = 0.0;
    double ms = 0.0;
};

static RunResult run_linear_sigmoid_once(double alpha)
{
    RunResult r;

    // ---- CKKS params (known-good on your setup) ----
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // IMPORTANT: use a chain that SEAL accepts reliably for 8192 in your build
    // Total bits = 180
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40 }));

    const double scale = pow(2.0, 40);

    SEALContext context(parms);
    if (!context.parameters_set())
    {
        throw logic_error("encryption parameters are not set correctly");
    }

    // ---- Keys / tools ----
    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();

    PublicKey public_key;
    keygen.create_public_key(public_key);

    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    // ---- Toy logistic regression input ----
    vector<double> x = { 0.2, -0.4, 0.6, 0.1, -0.3 };
    vector<double> w = { 1.1, -0.7, 0.9, 0.4, -1.0 };
    double b = 0.05;

    // ---- Plain reference ----
    double z_plain = b;
    for (size_t i = 0; i < x.size(); i++)
        z_plain += w[i] * x[i];

    auto sigmoid = [](double t) {
        return 1.0 / (1.0 + std::exp(-t));
    };
    double y_true = sigmoid(z_plain);

    // Start timing
    auto t0 = chrono::high_resolution_clock::now();

    // ---- Encrypt x ----
    Plaintext x_plain;
    encoder.encode(x, scale, x_plain);

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    // ---- Encode weights at SCALE 1.0 (prevents scale blow-up in multiply_plain) ----
    Plaintext w_plain;
    encoder.encode(w, 1.0, w_plain);
    modswitch_plain_to_match(x_encrypted, w_plain, evaluator);

    // prod = x * w (slotwise), stays around scale ~ 2^40
    Ciphertext prod;
    evaluator.multiply_plain(x_encrypted, w_plain, prod);

    // Sum slots -> dot product lands in slot 0
    Ciphertext z_encrypted = prod;
    int n = static_cast<int>(x.size());
    for (int step = 1; step < n; step <<= 1)
    {
        Ciphertext rotated;
        evaluator.rotate_vector(z_encrypted, step, gal_keys, rotated);
        evaluator.add_inplace(z_encrypted, rotated);
    }

    // Add bias: must match parms_id + scale
    Plaintext b_plain;
    encoder.encode(b, z_encrypted.scale(), b_plain);
    modswitch_plain_to_match(z_encrypted, b_plain, evaluator);
    b_plain.scale() = z_encrypted.scale();
    evaluator.add_plain_inplace(z_encrypted, b_plain);

    // ---- Linear sigmoid approx: y ≈ 0.5 + alpha * z ----
    // Encode alpha at SCALE 1.0 so multiply_plain doesn't change the ciphertext scale.
    Plaintext alpha_plain;
    encoder.encode(alpha, 1.0, alpha_plain);
    modswitch_plain_to_match(z_encrypted, alpha_plain, evaluator);

    Ciphertext y_encrypted = z_encrypted;
    evaluator.multiply_plain_inplace(y_encrypted, alpha_plain);
    // No rescale here (keeps the chain stable with 180-bit modulus)

    // Add 0.5 (encode at the SAME scale as ciphertext)
    Plaintext half_plain;
    encoder.encode(0.5, y_encrypted.scale(), half_plain);
    modswitch_plain_to_match(y_encrypted, half_plain, evaluator);
    half_plain.scale() = y_encrypted.scale();
    evaluator.add_plain_inplace(y_encrypted, half_plain);

    // ---- Decrypt + decode ----
    Plaintext dec;
    decryptor.decrypt(y_encrypted, dec);

    vector<double> out;
    encoder.decode(dec, out);

    auto t1 = chrono::high_resolution_clock::now();
    r.ms = chrono::duration<double, std::milli>(t1 - t0).count();

    r.y_dec = out[0];
    r.abs_err = fabs(r.y_dec - y_true);
    return r;
}

void example_encrypted_logistic_inference()
{
    print_example_banner("Example: Encrypted Logistic Inference (linear sigmoid approx)");

    // Print parameters once
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40 }));

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    // Plain reference
    vector<double> x = { 0.2, -0.4, 0.6, 0.1, -0.3 };
    vector<double> w = { 1.1, -0.7, 0.9, 0.4, -1.0 };
    double b = 0.05;

    double z_plain = b;
    for (size_t i = 0; i < x.size(); i++)
        z_plain += w[i] * x[i];

    auto sigmoid = [](double t) {
        return 1.0 / (1.0 + std::exp(-t));
    };
    double y_true = sigmoid(z_plain);

    cout << "Plain z = w*x + b: " << z_plain << endl;
    cout << "Plain sigmoid(z):  " << y_true << endl << endl;

    cout << "alpha   dec_y           abs_err         time_ms\n";
    cout << "--------------------------------------------------------\n";

    vector<double> alphas = { 0.25, 0.5, 1.0 };

    for (double alpha : alphas)
    {
        try
        {
            RunResult r = run_linear_sigmoid_once(alpha);
            cout << alpha << "    " << r.y_dec << "    " << r.abs_err << "    " << r.ms << "\n";
        }
        catch (const exception &e)
        {
            cout << alpha << "    [EXCEPTION] " << e.what() << "\n";
        }
    }
}
