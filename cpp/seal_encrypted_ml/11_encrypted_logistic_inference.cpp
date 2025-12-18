// 11_encrypted_logistic_inference.cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "examples.h"

using namespace std;
using namespace seal;

static inline void modswitch_plain_to_match(const Ciphertext &ct, Plaintext &pt, Evaluator &evaluator)
{
    if (pt.parms_id() != ct.parms_id())
        evaluator.mod_switch_to_inplace(pt, ct.parms_id());
}

void example_encrypted_logistic_inference()
{
    try
    {
        print_example_banner("Example: Encrypted Logistic Inference (linear sigmoid approx)");

        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);

        // Two rescales total (after x*w and after k*z), so 4 primes is enough.
        // IMPORTANT: end with 40 so rescale keeps scale sane.
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40 }));

        const double scale = pow(2.0, 40);

        SEALContext context(parms);
        print_parameters(context);
        cout << endl;

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

        vector<double> x = { 0.2, -0.4, 0.6, 0.1, -0.3 };
        vector<double> w = { 1.1, -0.7, 0.9, 0.4, -1.0 };
        double b = 0.05;

        double z_plain = b;
        for (size_t i = 0; i < x.size(); i++)
            z_plain += w[i] * x[i];

        auto sigmoid = [](double t) {
            return 1.0 / (1.0 + std::exp(-t));
        };
        double y_sigmoid_plain = sigmoid(z_plain);

        cout << "Plain z = w*x + b: " << z_plain << endl;
        cout << "Plain sigmoid(z):  " << y_sigmoid_plain << endl << endl;

        // Encrypt x
        Plaintext x_plain;
        encoder.encode(x, scale, x_plain);

        Ciphertext x_encrypted;
        encryptor.encrypt(x_plain, x_encrypted);

        // Encode weights at SAME scale (avoid quantizing to zero)
        Plaintext w_plain;
        encoder.encode(w, scale, w_plain);
        modswitch_plain_to_match(x_encrypted, w_plain, evaluator);

        // prod = x * w  (scale 2^80)
        Ciphertext prod;
        evaluator.multiply_plain(x_encrypted, w_plain, prod);

        // rescale -> back near 2^40
        evaluator.rescale_to_next_inplace(prod);
        prod.scale() = scale;

        // Sum slots -> dot product in slot 0
        Ciphertext z_encrypted = prod;
        int n = static_cast<int>(x.size());
        for (int step = 1; step < n; step <<= 1)
        {
            Ciphertext rotated;
            evaluator.rotate_vector(z_encrypted, step, gal_keys, rotated);
            evaluator.add_inplace(z_encrypted, rotated);
        }

        // Add bias
        Plaintext b_plain;
        encoder.encode(b, scale, b_plain);
        modswitch_plain_to_match(z_encrypted, b_plain, evaluator);
        b_plain.scale() = z_encrypted.scale();
        evaluator.add_plain_inplace(z_encrypted, b_plain);

        // Linear sigmoid approx: y = 0.5 + k*z
        // pick alpha to keep things in range; start with 0.5
        const double alpha = 0.25;
        const double k = 0.25 * alpha;

        Plaintext k_plain;
        encoder.encode(k, scale, k_plain);
        modswitch_plain_to_match(z_encrypted, k_plain, evaluator);

        Ciphertext y_encrypted;
        evaluator.multiply_plain(z_encrypted, k_plain, y_encrypted);

        // rescale again (second and final rescale)
        evaluator.rescale_to_next_inplace(y_encrypted);
        y_encrypted.scale() = scale;

        // +0.5
        Plaintext c0_plain;
        encoder.encode(0.5, scale, c0_plain);
        modswitch_plain_to_match(y_encrypted, c0_plain, evaluator);
        c0_plain.scale() = y_encrypted.scale();
        evaluator.add_plain_inplace(y_encrypted, c0_plain);

        // Decrypt
        Plaintext dec;
        decryptor.decrypt(y_encrypted, dec);

        vector<double> out;
        encoder.decode(dec, out);

        cout << "Decrypted approx sigmoid (slot 0): " << out[0] << endl;
        cout << "Abs error vs true sigmoid:         " << fabs(out[0] - y_sigmoid_plain) << endl;
        cout << "Note: linear approx; alpha=" << alpha << " (try 0.25 or 1.0)" << endl;
    }
    catch (const std::exception &e)
    {
        cout << "\n[EXCEPTION] " << e.what() << endl;
    }
}
