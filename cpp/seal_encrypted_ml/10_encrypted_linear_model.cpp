// 10_encrypted_linear_model.cpp

#include "examples.h"
#include <chrono>
#include <cmath>

using namespace std;
using namespace seal;
using clock_type = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

static double avg(const vector<double> &v)
{
    double s = 0.0;
    for (double x : v)
        s += x;
    return v.empty() ? 0.0 : s / v.size();
}

static double vmax(const vector<double> &v)
{
    double m = 0.0;
    for (double x : v)
        m = max(m, x);
    return m;
}

void example_encrypted_linear_model()
{
    print_example_banner("Example: Encrypted Linear Model (y = w*x + b)");

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

    double scale = pow(2.0, 40);
    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);

    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    // Toy ML-style data: y = w·x + b
    vector<double> x = { 1, 2, 3, 4, 5 };
    vector<double> w = { 0.5, -1.2, 2.0, 0.3, 1.1 };
    double b = 0.7;

    const int runs = 5;
    vector<double> t_plain_vec, t_enc_vec, t_mul_vec, t_rot_vec, t_bias_vec, t_dec_vec, t_total_vec;
    vector<double> err_vec;

    t_plain_vec.reserve(runs);
    t_enc_vec.reserve(runs);
    t_mul_vec.reserve(runs);
    t_rot_vec.reserve(runs);
    t_bias_vec.reserve(runs);
    t_dec_vec.reserve(runs);
    t_total_vec.reserve(runs);
    err_vec.reserve(runs);

    double y_plain = 0.0;
    double last_dec = 0.0;
    for (int r = 0; r < runs; r++){ 
        auto t_plain_start = clock_type::now();
        y_plain = b;
        for (size_t i = 0; i < x.size(); i++)
            y_plain += w[i] * x[i];
        auto t_plain_end = clock_type::now();
        double t_plain_ms = ms(t_plain_end - t_plain_start).count();
        t_plain_vec.push_back(t_plain_ms);

        // Encode + Encrypt timing
        auto t_enc_start = clock_type::now();
        Plaintext x_plain;
        encoder.encode(x, scale, x_plain);

        Ciphertext x_encrypted;
        encryptor.encrypt(x_plain, x_encrypted);
        auto t_enc_end = clock_type::now();
        double t_enc_ms = ms(t_enc_end - t_enc_start).count();
        t_enc_vec.push_back(t_enc_ms);

        // Encode weights (not encrypted)
        Plaintext w_plain;
        encoder.encode(w, scale, w_plain);

        // Multiply + Rescale timing
        auto t_mul_start = clock_type::now();
        Ciphertext prod;
        evaluator.multiply_plain(x_encrypted, w_plain, prod);
        evaluator.rescale_to_next_inplace(prod);
        auto t_mul_end = clock_type::now();
        double t_mul_ms = ms(t_mul_end - t_mul_start).count();
        t_mul_vec.push_back(t_mul_ms);

        // Rotations + Adds timing
        auto t_rot_start = clock_type::now();
        Ciphertext sum = prod;
        for (int step = 1; step < (int)x.size(); step <<= 1)
        {
            Ciphertext rotated;
            evaluator.rotate_vector(sum, step, gal_keys, rotated);
            evaluator.add_inplace(sum, rotated);
        }
        auto t_rot_end = clock_type::now();
        double t_rot_ms = ms(t_rot_end - t_rot_start).count();
        t_rot_vec.push_back(t_rot_ms);

        // Add bias timing
        auto t_bias_start = clock_type::now();
        Plaintext b_plain;
        encoder.encode(b, sum.scale(), b_plain);
        evaluator.mod_switch_to_inplace(b_plain, sum.parms_id());
        evaluator.add_plain_inplace(sum, b_plain);
        auto t_bias_end = clock_type::now();
        double t_bias_ms = ms(t_bias_end - t_bias_start).count();
        t_bias_vec.push_back(t_bias_ms);

        // Decrypt + Decode timing
        auto t_dec_start = clock_type::now();
        Plaintext dec;
        decryptor.decrypt(sum, dec);

        vector<double> out;
        encoder.decode(dec, out);
        auto t_dec_end = clock_type::now();
        double t_dec_ms = ms(t_dec_end - t_dec_start).count();
        t_dec_vec.push_back(t_dec_ms);

        last_dec = out[0];
        double err = fabs(y_plain - out[0]);
        err_vec.push_back(err);

        double t_total = t_enc_ms + t_mul_ms + t_rot_ms + t_bias_ms + t_dec_ms;
        t_total_vec.push_back(t_total);
    }

    // Print one representative result (last run)
    cout << "Plaintext y = w*x + b: " << y_plain << endl;
    cout << "Decrypted y (slot 0): " << last_dec << endl;
    cout << "Absolute error (last run): " << fabs(y_plain - last_dec) << endl;

    cout << "\nAveraged over " << runs << " runs (ms):" << endl;
    cout << "  Plaintext dot product: " << avg(t_plain_vec) << endl;
    cout << "  Encode+Encrypt:        " << avg(t_enc_vec) << endl;
    cout << "  Multiply+Rescale:      " << avg(t_mul_vec) << endl;
    cout << "  Rotations+Adds:        " << avg(t_rot_vec) << endl;
    cout << "  Add bias:              " << avg(t_bias_vec) << endl;
    cout << "  Decrypt+Decode:        " << avg(t_dec_vec) << endl;
    cout << "  Total encrypted path:  " << avg(t_total_vec) << endl;

    cout << "\nError:" << endl;
    cout << "  Avg abs error: " << avg(err_vec) << endl;
    cout << "  Max abs error: " << vmax(err_vec) << endl;
}
    
    
    
