// 9_encrypted_linear_regression.cpp
#include "examples.h"

using namespace std;
using namespace seal;

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

    // Plaintext dot product baseline
    double y_plain = b;
    for (size_t i = 0; i < x.size(); i++)
        y_plain += w[i] * x[i];

    cout << "Plaintext y = w*x + b: " << y_plain << endl;

    // Encode + encrypt x
    Plaintext x_plain;
    encoder.encode(x, scale, x_plain);

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    Plaintext w_plain;
    encoder.encode(w, scale, w_plain);

    // Multiply slot-wise: w*x
    Ciphertext prod;
    evaluator.multiply_plain(x_encrypted, w_plain, prod);
    evaluator.rescale_to_next_inplace(prod);

    // Sum slots (dot product into slot 0)
    Ciphertext sum = prod;
    for (int step = 1; step < (int)x.size(); step <<= 1)
    {
        Ciphertext rotated;
        evaluator.rotate_vector(sum, step, gal_keys, rotated);
        evaluator.add_inplace(sum, rotated);
    }

    // Add bias (match parms and scale)
    Plaintext b_plain;
    encoder.encode(b, sum.scale(), b_plain);
    evaluator.mod_switch_to_inplace(b_plain, sum.parms_id());
    evaluator.add_plain_inplace(sum, b_plain);

    // Decrypt + decode
    Plaintext dec;
    decryptor.decrypt(sum, dec);

    vector<double> out;
    encoder.decode(dec, out);

    cout << "Decrypted y (slot 0): " << out[0] << endl;
    cout << "Absolute error: " << fabs(y_plain - out[0]) << endl;
}