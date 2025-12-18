// 9_encrypted_linear_regression.cpp
#include "examples.h"

using namespace std;
using namespace seal;

void example_encrypted_linear_regression()
{
    print_example_banner("Example: Encrypted Linear Inference (y = a*x + b)");

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

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    // Toy data
    vector<double> x = { 1, 2, 3, 4, 5 };
    double a = 2.5;
    double b = 1.0;

    vector<double> y_plain(x.size());
    for (size_t i = 0; i < x.size(); i++)
        y_plain[i] = a * x[i] + b;

    cout << "Plaintext y = a*x + b:" << endl;
    print_vector(y_plain, 3, 7);

    // Encode + encrypt x
    Plaintext x_plain;
    encoder.encode(x, scale, x_plain);

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    // Encode a and b (broadcast)
    Plaintext a_plain, b_plain;
    encoder.encode(a, scale, a_plain);
    encoder.encode(b, scale, b_plain);

    // Compute a*x
    Ciphertext ax_encrypted;
    evaluator.multiply_plain(x_encrypted, a_plain, ax_encrypted);
    evaluator.rescale_to_next_inplace(ax_encrypted);

    // Make b compatible (same parms_id and scale)
    evaluator.mod_switch_to_inplace(b_plain, ax_encrypted.parms_id());
    b_plain.scale() = ax_encrypted.scale();

    // y = a*x + b
    Ciphertext y_encrypted;
    evaluator.add_plain(ax_encrypted, b_plain, y_encrypted);

    // Decrypt + decode
    Plaintext y_plain_result;
    decryptor.decrypt(y_encrypted, y_plain_result);

    vector<double> y_result;
    encoder.decode(y_plain_result, y_result);

    cout << "Decrypted y (first 5 slots):" << endl;
    vector<double> y_result_first5(y_result.begin(), y_result.begin() + x.size());
    print_vector(y_result_first5, 3, 7);

    cout << "Absolute errors |y_plain - y_dec|:" << endl;
    for (size_t i = 0; i < x.size(); i++)
        cout << "  x=" << x[i] << " error=" << fabs(y_plain[i] - y_result_first5[i]) << endl;
}
