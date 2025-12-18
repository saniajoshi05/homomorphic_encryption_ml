// 10_encrypted_linear_model.cpp
#include <chrono>
#include <cmath>
#include "examples.h"

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

struct ExperimentResult
{
    int n;
    double avg_total_ms;
    double avg_rot_ms;
    double avg_err;
    double max_err;
};

static ExperimentResult run_experiment(
    int n, int runs, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor,
    const GaloisKeys &gal_keys)
{
    double scale = pow(2.0, 40);

    vector<double> x(n), w(n);
    for (int i = 0; i < n; i++)
    {
        x[i] = i + 1;
        w[i] = 0.1 + 0.01 * (i % 50);
    }
    double b = 0.7;

    vector<double> t_total_vec, t_rot_vec, err_vec;
    t_total_vec.reserve(runs);
    t_rot_vec.reserve(runs);
    err_vec.reserve(runs);

    for (int r = 0; r < runs; r++)
    {
        // Plain baseline
        double y_plain = b;
        for (int i = 0; i < n; i++)
            y_plain += w[i] * x[i];

        auto t_total_start = clock_type::now();

        // Encode + Encrypt
        Plaintext x_plain;
        encoder.encode(x, scale, x_plain);
        Ciphertext x_enc;
        encryptor.encrypt(x_plain, x_enc);

        Plaintext w_plain;
        encoder.encode(w, scale, w_plain);

        Ciphertext prod;
        evaluator.multiply_plain(x_enc, w_plain, prod);
        evaluator.rescale_to_next_inplace(prod);

        // Rotations (dominant cost)
        auto t_rot_start = clock_type::now();
        Ciphertext sum = prod;
        for (int step = 1; step < n; step <<= 1)
        {
            Ciphertext rot;
            evaluator.rotate_vector(sum, step, gal_keys, rot);
            evaluator.add_inplace(sum, rot);
        }
        auto t_rot_end = clock_type::now();
        t_rot_vec.push_back(ms(t_rot_end - t_rot_start).count());

        // Add bias
        Plaintext b_plain;
        encoder.encode(b, sum.scale(), b_plain);
        evaluator.mod_switch_to_inplace(b_plain, sum.parms_id());
        evaluator.add_plain_inplace(sum, b_plain);

        // Decrypt + Decode
        Plaintext dec;
        decryptor.decrypt(sum, dec);
        vector<double> out;
        encoder.decode(dec, out);

        auto t_total_end = clock_type::now();
        t_total_vec.push_back(ms(t_total_end - t_total_start).count());

        err_vec.push_back(fabs(y_plain - out[0]));
    }

    ExperimentResult res;
    res.n = n;
    res.avg_total_ms = avg(t_total_vec);
    res.avg_rot_ms = avg(t_rot_vec);
    res.avg_err = avg(err_vec);
    res.max_err = vmax(err_vec);
    return res;
}

void example_encrypted_linear_model()
{
    print_example_banner("Example: Encrypted Linear Model (Size Sweep)");

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

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

    int runs = 5;
    vector<int> sizes = { 8, 16, 32, 64, 128 };

    cout << "Slots available: " << encoder.slot_count() << endl;
    cout << "Runs per size: " << runs << endl << endl;

    cout << "n\tavg_total_ms\tavg_rot_ms\tavg_err\t\tmax_err" << endl;
    cout << "---------------------------------------------------------------" << endl;

    for (int n : sizes)
    {
        if (n > (int)encoder.slot_count())
        {
            cout << n << "\t(SKIPPED)" << endl;
            continue;
        }

        auto res = run_experiment(n, runs, encoder, encryptor, evaluator, decryptor, gal_keys);

        cout << res.n << "\t" << res.avg_total_ms << "\t\t" << res.avg_rot_ms << "\t\t" << res.avg_err << "\t"
             << res.max_err << endl;
    }

    cout << endl << "Experiment complete." << endl;
}
