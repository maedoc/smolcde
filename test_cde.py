import numpy as np
import sys
import os
import subprocess
import glob
from pathlib import Path
import ctypes
from cde import MDNEstimator, MAFEstimator, generate_test_data
from scipy.integrate import odeint
from scipy.signal import welch
import autograd.numpy as anp
import csv
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


def test_mdn_basic():
    # Create a simple MDN
    mdn = MDNEstimator(param_dim=2, feature_dim=1, n_components=3, hidden_sizes=(16,))
    # Generate simple test data
    params, features = generate_test_data('banana', n_samples=100, seed=42)
    # Train briefly
    mdn.train(params, features, n_epochs=50, learning_rate=1e-3, use_tqdm=False)
    # Test sampling
    rng = anp.random.RandomState(42)
    test_features = anp.array([[1.0]])
    samples = mdn.sample(test_features, 10, rng)
    assert samples.shape == (1, 10, 2), f"Expected shape (1, 10, 2), got {samples.shape}"
    assert not anp.any(anp.isnan(samples)), "Samples contain NaN values"
    # Test log probability
    log_probs = mdn.log_prob(features[:10], params[:10])
    assert log_probs.shape == (10,), f"Expected shape (10,), got {log_probs.shape}"
    assert anp.all(anp.isfinite(log_probs)), "Log probabilities contain non-finite values"


def test_maf_basic():
    # Create a simple MAF
    maf = MAFEstimator(param_dim=2, feature_dim=1, n_flows=2, hidden_units=16)
    # Generate simple test data
    params, features = generate_test_data('moons', n_samples=100, seed=42)
    # Train briefly
    maf.train(params, features, n_epochs=50, learning_rate=1e-3, use_tqdm=False)
    # Test sampling
    rng = anp.random.RandomState(42)
    test_features = anp.array([[0.1]])
    samples = maf.sample(test_features, 10, rng)
    assert samples.shape == (1, 10, 2), f"Expected shape (1, 10, 2), got {samples.shape}"
    assert not anp.any(anp.isnan(samples)), "Samples contain NaN values"
    # Test log probability
    log_probs = maf.log_prob(features[:10], params[:10])
    assert log_probs.shape == (10,), f"Expected shape (10,), got {log_probs.shape}"
    assert anp.all(anp.isfinite(log_probs)), "Log probabilities contain non-finite values"


def test_data_generation():
    datasets = ['banana', 'student_t', 'moons']
    for dataset in datasets:
        params, features = generate_test_data(dataset, n_samples=50, seed=42)
        assert params.shape == (50, 2), f"Expected params shape (50, 2), got {params.shape}"
        assert features.shape == (50, 1), f"Expected features shape (50, 1), got {features.shape}"
        assert not anp.any(anp.isnan(params)), f"Params contain NaN for dataset {dataset}"
        assert not anp.any(anp.isnan(features)), f"Features contain NaN for dataset {dataset}"


def test_error_handling():
    try:
        mdn = MDNEstimator(param_dim=0, feature_dim=1)
        assert False, "Should raise ValueError for invalid param_dim"
    except ValueError:
        pass
    
    try:
        maf = MAFEstimator(param_dim=1, feature_dim=-1)
        assert False, "Should raise ValueError for invalid feature_dim"
    except ValueError:
        pass
    
    mdn = MDNEstimator(param_dim=2, feature_dim=1)
    try:
        test_features = anp.array([[1.0]])
        test_params = anp.array([[1.0, 2.0]])
        mdn.log_prob(test_features, test_params)
        assert False, "Should raise RuntimeError when model not trained"
    except RuntimeError:
        pass


def lorenz_ode(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(n_samples=200):
    # Parameter ranges
    # sigma: [5, 15]
    # rho: [20, 40]
    # beta: [1, 5]
    params = []
    features = []
    # Simulation settings
    t = np.linspace(0, 5, 500) # 5 seconds, 100Hz
    fs = 100
    for i in range(n_samples):
        sigma = np.random.uniform(5, 15)
        rho = np.random.uniform(20, 40)
        beta = np.random.uniform(1, 5)
        state0 = [1.0, 1.0, 1.0]
        try:
            states = odeint(lorenz_ode, state0, t, args=(sigma, rho, beta))
        except Exception as e:
            print(f"Integration failed for {sigma, rho, beta}: {e}")
            continue
        # Take x component
        x_ts = states[:, 0]
        # Compute PSD
        f, Pxx = welch(x_ts, fs=fs, nperseg=128)
        # Pxx length = 128/2 + 1 = 65
        # Log PSD is usually better behaved
        log_Pxx = np.log10(Pxx + 1e-10)
        params.append([sigma, rho, beta])
        features.append(log_Pxx)
    return np.array(features), np.array(params)

def test_lorenz():
    np.random.seed(42)
    features, params = generate_lorenz_data(n_samples=1000) 
    # Split train/test
    n_train = int(0.8 * len(features))
    X_train, Y_train = features[:n_train], params[:n_train]
    X_test, Y_test = features[n_train:], params[n_train:]
    # High dimensionality in features, low in params.
    model = MAFEstimator(n_flows=4, hidden_units=64, param_dim=3, feature_dim=features.shape[1])
    model.train(Y_train, X_train, n_epochs=500, batch_size=32, learning_rate=0.0002)
    rng = anp.random.RandomState(42)
    # Evaluate on a few test samples
    n_eval = 5
    for i in range(n_eval):
        x_cond = X_test[i:i+1]
        y_true = Y_test[i]
        # Sample
        samples = model.sample(x_cond, 500, rng)
        # samples shape (1, 500, 3)
        s_flat = samples[0]
        mean_est = np.mean(s_flat, axis=0)
        np.testing.assert_allclose(y_true, mean_est, rtol=1.0)


class MAF_C_Trainer:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        
        # Define types
        self.lib.maf_init_random_model.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        self.lib.maf_init_random_model.restype = ctypes.c_void_p
        
        self.lib.maf_create_workspace.argtypes = [ctypes.c_void_p]
        self.lib.maf_create_workspace.restype = ctypes.c_void_p
        
        self.lib.maf_create_cache.argtypes = [ctypes.c_void_p]
        self.lib.maf_create_cache.restype = ctypes.c_void_p
        
        self.lib.maf_create_grad.argtypes = [ctypes.c_void_p]
        self.lib.maf_create_grad.restype = ctypes.c_void_p
        
        self.lib.maf_create_adam.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.maf_create_adam.restype = ctypes.c_void_p
        
        self.lib.maf_zero_grad.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        
        self.lib.maf_forward_train.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.maf_forward_train.restype = ctypes.c_float
        
        self.lib.maf_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        
        self.lib.maf_adam_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        
        self.lib.maf_sample.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        
        # Free functions
        self.lib.maf_free_model.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_workspace.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_cache.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_grad.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_adam.argtypes = [ctypes.c_void_p]

    def train(self, params, features, n_flows, hidden_units, n_epochs, lr=0.001, batch_size=8, seed=42):
        params = np.asarray(params, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        
        N, D = params.shape
        _, C = features.shape

        # Initialize model
        model = self.lib.maf_init_random_model(n_flows, D, C, hidden_units)
        if not model: raise RuntimeError("Failed to init model")
        
        ws = self.lib.maf_create_workspace(model)
        cache = self.lib.maf_create_cache(model)
        grad = self.lib.maf_create_grad(model)
        adam = self.lib.maf_create_adam(model, lr, 0.9, 0.999, 1e-8)
        
        rng = np.random.RandomState(seed)
        
        # Training Loop
        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0
            n_batches = 0
            for i in range(0, N, batch_size):
                if i + batch_size > N: break # Drop last incomplete
                idx = perm[i:i+batch_size]
                p_batch = params[idx]
                f_batch = features[idx]
                p_ptr = p_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                f_ptr = f_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                self.lib.maf_zero_grad(model, grad)
                # Assume batch_size matches MAF_BATCH_SIZE (8) for simplicity in this low-level loop
                loss = self.lib.maf_forward_train(model, ws, cache, f_ptr, p_ptr)
                self.lib.maf_backward(model, cache, grad, f_ptr, p_ptr)
                self.lib.maf_adam_step(model, adam, grad)
                
                epoch_loss -= loss # maf_forward_train returns Log Prob (negative NLL)
                n_batches += 1
            
        # Cleanup aux
        self.lib.maf_free_workspace(ws)
        self.lib.maf_free_cache(cache)
        self.lib.maf_free_grad(grad)
        self.lib.maf_free_adam(adam)
        
        return model, epoch_loss/n_batches

    def sample(self, model, features, n_samples):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        _, C = features.shape
        f_vec = features[0] 
        param_dim = 2 # hardcoded for banana
        samples = np.zeros((n_samples, param_dim), dtype=np.float32)
        f_ptr = f_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        s_ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.maf_sample(model, f_ptr, n_samples, s_ptr, 1234)
        return samples

def test_c_training():
    # Look for libsmolmaf.so in current dir or likely build locations
    candidates = glob.glob('libsmolmaf.so') + glob.glob('*/libsmolmaf.so') + glob.glob('*/*/libsmolmaf.so')
    if not candidates:
        print("Error: libsmolmaf.so not found")
        sys.exit(1)
    
    lib_path = Path(candidates[0]).absolute()
    trainer = MAF_C_Trainer(str(lib_path))
    params, features = generate_test_data("banana", 2000)
    # train C Model
    batch_size = 8
    c_model, c_batch_loss = trainer.train(params, features, n_flows=3, hidden_units=32, n_epochs=50, lr=0.001, batch_size=batch_size)
    c_sample_loss = c_batch_loss / batch_size
    # train Python Model
    py_model = MAFEstimator(2, 1, n_flows=3, hidden_units=32)
    py_model.train(params, features, n_epochs=50, batch_size=8, learning_rate=0.001, use_tqdm=False)
    # compare sampling
    test_feat = np.array([0.5], dtype=np.float32)
    n_samples = 1000
    c_samples = trainer.sample(c_model, test_feat, n_samples)
    py_samples = py_model.sample(test_feat.reshape(1,-1), n_samples, np.random.RandomState(42)).reshape(-1, 2)
    c_mean = c_samples.mean(axis=0)
    py_mean = py_samples.mean(axis=0)
    np.testing.assert_allclose(c_mean, py_mean, rtol=1.0)
    trainer.lib.maf_free_model(c_model)

def save_csv(data, filename, header=None):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def test_mnist_cli_workflow(tmp_path):
    """
    Test the full CLI workflow using the MNIST digits dataset.
    1. Prepare data (features=pixels, params=digit_class)
    2. Train model via CLI
    3. Infer/Sample via CLI
    4. Verify accuracy of predicted digits
    """
    # 1. Prepare Data
    digits = load_digits()
    X = digits.data  # (1797, 64)
    y = digits.target # (1797,)
    
    # Normalize features for better training stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We'll use a subset for speed in testing
    n_samples = 500
    X_subset = X_scaled[:n_samples]
    y_subset = y[:n_samples].reshape(-1, 1).astype(float)
    
    # Add some noise to y to make it strictly continuous-like (MAF assumes continuous)
    # Though it works okay for discrete if we don't care about exact density values
    rng = np.random.RandomState(42)
    y_noisy = y_subset + rng.normal(0, 0.1, size=y_subset.shape)
    
    feat_file = tmp_path / "features.csv"
    param_file = tmp_path / "params.csv"
    model_file = tmp_path / "mnist.maf"
    pred_file = tmp_path / "predictions.csv"
    
    save_csv(X_subset, feat_file)
    save_csv(y_noisy, param_file)
    
    # 2. Train
    train_cmd = [
        "./smolcde", "train",
        "--features", str(feat_file),
        "--params", str(param_file),
        "--out", str(model_file),
        "--epochs", "300",
        "--hidden", "32",
        "--blocks", "5",
        "--lr", "0.001",
        "--batch", "32"
    ]
    
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, "Training failed"
    assert model_file.exists(), "Model file not created"

    # 3. Infer (Sample)
    infer_cmd = [
        "./smolcde", "infer",
        "--model", str(model_file),
        "--features", str(feat_file),
        "--out", str(pred_file),
        "--mode", "stats", # Get mean/std directly
        "--samples", "100"
    ]
    
    print(f"Running: {' '.join(infer_cmd)}")
    result = subprocess.run(infer_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, "Inference failed"
    assert pred_file.exists(), "Prediction file not created"

    # 4. Verify
    # Load predictions. Format for stats mode: feature_idx, stat, p0...
    # We want rows where stat='mean'
    
    predictions = []
    with open(pred_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['stat'] == 'mean':
                # Parse p0 (only 1 param dim)
                val = float(row['p0'])
                predictions.append(val)
    
    predictions = np.array(predictions)
    
    # Verify length
    assert len(predictions) == n_samples
    
    # Check correlation or error
    # Round predictions to nearest integer to check classification accuracy
    y_true = y[:n_samples]
    y_pred = np.round(predictions).astype(int)
    
    accuracy = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - predictions))
    
    print(f"MNIST MAF Accuracy: {accuracy:.4f}")
    print(f"MNIST MAF MAE: {mae:.4f}")
    
    assert accuracy > 0.75, f"Accuracy {accuracy} too low, model failed to learn basic structure"
