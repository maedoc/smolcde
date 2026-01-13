"""
A module for conditional density estimation using Mixture Density
Networks (MDNs) and Masked Autoregressive Flows (MAFs).

"""

import abc
import math
from dataclasses import dataclass, field

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
from sklearn.datasets import make_moons
from scipy.stats import t
from tqdm.auto import trange

# =============================================================================
# == Base Class for Conditional Density Estimators
# =============================================================================

class ConditionalDensityEstimator(abc.ABC):
    """
    Abstract base class for conditional density estimators.

    This class provides a unified training interface using the Adam optimizer
    and standardizes the API for training, sampling, and log-probability
    evaluation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable (parameters to be estimated).
    feature_dim : int
        Dimensionality of the conditional variable (features).
    """
    def __init__(self, param_dim: int, feature_dim: int):
        if not (param_dim > 0 and feature_dim >= 0):
            raise ValueError("Parameter and feature dimensions must be positive/non-negative.")
        self.param_dim = param_dim
        self.feature_dim = feature_dim
        self.weights = None
        self.loss_history = []

    @abc.abstractmethod
    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """
        Initialize the trainable weights of the model.

        Parameters
        ----------
        rng : autograd.numpy.random.RandomState
            A random number generator for reproducible initialization.

        Returns
        -------
        dict
            A dictionary of initialized weight arrays.
        """
        pass

    @abc.abstractmethod
    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        """
        Compute the negative log-likelihood loss for a batch of data.

        Parameters
        ----------
        weights : dict
            A dictionary of the model's trainable weights.
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        float
            The mean negative log-likelihood of the batch.
        """
        pass

    @abc.abstractmethod
    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        """
        Generate samples from the learned conditional distribution p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (n_conditions, feature_dim) array of features to condition on.
        n_samples : int
            The number of samples to generate for each condition.
        rng : autograd.numpy.random.RandomState
            A random number generator for sampling.

        Returns
        -------
        anp.ndarray
            An array of generated samples of shape (n_conditions, n_samples, param_dim).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @abc.abstractmethod
    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Compute the log-probability log p(params|features).

        Parameters
        ----------
        features : anp.ndarray
            A (N, feature_dim) array of conditional features.
        params : anp.ndarray
            A (N, param_dim) array of target parameters.

        Returns
        -------
        anp.ndarray
            A (N,) array of log-probabilities.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
    
    def train(self, params: anp.ndarray, features: anp.ndarray,
                  batch_size: int = 32, train_ratio: float = 0.8,
                  n_epochs: int = 100, learning_rate: float = 1e-3,
                  seed: int = 0, use_tqdm: bool = True):
            """
            Trains the model using Adam with mini-batches and validation.
            """
            # --- 1. Data Validation & Splitting ---
            if params.shape[0] != features.shape[0]:
                raise ValueError("Params and features must have the same number of samples.")
            
            # Filter finite values
            finite_idx = anp.all(anp.isfinite(params), axis=1) & anp.all(anp.isfinite(features), axis=1)
            params = params[finite_idx].astype('f')
            features = features[finite_idx].astype('f')
            N = params.shape[0]
    
            rng = anp.random.RandomState(seed)
            perm = rng.permutation(N)
            n_train = int(N * train_ratio)
            train_idx, val_idx = perm[:n_train], perm[n_train:]
            
            # --- 2. Initialization ---
            # Reset weights for every training run (consistent with original implementation)
            self.weights = self._initialize_weights(rng)
            self.loss_history = {'train': [], 'val': []}
    
            # Adam optimizer state
            m = {key: anp.zeros_like(val) for key, val in self.weights.items()}
            v = {key: anp.zeros_like(val) for key, val in self.weights.items()}
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            global_step = 0
    
            # --- 3. Optimization Loop ---
            gradient_func = grad(self._loss_function)
            iterator = trange(n_epochs, desc="Training", disable=not use_tqdm)
    
            for epoch in iterator:
                # Shuffle training data for this epoch
                epoch_perm = rng.permutation(train_idx)
                epoch_train_losses = []
    
                # Mini-batch loop
                for i in range(0, n_train, batch_size):
                    batch_indices = epoch_perm[i : i + batch_size]
                    p_batch = params[batch_indices]
                    f_batch = features[batch_indices]
                    
                    global_step += 1
                    g = gradient_func(self.weights, f_batch, p_batch)
                    loss = self._loss_function(self.weights, f_batch, p_batch)
                    epoch_train_losses.append(loss)
    
                    # Adam update
                    for key in self.weights:
                        m[key] = beta1 * m[key] + (1 - beta1) * g[key]
                        v[key] = beta2 * v[key] + (1 - beta2) * (g[key]**2)
                        m_hat = m[key] / (1 - beta1**global_step)
                        v_hat = v[key] / (1 - beta2**global_step)
                        self.weights[key] -= learning_rate * m_hat / (anp.sqrt(v_hat) + epsilon)
    
                # Validation step
                avg_train_loss = anp.mean(epoch_train_losses)
                val_loss = self._loss_function(self.weights, features[val_idx], params[val_idx])
                
                self.loss_history['train'].append(avg_train_loss)
                self.loss_history['val'].append(val_loss)
    
                if use_tqdm:
                    iterator.set_postfix(train=f"{avg_train_loss:.3f}", val=f"{val_loss:.3f}")
    

# =============================================================================
# == MDN Implementation
# =============================================================================

@dataclass
class MDNEstimator(ConditionalDensityEstimator):
    """
    Mixture Density Network for conditional density estimation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_components : int, optional
        The number of Gaussian mixture components.
    hidden_sizes : tuple[int, ...], optional
        A tuple specifying the number of units in each hidden layer.
    """
    param_dim: int
    feature_dim: int
    n_components: int = 5
    hidden_sizes: tuple[int, ...] = (32, 32)

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self._offdiag_basis = self._create_offdiag_basis()

    def _create_offdiag_basis(self):
        n_off_diag = self.param_dim * (self.param_dim - 1) // 2
        if n_off_diag == 0:
            return None
        basis = anp.zeros((n_off_diag, self.param_dim, self.param_dim), dtype='f')
        rows, cols = anp.triu_indices(self.param_dim, k=1)
        basis[anp.arange(n_off_diag), rows, cols] = 1
        return basis

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights for the MLP and GMM output layers."""
        weights = {}
        in_size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f'W{i}'] = (rng.randn(in_size, out_size) * anp.sqrt(2.0 / in_size)).astype('f')
            weights[f'b{i}'] = anp.zeros(out_size, dtype='f')
            in_size = out_size

        last_hidden_size = self.hidden_sizes[-1] if self.hidden_sizes else self.feature_dim

        # GMM output layers
        K, D_out = self.n_components, self.param_dim
        weights['W_alpha'] = (rng.randn(last_hidden_size, K) * 0.01).astype('f')
        weights['b_alpha'] = anp.zeros(K, dtype='f')
        weights['W_mu'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_mu'] = anp.zeros(K * D_out, dtype='f')
        weights['W_L_prec_log_diag'] = (rng.randn(last_hidden_size, K * D_out) * 0.01).astype('f')
        weights['b_L_prec_log_diag'] = anp.zeros(K * D_out, dtype='f')

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            weights['W_L_prec_offdiag'] = (rng.randn(last_hidden_size, K * n_off_diag) * 0.01).astype('f')
            weights['b_L_prec_offdiag'] = anp.zeros(K * n_off_diag, dtype='f')

        return weights

    def _forward_pass(self, weights: dict, features: anp.ndarray):
        """Maps input features to GMM parameters."""
        h = features
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(h @ weights[f'W{i}'] + weights[f'b{i}'])

        K, D_out = self.n_components, self.param_dim
        log_alpha = h @ weights['W_alpha'] + weights['b_alpha']
        alpha = anp.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        mu = (h @ weights['W_mu'] + weights['b_mu']).reshape(-1, K, D_out)

        L_prec_log_diag = (h @ weights['W_L_prec_log_diag'] + weights['b_L_prec_log_diag']).reshape(-1, K, D_out)
        L_prec_diag_mat = anp.einsum('nki,ij->nkij', anp.exp(L_prec_log_diag), anp.eye(D_out, dtype='f'))

        n_off_diag = D_out * (D_out - 1) // 2
        if n_off_diag > 0:
            L_prec_offdiag_vals = (h @ weights['W_L_prec_offdiag'] + weights['b_L_prec_offdiag']).reshape(-1, K, n_off_diag)
            L_prec_offdiag_mat = anp.einsum('nkl,lij->nkij', L_prec_offdiag_vals, self._offdiag_basis)
            L_prec = L_prec_diag_mat + L_prec_offdiag_mat
        else:
            L_prec = L_prec_diag_mat

        return alpha, mu, L_prec, L_prec_log_diag

    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        """Computes the negative log-likelihood of the true parameters under the GMM."""
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(weights, features)

        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu

        z = anp.einsum('nkij,nkj->nki', L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)

        log_probs_k = quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)

        return -anp.mean(total_log_prob)

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        """
        Computes the log-probability log p(params|features) for each sample.
        """
        super().log_prob(features, params)
    
        # Perform a forward pass to get GMM parameters
        alpha, mu, L_prec, L_prec_log_diag = self._forward_pass(self.weights, features)
    
        # Reshape parameters for broadcasting against mixture components
        y_true_reshaped = params[:, anp.newaxis, :]
        delta = y_true_reshaped - mu
    
        # Compute the log-probability for each component (k) for each sample (n)
        z = anp.einsum('nkij,nkj->nki', L_prec, delta)
        quad_term = -0.5 * anp.sum(z**2, axis=2)
        log_det_term = anp.sum(L_prec_log_diag, axis=2)
    
        log_probs_k = quad_term + log_det_term - 0.5 * self.param_dim * anp.log(2 * math.pi)
    
        # Combine component log-probabilities using the mixture weights (alpha)
        # This returns a vector of shape (N,)
        total_log_prob = logsumexp(anp.log(alpha + 1e-9) + log_probs_k, axis=1)
    
        return total_log_prob


    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        alpha, mu, L_prec, _ = self._forward_pass(self.weights, features)
        n_cond, K, D_out = mu.shape

        log_alpha = anp.log(alpha + 1e-9)
        gumbel_noise = -anp.log(-anp.log(rng.uniform(size=(n_cond, n_samples, K))))
        component_indices = anp.argmax(log_alpha[:, anp.newaxis, :] + gumbel_noise, axis=2)

        cond_idx = anp.arange(n_cond)[:, anp.newaxis]
        chosen_mu = mu[cond_idx, component_indices]
        chosen_L_prec = L_prec[cond_idx, component_indices]

        try:
            L_cov_factor = anp.linalg.inv(chosen_L_prec)
        except anp.linalg.LinAlgError:
            print("Warning: Singular precision matrix encountered during sampling. Returning NaNs.")
            return anp.full((n_cond, n_samples, D_out), anp.nan)

        z = rng.randn(n_cond, n_samples, D_out)
        samples = chosen_mu + anp.einsum('nsij,nsj->nsi', L_cov_factor, z)

        return samples

# =============================================================================
# == MAF Implementation
# =============================================================================

@dataclass
class MAFEstimator(ConditionalDensityEstimator):
    """
    Masked Autoregressive Flow for conditional density estimation.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the target variable.
    feature_dim : int
        Dimensionality of the conditional variable.
    n_flows : int, optional
        The number of flow layers (MADE blocks).
    hidden_units : int, optional
        The number of hidden units in each MADE block.
    """
    param_dim: int
    feature_dim: int
    n_flows: int = 4
    hidden_units: int = 64

    def __post_init__(self):
        super().__init__(self.param_dim, self.feature_dim)
        self.model_constants = None # For non-trainable parts like masks

    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights and model constants (masks, permutations)."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units

        for k in range(self.n_flows):
            # MADE masks and permutation
            m_in = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
            m_out = m_in.copy()
            M2 = (m_hidden[None, :] < m_out[:, None]).astype('f')
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int); inv_perm[perm] = anp.arange(D)

            layers.append({'M1': M1, 'M2': M2, 'perm': perm, 'inv_perm': inv_perm})

            # Trainable parameters
            w_std = 0.01
            weights[f'W1y_{k}'] = (rng.randn(H, D) * w_std).astype('f')
            weights[f'W1c_{k}'] = (rng.randn(H, C) * w_std).astype('f') if C > 0 else anp.zeros((H, C), dtype='f')
            weights[f'b1_{k}'] = anp.zeros(H, dtype='f')
            weights[f'W2_{k}'] = anp.zeros((2 * D, H), dtype='f')
            weights[f'W2c_{k}'] = anp.zeros((2 * D, C), dtype='f') if C > 0 else anp.zeros((2*D, C), dtype='f')
            weights[f'b2_{k}'] = anp.zeros(2 * D, dtype='f')

        self.model_constants = {'layers': layers}
        return weights

    def _made_forward(self, y, ctx, layer_const, k, weights):
        """Single forward pass through a MADE block."""
        M1, M2 = layer_const['M1'], layer_const['M2']
        W1y, W1c, b1 = weights[f'W1y_{k}'], weights[f'W1c_{k}'], weights[f'b1_{k}']
        W2, W2c, b2 = weights[f'W2_{k}'], weights[f'W2c_{k}'], weights[f'b2_{k}']

        y_h = anp.dot(y, (W1y * M1).T)
        c_h = anp.dot(ctx, W1c.T) if self.feature_dim > 0 else 0.0
        h = anp.tanh(y_h + c_h + b1)

        M2_tiled = anp.concatenate([M2, M2], axis=0)
        out = anp.dot(h, (W2 * M2_tiled).T)
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, W2c.T)
        out = out + b2

        mu, alpha = out[:, :self.param_dim], anp.clip(out[:, self.param_dim:], -7.0, 7.0)
        return mu, alpha

    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """Computes log probability for the MAF."""
        u = params
        log_det = anp.zeros(params.shape[0])

        for k, layer_const in enumerate(self.model_constants['layers']):
            u = u[:, layer_const['perm']]
            mu, alpha = self._made_forward(u, features, layer_const, k, weights)
            u = (u - mu) * anp.exp(-alpha)
            log_det -= anp.sum(alpha, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base_logp + log_det

    def _loss_function(self, weights: dict, features: anp.ndarray, params: anp.ndarray) -> float:
        return -anp.mean(self._get_log_prob(weights, features, params))

    def log_prob(self, features: anp.ndarray, params: anp.ndarray) -> anp.ndarray:
        super().log_prob(features, params)
        return self._get_log_prob(self.weights, features, params)

    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        super().sample(features, n_samples, rng)
        features = features.astype('f')
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_cond = features.shape[0]
        # Broadcast features to match number of samples
        if n_cond != n_samples:
            features = anp.repeat(features, n_samples, axis=0)

        z = rng.randn(n_samples, self.param_dim).astype('f')
        x = z

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants['layers']))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            for i in range(self.param_dim):
                mu, alpha = self._made_forward(u, features, layer_const, k, self.weights)
                u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]
            x = u[:, layer_const['inv_perm']]

        # Reshape to (n_conditions, n_samples, param_dim)
        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)

# NSF
def rational_quadratic_spline(inputs, unnorm_widths, unnorm_heights, unnorm_derivs, inverse=False,
                              left=-3., right=3., bottom=-3., top=3., min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
    """
    Apply a rational quadratic spline to the inputs.
    
    If inverse=False (Forward): Maps Data (x) -> Noise (z). Used in log_prob.
    If inverse=True (Inverse): Maps Noise (z) -> Data (x). Used in sample.
    """
    num_bins = unnorm_widths.shape[-1]

    # --- 1. Parameter Constraints (Softmax/Softplus) ---
    # Widths (x-axis)
    widths = anp.exp(unnorm_widths - logsumexp(unnorm_widths, axis=-1, keepdims=True))
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    # Heights (y-axis)
    heights = anp.exp(unnorm_heights - logsumexp(unnorm_heights, axis=-1, keepdims=True))
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    # Derivatives (slopes at knots)
    derivs = min_derivative + anp.log(1 + anp.exp(unnorm_derivs))

    # --- 2. Build Cumulative Knots ---
    pad_shape = widths.shape[:-1] + (1,)
    
    cum_widths = anp.cumsum(widths, axis=-1)
    cum_widths = anp.concatenate([anp.zeros(pad_shape), cum_widths], axis=-1)
    cum_widths = cum_widths * (right - left) + left # Scale to domain
    
    cum_heights = anp.cumsum(heights, axis=-1)
    cum_heights = anp.concatenate([anp.zeros(pad_shape), cum_heights], axis=-1)
    cum_heights = cum_heights * (top - bottom) + bottom # Scale to codomain

    # --- 3. Identify Bins ---
    if inverse:
        # Searching based on y (codomain) to find x
        knots = cum_heights
        # Scale input to [0,1] for bin search stability, then map back
        inputs_clamped = anp.clip(inputs, bottom + 1e-5, top - 1e-5)
    else:
        # Searching based on x (domain) to find y
        knots = cum_widths
        inputs_clamped = anp.clip(inputs, left + 1e-5, right - 1e-5)

    # Hard mask for bin selection (Autograd friendly compared to searchsorted)
    # shape: (..., num_bins)
    mask = (inputs_clamped[..., None] >= knots[..., :-1]) & (inputs_clamped[..., None] < knots[..., 1:])
    
    # Handle edge case where input is exactly the max boundary
    # (Assign to last bin if not caught)
    mask_sum = anp.sum(mask, axis=-1, keepdims=True)
    mask_fill = anp.zeros_like(mask)
    mask_fill = anp.concatenate([mask_fill[..., :-1], anp.ones_like(mask_fill[..., -1:])], axis=-1)
    # If no bin selected (input=max), use last bin
    mask = anp.where(mask_sum > 0, mask, mask_fill)

    # Extract parameters for the active bin
    w_k = anp.sum(mask * widths, axis=-1)
    h_k = anp.sum(mask * heights, axis=-1)
    d_k = anp.sum(mask * derivs[..., :-1], axis=-1)
    d_kp1 = anp.sum(mask * derivs[..., 1:], axis=-1)
    
    x_k = anp.sum(mask * cum_widths[..., :-1], axis=-1)
    y_k = anp.sum(mask * cum_heights[..., :-1], axis=-1)

    # --- 4. Spline Transform ---
    
    if inverse:
        # INVERSE: Solve Quadratic to find xi given (y - y_k)
        # y = params -> x = result
        # This is used in Sampling (Noise -> Data)
        
        delta_y = inputs - y_k # (y - y_k)
        s_y = h_k # height of bin
        
        # Quadratic coefficients: a*xi^2 + b*xi + c = 0
        # Equation derived from RQS inverse definition
        term = delta_y * (d_k + d_kp1 - 2 * s_y)
        a = s_y - delta_y + term
        b = s_y * d_k - delta_y * d_k + term - (s_y * d_k) # Simplified: (h - dy)*d_k + term
        b = h_k * d_k - delta_y * (d_k + d_kp1 - 2*h_k) # Re-check algebra, standard form:
        # Let's use the robust form from standard implementations:
        
        theta = (inputs - y_k) / h_k # relative y in [0,1]
        
        # Coeffs for: alpha * xi^2 + beta * xi + gamma = 0
        # This is derived from rearranging the rational quadratic fraction
        cdk = d_k
        cdkp1 = d_kp1
        
        # alpha = (d_k + d_kp1 - 2) * theta + (1 - theta)
        # Note: We need careful derivation here to match the forward pass exactly. 
        # For conciseness/stability in autograd, we use the standard algebraic solution:
        
        numerator_term = h_k * (h_k * theta - theta * s_y) # This gets messy manually.
        
        # ALGEBRAIC SHORTCUT:
        # We solve for xi in: y_rel = (h * (s*xi^2*d + xi*s^2*dkp1) + ...) / s^2 ...
        # Standard solution:
        
        a = (h_k - s_y) * (d_k + d_kp1 - 2 * s_y) # Wait, s_y is h_k here? No, s_y=slope=h/w
        s_k = h_k / w_k
        
        # Let's use the explicit quadratic formula for xi
        # A_ * xi^2 + B_ * xi + C_ = 0
        # Common RQS result:
        q_a = h_k * (s_k - d_k) + (inputs - y_k) * (d_k + d_kp1 - 2 * s_k)
        q_b = h_k * d_k - (inputs - y_k) * (d_k + d_kp1 - 2 * s_k)
        q_c = - s_k * (inputs - y_k)
        
        # Robust quadratic root
        xi = 2 * q_c / (-q_b - anp.sqrt(q_b**2 - 4 * q_a * q_c))
        
        # Calculate resulting x
        outputs = xi * w_k + x_k
        
        # Log determinant for inverse is -log_det_forward
        # Re-compute forward deriv at this xi
        s = 1 - xi
        numerator = (s * w_k)**2 * (d_k * xi**2 + 2 * s_k * xi * s + d_kp1 * s**2)
        denominator = (s_k + (d_k + d_kp1 - 2 * s_k) * xi * s)**2
        dydx = numerator / denominator
        
        log_det = -anp.log(dydx)
        
        return outputs, log_det

    else:
        # FORWARD: Evaluate Spline (Data -> Noise)
        # This is used in Log_Prob
        
        xi = (inputs - x_k) / w_k
        xi = anp.clip(xi, 0, 1)
        
        s = 1 - xi
        s_k = h_k / w_k
        
        # Rational Quadratic Function
        numerator = h_k * (s_k * xi**2 + d_k * s**2 * xi) # Simplified
        numerator = (h_k * xi**2 * (d_kp1 * xi + 2 * s_k * s) + h_k * s**2 * (d_k * xi)) # Full expansion check
        # Correct Form:
        numerator = h_k * (s * d_k * xi + xi * s_k * xi) # No, this is getting confusing.
        
        # Standard Clean Form:
        num = (h_k * s * xi) * (d_k * s + d_kp1 * xi) + h_k * xi * xi * s_k # No.
        
        # Let's use the Gregory/Delbourgo form:
        # y(xi) = y_k + h_k * (xi * (d_k * s + s_k * xi)) / (s_k + (d_k + d_kp1 - 2*s_k) * xi * s)
        
        common_denom = s_k + (d_k + d_kp1 - 2 * s_k) * xi * s
        
        y_rel = h_k * xi * (d_k * s + s_k * xi) / common_denom
        outputs = y_k + y_rel

        # Derivative dy/dx
        # dy/dx = (s_k^2 * (d_kp1 * xi^2 + 2 * s_k * xi * s + d_k * s^2)) / common_denom^2
        # Note: s_k in numerator might act differently depending on normalization.
        
        grad_numerator = s_k**2 * (d_kp1 * xi**2 + 2 * s_k * xi * s + d_k * s**2)
        dydx = grad_numerator / (common_denom**2)
        
        log_det = anp.log(dydx)
        
        return outputs, log_det

@dataclass
class NSFEstimator(MAFEstimator):
    """
    Neural Spline Flow (Autoregressive) estimator.
    Replaces affine transforms with monotonic rational-quadratic splines.
    """
    n_bins: int = 8
    
    def _initialize_weights(self, rng: anp.random.RandomState) -> dict:
        """Initializes weights with expanded output for spline parameters."""
        weights = {}
        layers = []
        D, C, H = self.param_dim, self.feature_dim, self.hidden_units
        K = self.n_bins
        
        # Output dim: Widths (K), Heights (K), Derivatives (K-1) + 2 (boundary) -> (3*K - 1)
        # Actually standard is: K widths, K heights, K+1 derivatives
        out_mult = 3 * K + 1

        for k in range(self.n_flows):
            # MADE masks
            m_in = anp.arange(1, D + 1)
            m_hidden = rng.randint(1, D, size=H)
            M1 = (m_in[None, :] <= m_hidden[:, None]).astype('f')
            
            # M2 must now be repeated for the spline params
            m_out = m_in.copy()
            M2_base = (m_hidden[None, :] < m_out[:, None]).astype('f')
            # Shape (D * out_mult, H)
            M2 = anp.repeat(M2_base, out_mult, axis=0)
            
            perm = rng.permutation(D)
            inv_perm = anp.empty(D, dtype=int); inv_perm[perm] = anp.arange(D)
            layers.append({'M1': M1, 'M2': M2, 'perm': perm, 'inv_perm': inv_perm})

            # Weights
            w_std = 0.01
            weights[f'W1y_{k}'] = (rng.randn(H, D) * w_std).astype('f')
            weights[f'W1c_{k}'] = (rng.randn(H, C) * w_std).astype('f') if C > 0 else anp.zeros((H, C), dtype='f')
            weights[f'b1_{k}'] = anp.zeros(H, dtype='f')
            
            # Output layer is larger now
            weights[f'W2_{k}'] = (rng.randn(D * out_mult, H) * w_std).astype('f')
            weights[f'W2c_{k}'] = (rng.randn(D * out_mult, C) * w_std).astype('f') if C > 0 else anp.zeros((D*out_mult, C), dtype='f')
            weights[f'b2_{k}'] = anp.zeros(D * out_mult, dtype='f')

        self.model_constants = {'layers': layers}
        return weights

    def _made_forward_spline(self, y, ctx, layer_const, k, weights):
        """Computes spline parameters (W, H, D) for the given input."""
        M1, M2 = layer_const['M1'], layer_const['M2']
        
        # 1. Hidden Layer (Same as MAF)
        y_h = anp.dot(y, (weights[f'W1y_{k}'] * M1).T)
        c_h = anp.dot(ctx, weights[f'W1c_{k}'].T) if self.feature_dim > 0 else 0.0
        h = anp.tanh(y_h + c_h + weights[f'b1_{k}'])

        # 2. Output Layer (Expanded)
        out = anp.dot(h, (weights[f'W2_{k}'] * M2).T) + weights[f'b2_{k}']
        if self.feature_dim > 0:
            out = out + anp.dot(ctx, weights[f'W2c_{k}'].T)
            
        # 3. Reshape: (Batch, D * (3K+1)) -> (Batch, D, 3K+1)
        params = out.reshape(y.shape[0], self.param_dim, -1)
        
        # Split into Widths, Heights, Derivatives
        K = self.n_bins
        w = params[..., :K]
        h = params[..., K:2*K]
        d = params[..., 2*K:]
        
        return w, h, d

    def _get_log_prob(self, weights: dict, features: anp.ndarray, params: anp.ndarray):
        """
        Flow Forward: Data (x) -> Noise (z)
        Uses rational_quadratic_spline(inverse=False)
        """
        u = params
        log_det_total = anp.zeros(params.shape[0])

        for k, layer_const in enumerate(self.model_constants['layers']):
            u = u[:, layer_const['perm']]
            
            # Get spline params based on u
            # (Note: due to masking, u[i]'s params only depend on u[<i])
            w, h, d = self._made_forward_spline(u, features, layer_const, k, weights)
            
            # Apply Spline: Data -> Noise
            u_new, log_det_layer = rational_quadratic_spline(u, w, h, d, inverse=False)
            
            u = u_new
            log_det_total += anp.sum(log_det_layer, axis=1)

        base_logp = -0.5 * anp.sum(u**2, axis=1) - 0.5 * self.param_dim * anp.log(2.0 * anp.pi)
        return base_logp + log_det_total

    def sample(self, features: anp.ndarray, n_samples: int, rng: anp.random.RandomState) -> anp.ndarray:
        """
        Flow Inverse: Noise (z) -> Data (x)
        Uses rational_quadratic_spline(inverse=True) sequentially.
        """
        # (This setup code mirrors MAFEstimator.sample)
        if self.weights is None: raise RuntimeError("Train first")
        features = features.astype('f')
        if features.ndim == 1: features = features.reshape(1, -1)
        if features.shape[0] != n_samples: features = anp.repeat(features, n_samples, axis=0)

        z = rng.randn(n_samples, self.param_dim).astype('f')
        x = z 

        # Invert the flow stack
        for k, layer_const in reversed(list(enumerate(self.model_constants['layers']))):
            y_perm = x
            u = anp.zeros_like(y_perm)
            
            # Sequential autoregressive sampling
            for i in range(self.param_dim):
                # We need w,h,d for dimension i. 
                # These depend on u[:, <i], which we have already computed in this loop.
                w, h, d = self._made_forward_spline(u, features, layer_const, k, self.weights)
                
                # We only transform dimension i
                # Select params for dim i: shape (Batch, K)
                w_i = w[:, i, :]
                h_i = h[:, i, :]
                d_i = d[:, i, :]
                
                input_z = y_perm[:, i] # The noise value
                
                # Invert Spline: Noise -> Data
                # Note: function expects (Batch, Dim, K), so we add axis
                val, _ = rational_quadratic_spline(
                    input_z[:, None], 
                    w_i[:, None, :], 
                    h_i[:, None, :], 
                    d_i[:, None, :], 
                    inverse=True
                )
                
                u[:, i] = val[:, 0]
                
            x = u[:, layer_const['inv_perm']]

        return x.reshape(features.shape[0] // n_samples, n_samples, self.param_dim)


# =============================================================================
# == Test Datasets and Visualization
# =============================================================================

def generate_test_data(dataset_name: str, n_samples: int, seed: int = 42):
    """
    Generates complex, conditional 2D datasets for testing.

    The first feature dimension is the conditioning variable.
    The two parameter dimensions are the target variables.

    Parameters
    ----------
    dataset_name : {'banana', 'student_t', 'moons'}
        The name of the dataset to generate.
    n_samples : int
        The number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[anp.ndarray, anp.ndarray]
        A tuple containing (parameters, features).
    """
    rng = anp.random.RandomState(seed)
    params = anp.zeros((n_samples, 2), dtype='f')
    features = anp.zeros((n_samples, 1), dtype='f')

    if dataset_name == 'banana':
        # Feature controls the curvature of the banana shape
        features[:, 0] = rng.uniform(0.5, 2.0, size=n_samples)
        x = rng.randn(n_samples, 2).astype('f')
        params[:, 0] = x[:, 0]
        params[:, 1] = x[:, 1] - features[:, 0] * (x[:, 0]**2 - 2.0)

    elif dataset_name == 'student_t':
        # Feature controls the degrees of freedom (tail heaviness)
        features[:, 0] = rng.uniform(1.0, 10.0, size=n_samples)
        for i in range(n_samples):
            df = features[i, 0]
            params[i, :] = t.rvs(df, size=2, random_state=rng)

    elif dataset_name == 'moons':
        # Feature controls the noise level
        features[:, 0] = rng.uniform(0.05, 0.2, size=n_samples)
        for i in range(n_samples):
            p, _ = make_moons(n_samples=2, noise=features[i, 0], random_state=rng)
            params[i, :] = p[0]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return params, features


def run_test(estimator: ConditionalDensityEstimator, dataset_name: str, plot: bool = True):
    """
    Runs a standardized test for a given estimator and dataset.

    Parameters
    ----------
    estimator : ConditionalDensityEstimator
        An instance of the estimator to test.
    dataset_name : str
        The name of the dataset to use for the test.
    plot : bool, optional
        If True, generates and displays a plot of the results.
    """
    print(f"\n--- Testing {estimator.__class__.__name__} on '{dataset_name}' dataset ---")

    # 1. Generate data and train
    params, features = generate_test_data(dataset_name, n_samples=5000)
    estimator.train(params, features, n_iter=400, learning_rate=1e-3)

    if not plot:
        return

    # 2. Setup for plotting
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Define conditions to test (low and high ends of the feature range)
    low_cond = anp.array([[features.min()]])
    high_cond = anp.array([[features.max()]])

    # 3. Generate samples from the trained model
    rng = anp.random.RandomState(0)
    samples_low = estimator.sample(low_cond, 1000, rng)[0]
    samples_high = estimator.sample(high_cond, 1000, rng)[0]

    # 4. Plot results
    # Create a grid to evaluate the learned density
    x_range = (params[:, 0].min() - 1, params[:, 0].max() + 1)
    y_range = (params[:, 1].min() - 1, params[:, 1].max() + 1)
    grid_x, grid_y = anp.meshgrid(anp.linspace(*x_range, 100), anp.linspace(*y_range, 100))
    grid_params = anp.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Plot for the 'low' condition
    grid_feat_low = anp.repeat(low_cond, grid_params.shape[0], axis=0)
    log_p_low = estimator.log_prob(grid_feat_low, grid_params).reshape(100, 100)
    ax1.contourf(grid_x, grid_y, anp.exp(log_p_low), levels=10, cmap='Blues', alpha=0.8)
    ax1.plot(samples_low[:, 0], samples_low[:, 1], ',', c='navy', alpha=0.5, label='Generated Samples')
    ax1.set_title(f"Condition: feature = {low_cond[0,0]:.2f}")
    ax1.set(xlim=x_range, ylim=y_range, xlabel="Param 1", ylabel="Param 2")
    ax1.legend()

    # Plot for the 'high' condition
    grid_feat_high = anp.repeat(high_cond, grid_params.shape[0], axis=0)
    log_p_high = estimator.log_prob(grid_feat_high, grid_params).reshape(100, 100)
    ax2.contourf(grid_x, grid_y, anp.exp(log_p_high), levels=10, cmap='Oranges', alpha=0.8)
    ax2.plot(samples_high[:, 0], samples_high[:, 1], ',', c='darkred', alpha=0.5, label='Generated Samples')
    ax2.set_title(f"Condition: feature = {high_cond[0,0]:.2f}")
    ax2.set(xlim=x_range, ylim=y_range, xlabel="Param 1", yticklabels=[])
    ax2.legend()

    fig.suptitle(f"Density Estimation Results for {estimator.__class__.__name__} on '{dataset_name}'", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and test conditional density estimators')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    
    plot = not args.no_plot
    
    if args.test:
        # Quick tests for CI
        print("Running quick tests...")
        mdn = MDNEstimator(param_dim=2, feature_dim=1, n_components=3, hidden_sizes=(16, 16))
        run_test(mdn, 'banana', plot=False)
        
        maf = MAFEstimator(param_dim=2, feature_dim=1, n_flows=2, hidden_units=32)
        run_test(maf, 'moons', plot=False)
        
        print("All tests passed!")
    else:
        # --- MDN Tests ---
        mdn_banana = MDNEstimator(param_dim=2, feature_dim=1, n_components=8, hidden_sizes=(64, 64))
        run_test(mdn_banana, 'banana', plot)

        mdn_moons = MDNEstimator(param_dim=2, feature_dim=1, n_components=10, hidden_sizes=(64, 64))
        run_test(mdn_moons, 'moons', plot)

        # --- MAF Tests ---
        maf_banana = MAFEstimator(param_dim=2, feature_dim=1, n_flows=5, hidden_units=128)
        run_test(maf_banana, 'banana', plot)

        maf_moons = MAFEstimator(param_dim=2, feature_dim=1, n_flows=5, hidden_units=128)
        run_test(maf_moons, 'moons', plot)