/**
 * @file maf.c
 * @brief MAF inference implementation
 */

#include "maf.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Simple random number generator for sampling (LCG) */
static uint32_t maf_rng_state = 12345;

static void maf_seed_rng(uint32_t seed) { maf_rng_state = seed; }

static float maf_randn(void) {
  /* Box-Muller transform for Gaussian samples */
  static int has_spare = 0;
  static float spare;

  if (has_spare) {
    has_spare = 0;
    return spare;
  }

  /* Generate two uniform random numbers */
  maf_rng_state = maf_rng_state * 1103515245 + 12345;
  float u1 = (float)(maf_rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
  maf_rng_state = maf_rng_state * 1103515245 + 12345;
  float u2 = (float)(maf_rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;

  /* Box-Muller */
  float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
  float theta = 2.0f * M_PI * u2;

  spare = r * sinf(theta);
  has_spare = 1;

  return r * cosf(theta);
}

/* =============================================================================
 * Helpers
 * ========================================================================== */

static void transpose_8_to_workspace(const float *src, float *dst, int dim) {
  /* src is [8][dim] (row-major), dst is [dim][8] (col-major) */
  for (int d = 0; d < dim; d++) {
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      dst[d * MAF_BATCH_SIZE + s] = src[s * dim + d];
    }
  }
}

static void broadcast_feature_to_workspace(const float *feature, float *dst,
                                           int dim) {
  /* replicate feature [dim] to [dim][8] */
  for (int d = 0; d < dim; d++) {
    float val = feature[d];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      dst[d * MAF_BATCH_SIZE + s] = val;
    }
  }
}

static void transpose_workspace_to_8(const float *src, float *dst, int dim) {
  /* src is [dim][8], dst is [8][dim] */
  for (int s = 0; s < MAF_BATCH_SIZE; s++) {
    for (int d = 0; d < dim; d++) {
      dst[s * dim + d] = src[d * MAF_BATCH_SIZE + s];
    }
  }
}

/* =============================================================================
 * Model Loading and Memory Management
 * ========================================================================== */

maf_model_t *maf_load_model(const maf_weights_t *weights) {
  if (weights == NULL) {
    return NULL;
  }

  maf_model_t *model = (maf_model_t *)malloc(sizeof(maf_model_t));
  if (model == NULL) {
    return NULL;
  }

  model->n_flows = weights->n_flows;
  model->param_dim = weights->param_dim;
  model->feature_dim = weights->feature_dim;

  /* Allocate layers array */
  model->layers = (maf_layer_t *)calloc(weights->n_flows, sizeof(maf_layer_t));
  if (model->layers == NULL) {
    free(model);
    return NULL;
  }

  uint16_t D = weights->param_dim;
  uint16_t C = weights->feature_dim;
  uint16_t H = weights->hidden_units;

  /* Offset into flattened arrays */
  size_t m1_offset = 0;
  size_t m2_offset = 0;
  size_t perm_offset = 0;
  size_t w1y_offset = 0;
  size_t w1c_offset = 0;
  size_t b1_offset = 0;
  size_t w2_offset = 0;
  size_t w2c_offset = 0;
  size_t b2_offset = 0;

  /* Load each layer */
  for (uint16_t k = 0; k < weights->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];

    layer->param_dim = D;
    layer->feature_dim = C;
    layer->hidden_units = H;

    /* Allocate and copy masks */
    layer->M1 = (float *)malloc(H * D * sizeof(float));
    layer->M2 = (float *)malloc(D * H * sizeof(float));
    layer->perm = (uint16_t *)malloc(D * sizeof(uint16_t));
    layer->inv_perm = (uint16_t *)malloc(D * sizeof(uint16_t));

    if (!layer->M1 || !layer->M2 || !layer->perm || !layer->inv_perm) {
      maf_free_model(model);
      return NULL;
    }

    memcpy(layer->M1, &weights->M1_data[m1_offset], H * D * sizeof(float));
    memcpy(layer->M2, &weights->M2_data[m2_offset], D * H * sizeof(float));
    memcpy(layer->perm, &weights->perm_data[perm_offset], D * sizeof(uint16_t));
    memcpy(layer->inv_perm, &weights->inv_perm_data[perm_offset],
           D * sizeof(uint16_t));

    m1_offset += H * D;
    m2_offset += D * H;
    perm_offset += D;

    /* Allocate and copy weights */
    layer->W1y = (float *)malloc(H * D * sizeof(float));
    layer->W1c = (float *)malloc(H * C * sizeof(float));
    layer->b1 = (float *)malloc(H * sizeof(float));
    layer->W2 = (float *)malloc(2 * D * H * sizeof(float));
    layer->W2c = (float *)malloc(2 * D * C * sizeof(float));
    layer->b2 = (float *)malloc(2 * D * sizeof(float));

    if (!layer->W1y || !layer->W1c || !layer->b1 || !layer->W2 || !layer->W2c ||
        !layer->b2) {
      maf_free_model(model);
      return NULL;
    }

    memcpy(layer->W1y, &weights->W1y_data[w1y_offset], H * D * sizeof(float));
    memcpy(layer->W1c, &weights->W1c_data[w1c_offset], H * C * sizeof(float));
    memcpy(layer->b1, &weights->b1_data[b1_offset], H * sizeof(float));
    memcpy(layer->W2, &weights->W2_data[w2_offset], 2 * D * H * sizeof(float));
    memcpy(layer->W2c, &weights->W2c_data[w2c_offset],
           2 * D * C * sizeof(float));
    memcpy(layer->b2, &weights->b2_data[b2_offset], 2 * D * sizeof(float));

    w1y_offset += H * D;
    w1c_offset += H * C;
    b1_offset += H;
    w2_offset += 2 * D * H;
    w2c_offset += 2 * D * C;
    b2_offset += 2 * D;
  }

  return model;
}

void maf_free_model(maf_model_t *model) {
  if (model == NULL) {
    return;
  }

  if (model->layers != NULL) {
    for (uint16_t k = 0; k < model->n_flows; k++) {
      maf_layer_t *layer = &model->layers[k];
      free(layer->M1);
      free(layer->M2);
      free(layer->perm);
      free(layer->inv_perm);
      free(layer->W1y);
      free(layer->W1c);
      free(layer->b1);
      free(layer->W2);
      free(layer->W2c);
      free(layer->b2);
    }
    free(model->layers);
  }

  free(model);
}

maf_workspace_t *maf_create_workspace(const maf_model_t *model) {
  if (model == NULL) {
    return NULL;
  }

  maf_workspace_t *ws = (maf_workspace_t *)calloc(1, sizeof(maf_workspace_t));
  if (ws == NULL) {
    return NULL;
  }

  uint16_t max_H = 0;
  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;

  for (uint16_t k = 0; k < model->n_flows; k++) {
    if (model->layers[k].hidden_units > max_H) {
      max_H = model->layers[k].hidden_units;
    }
  }

  /* Allocate buffers - scaled by MAF_BATCH_SIZE */
  ws->h = (float *)malloc(max_H * MAF_BATCH_SIZE * sizeof(float));
  ws->out = (float *)malloc(2 * D * MAF_BATCH_SIZE * sizeof(float));
  ws->u = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->u_perm = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->mu = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->alpha = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->x = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->y_perm = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  ws->feat_perm = (float *)malloc(C * MAF_BATCH_SIZE * sizeof(float));

  /* Check allocation success */
  if (!ws->h || !ws->out || !ws->u || !ws->u_perm || !ws->mu || !ws->alpha ||
      !ws->x || !ws->y_perm || !ws->feat_perm) {
    maf_free_workspace(ws);
    return NULL;
  }

  return ws;
}

void maf_free_workspace(maf_workspace_t *ws) {
  if (ws == NULL) {
    return;
  }

  free(ws->h);
  free(ws->out);
  free(ws->u);
  free(ws->u_perm);
  free(ws->mu);
  free(ws->alpha);
  free(ws->x);
  free(ws->y_perm);
  free(ws->feat_perm);
  free(ws);
}

size_t maf_get_memory_usage(const maf_model_t *model) {
  if (model == NULL) {
    return 0;
  }

  size_t total = sizeof(maf_model_t);
  total += model->n_flows * sizeof(maf_layer_t);

  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;

  for (uint16_t k = 0; k < model->n_flows; k++) {
    uint16_t H = model->layers[k].hidden_units;

    /* Masks and permutations */
    total += H * D * sizeof(float); /* M1 */
    total += D * H * sizeof(float); /* M2 */
    total += D * sizeof(uint16_t);  /* perm */
    total += D * sizeof(uint16_t);  /* inv_perm */

    /* Weights */
    total += H * D * sizeof(float);     /* W1y */
    total += H * C * sizeof(float);     /* W1c */
    total += H * sizeof(float);         /* b1 */
    total += 2 * D * H * sizeof(float); /* W2 */
    total += 2 * D * C * sizeof(float); /* W2c */
    total += 2 * D * sizeof(float);     /* b2 */
  }

  return total;
}

/* =============================================================================
 * MADE Forward Pass (Batched)
 * ========================================================================== */

void maf_made_forward(const maf_layer_t *layer, maf_workspace_t *ws,
                      const float *y, const float *context, float *mu_out,
                      float *alpha_out) {
  uint16_t D = layer->param_dim;
  uint16_t C = layer->feature_dim;
  uint16_t H = layer->hidden_units;

  /* Use workspace buffers */
  float *h = ws->h;
  float *out = ws->out;

  /* Hidden layer: h = tanh((y @ W1y.T) * M1 + (ctx @ W1c.T) + b1) */
  /* Input layout: y[D][8], context[C][8] */
  /* Output layout: h[H][8] */

  for (uint16_t i = 0; i < H; i++) {
    float sum[MAF_BATCH_SIZE];
    float bias = layer->b1[i];

    for (int s = 0; s < MAF_BATCH_SIZE; s++)
      sum[s] = bias;

    /* y @ W1y.T * M1 */
    for (uint16_t j = 0; j < D; j++) {
      float w = layer->W1y[i * D + j] * layer->M1[i * D + j];
      const float *y_vec = &y[j * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        sum[s] += y_vec[s] * w;
      }
    }

    /* context @ W1c.T */
    for (uint16_t j = 0; j < C; j++) {
      float w = layer->W1c[i * C + j];
      const float *ctx_vec = &context[j * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        sum[s] += ctx_vec[s] * w;
      }
    }

    float *h_vec = &h[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      h_vec[s] = tanhf(sum[s]);
    }
  }

  /* Output layer: out = (h @ W2.T) * M2 + (ctx @ W2c.T) + b2 */
  /* Output layout: out[2*D][8] */

  for (uint16_t i = 0; i < 2 * D; i++) {
    float sum[MAF_BATCH_SIZE];
    float bias = layer->b2[i];

    for (int s = 0; s < MAF_BATCH_SIZE; s++)
      sum[s] = bias;

    /* h @ W2.T * M2_tiled */
    uint16_t d_idx = i % D; /* Which dimension of D */
    for (uint16_t j = 0; j < H; j++) {
      float w = layer->W2[i * H + j] * layer->M2[d_idx * H + j];
      const float *h_vec = &h[j * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        sum[s] += h_vec[s] * w;
      }
    }

    /* context @ W2c.T */
    for (uint16_t j = 0; j < C; j++) {
      float w = layer->W2c[i * C + j];
      const float *ctx_vec = &context[j * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        sum[s] += ctx_vec[s] * w;
      }
    }

    float *out_vec = &out[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      out_vec[s] = sum[s];
    }
  }

  /* Split into mu and alpha */
  for (uint16_t i = 0; i < D; i++) {
    const float *mu_src = &out[i * MAF_BATCH_SIZE];
    const float *alpha_src = &out[(D + i) * MAF_BATCH_SIZE];
    float *mu_dst = &mu_out[i * MAF_BATCH_SIZE];
    float *alpha_dst = &alpha_out[i * MAF_BATCH_SIZE];

    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      mu_dst[s] = mu_src[s];
      /* Clip alpha to [-7, 7] for numerical stability */
      alpha_dst[s] = fminf(fmaxf(alpha_src[s], -7.0f), 7.0f);
    }
  }
}

/* =============================================================================
 * Inverse Layer (for Sampling)
 * ========================================================================== */

void maf_inverse_layer(const maf_layer_t *layer, maf_workspace_t *ws,
                       const float *y_perm, const float *context,
                       float *x_out) {
  uint16_t D = layer->param_dim;

  /* Use workspace buffers */
  float *u = ws->u;
  float *mu = ws->mu;
  float *alpha = ws->alpha;

  /* Clear u buffer (important for autoregressive property) */
  memset(u, 0, D * MAF_BATCH_SIZE * sizeof(float));

  /* Autoregressive inversion: for each dimension in order */
  for (uint16_t i = 0; i < D; i++) {
    /* Compute mu and alpha conditioned on u[:i] */
    maf_made_forward(layer, ws, u, context, mu, alpha);

    /* Invert: u[i] = y_perm[i] * exp(alpha[i]) + mu[i] */
    float *u_vec = &u[i * MAF_BATCH_SIZE];
    const float *y_vec = &y_perm[i * MAF_BATCH_SIZE];
    const float *mu_vec = &mu[i * MAF_BATCH_SIZE];
    const float *alpha_vec = &alpha[i * MAF_BATCH_SIZE];

    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      u_vec[s] = y_vec[s] * expf(alpha_vec[s]) + mu_vec[s];
    }
  }

  /* Apply inverse permutation */
  for (uint16_t i = 0; i < D; i++) {
    float *dst = &x_out[layer->inv_perm[i] * MAF_BATCH_SIZE];
    float *src = &u[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      dst[s] = src[s];
    }
  }
}

/* =============================================================================
 * Sampling
 * ========================================================================== */

int maf_sample_from_noise(const maf_model_t *model, const float *features,
                          const float *base_noise, uint32_t n_samples,
                          float *samples_out) {
  if (model == NULL || features == NULL || base_noise == NULL ||
      samples_out == NULL) {
    return -1;
  }

  if (n_samples % MAF_BATCH_SIZE != 0) {
    return -3; /* Error: n_samples must be multiple of MAF_BATCH_SIZE */
  }

  /* Create workspace for this batch */
  maf_workspace_t *ws = maf_create_workspace(model);
  if (ws == NULL) {
    return -2;
  }

  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;
  float *x = ws->x;                // [D * 8]
  float *y_perm = ws->y_perm;      // [D * 8]
  float *ctx_perm = ws->feat_perm; // [C * 8]

  /* Pre-broadcast features to workspace */
  broadcast_feature_to_workspace(features, ctx_perm, C);

  /* Generate samples in chunks */
  for (uint32_t s_idx = 0; s_idx < n_samples; s_idx += MAF_BATCH_SIZE) {
    // Determine if we have a full batch
    // Since implementation is strict, we might read/write past buffer if not
    // careful? But internal buffers are safe. Input/Output buffers are caller's
    // responsibility. If n_samples is not multiple of 8, we might read garbage
    // from base_noise or write garbage to samples_out. The prompt says
    // "validate... multiple of 8". So we assume it is.

    // Load base noise [8][D] -> x [D][8]
    transpose_8_to_workspace(&base_noise[s_idx * D], x, D);

    /* Invert flow stack (reverse order) */
    for (int k = (int)model->n_flows - 1; k >= 0; k--) {
      const maf_layer_t *layer = &model->layers[k];

      /* Copy x to y_perm for input to inverse layer */
      memcpy(y_perm, x, D * MAF_BATCH_SIZE * sizeof(float));

      /* Invert layer */
      maf_inverse_layer(layer, ws, y_perm, ctx_perm, x);
    }

    /* Copy result to output [D][8] -> [8][D] */
    transpose_workspace_to_8(x, &samples_out[s_idx * D], D);
  }

  maf_free_workspace(ws);
  return 0;
}

int maf_sample(const maf_model_t *model, const float *features,
               uint32_t n_samples, float *samples_out, uint32_t seed) {
  if (model == NULL || features == NULL || samples_out == NULL) {
    return -1;
  }

  maf_seed_rng(seed);
  uint16_t D = model->param_dim;

  /* Allocate base noise buffer */
  float *base_noise = (float *)malloc(n_samples * D * sizeof(float));
  if (base_noise == NULL) {
    return -2;
  }

  /* Generate standard Gaussian noise */
  for (uint32_t s = 0; s < n_samples; s++) {
    for (uint16_t i = 0; i < D; i++) {
      base_noise[s * D + i] = maf_randn();
    }
  }

  /* Use the deterministic transformation */
  int ret = maf_sample_from_noise(model, features, base_noise, n_samples,
                                  samples_out);

  free(base_noise);
  return ret;
}

/* =============================================================================
 * Log Probability
 * ========================================================================== */

float maf_log_prob(const maf_model_t *model, maf_workspace_t *ws,
                   const float *features, const float *params) {
  if (model == NULL || features == NULL || params == NULL) {
    return -INFINITY;
  }

  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;

  /* Use workspace buffers */
  float *u = ws->u;
  float *u_perm = ws->u_perm;
  float *mu = ws->mu;
  float *alpha = ws->alpha;
  float *feat_perm = ws->feat_perm;

  // Transpose inputs to [D][8] and [C][8]
  transpose_8_to_workspace(params, u, D);
  transpose_8_to_workspace(features, feat_perm, C);

  float log_det = 0.0f;

  /* Forward through flow stack */
  for (uint16_t k = 0; k < model->n_flows; k++) {
    const maf_layer_t *layer = &model->layers[k];

    /* Apply permutation */
    for (uint16_t i = 0; i < D; i++) {
      float *dst = &u_perm[i * MAF_BATCH_SIZE];
      float *src = &u[layer->perm[i] * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++)
        dst[s] = src[s];
    }

    /* Forward pass */
    maf_made_forward(layer, ws, u_perm, feat_perm, mu, alpha);

    /* Transform: u = (u - mu) * exp(-alpha) */
    for (uint16_t i = 0; i < D; i++) {
      float *u_dst = &u[i * MAF_BATCH_SIZE];
      float *u_src = &u_perm[i * MAF_BATCH_SIZE];
      float *mu_vec = &mu[i * MAF_BATCH_SIZE];
      float *alpha_vec = &alpha[i * MAF_BATCH_SIZE];

      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        u_dst[s] = (u_src[s] - mu_vec[s]) * expf(-alpha_vec[s]);
        log_det -= alpha_vec[s];
      }
    }
  }

  /* Base distribution: N(0, I) */
  float base_logp = 0.0f;
  for (uint16_t i = 0; i < D; i++) {
    float *u_vec = &u[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      base_logp -= 0.5f * u_vec[s] * u_vec[s];
    }
  }
  // Correct constant term for batch of 8
  base_logp -= 0.5f * D * MAF_BATCH_SIZE * logf(2.0f * M_PI);

  return base_logp + log_det;
}

/* =============================================================================
 * Training Utilities
 * ========================================================================== */

maf_cache_t *maf_create_cache(const maf_model_t *model) {
  if (model == NULL)
    return NULL;

  maf_cache_t *cache = (maf_cache_t *)malloc(sizeof(maf_cache_t));
  if (!cache)
    return NULL;

  cache->n_flows = model->n_flows;
  cache->layers =
      (maf_layer_cache_t *)calloc(model->n_flows, sizeof(maf_layer_cache_t));
  if (!cache->layers) {
    free(cache);
    return NULL;
  }

  for (uint16_t k = 0; k < model->n_flows; k++) {
    uint16_t D = model->layers[k].param_dim;
    uint16_t H = model->layers[k].hidden_units;

    cache->layers[k].input =
        (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
    cache->layers[k].h = (float *)malloc(H * MAF_BATCH_SIZE * sizeof(float));
    cache->layers[k].mu = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
    cache->layers[k].alpha =
        (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));

    if (!cache->layers[k].input || !cache->layers[k].h ||
        !cache->layers[k].mu || !cache->layers[k].alpha) {
      maf_free_cache(cache);
      return NULL;
    }
  }

  return cache;
}

void maf_free_cache(maf_cache_t *cache) {
  if (cache == NULL)
    return;

  if (cache->layers) {
    for (uint16_t k = 0; k < cache->n_flows; k++) {
      free(cache->layers[k].input);
      free(cache->layers[k].h);
      free(cache->layers[k].mu);
      free(cache->layers[k].alpha);
    }
    free(cache->layers);
  }
  free(cache);
}

maf_grad_t *maf_create_grad(const maf_model_t *model) {
  if (model == NULL)
    return NULL;

  maf_grad_t *grad = (maf_grad_t *)malloc(sizeof(maf_grad_t));
  if (!grad)
    return NULL;

  grad->n_flows = model->n_flows;
  grad->layers =
      (maf_layer_grad_t *)calloc(model->n_flows, sizeof(maf_layer_grad_t));
  if (!grad->layers) {
    free(grad);
    return NULL;
  }

  for (uint16_t k = 0; k < model->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    grad->layers[k].dW1y = (float *)malloc(H * D * sizeof(float));
    grad->layers[k].dW1c = (float *)malloc(H * C * sizeof(float));
    grad->layers[k].db1 = (float *)malloc(H * sizeof(float));
    grad->layers[k].dW2 = (float *)malloc(2 * D * H * sizeof(float));
    grad->layers[k].dW2c = (float *)malloc(2 * D * C * sizeof(float));
    grad->layers[k].db2 = (float *)malloc(2 * D * sizeof(float));

    if (!grad->layers[k].dW1y || !grad->layers[k].dW1c ||
        !grad->layers[k].db1 || !grad->layers[k].dW2 || !grad->layers[k].dW2c ||
        !grad->layers[k].db2) {
      maf_free_grad(grad);
      return NULL;
    }
  }

  maf_zero_grad(model, grad);
  return grad;
}

void maf_free_grad(maf_grad_t *grad) {
  if (grad == NULL)
    return;

  if (grad->layers) {
    for (uint16_t k = 0; k < grad->n_flows; k++) {
      free(grad->layers[k].dW1y);
      free(grad->layers[k].dW1c);
      free(grad->layers[k].db1);
      free(grad->layers[k].dW2);
      free(grad->layers[k].dW2c);
      free(grad->layers[k].db2);
    }
    free(grad->layers);
  }
  free(grad);
}

void maf_zero_grad(const maf_model_t *model, maf_grad_t *grad) {
  if (model == NULL || grad == NULL)
    return;

  for (uint16_t k = 0; k < model->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    memset(grad->layers[k].dW1y, 0, H * D * sizeof(float));
    memset(grad->layers[k].dW1c, 0, H * C * sizeof(float));
    memset(grad->layers[k].db1, 0, H * sizeof(float));
    memset(grad->layers[k].dW2, 0, 2 * D * H * sizeof(float));
    memset(grad->layers[k].dW2c, 0, 2 * D * C * sizeof(float));
    memset(grad->layers[k].db2, 0, 2 * D * sizeof(float));
  }
}

float maf_forward_train(const maf_model_t *model, maf_workspace_t *ws,
                        maf_cache_t *cache, const float *features,
                        const float *params) {
  /* Wraps log_prob logic but stores cache */
  if (model == NULL || ws == NULL || cache == NULL || features == NULL ||
      params == NULL) {
    return -INFINITY;
  }

  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;

  /* Use workspace buffers */
  float *u = ws->u;
  float *u_perm = ws->u_perm;
  float *mu = ws->mu;
  float *alpha = ws->alpha;
  float *feat_perm = ws->feat_perm;

  /* Transpose inputs */
  transpose_8_to_workspace(params, u, D);
  transpose_8_to_workspace(features, feat_perm, C);

  float log_det = 0.0f;

  /* Forward through flow stack */
  for (uint16_t k = 0; k < model->n_flows; k++) {
    const maf_layer_t *layer = &model->layers[k];
    maf_layer_cache_t *lcache = &cache->layers[k];

    /* Apply permutation */
    for (uint16_t i = 0; i < D; i++) {
      float *dst = &u_perm[i * MAF_BATCH_SIZE];
      float *src = &u[layer->perm[i] * MAF_BATCH_SIZE];
      for (int s = 0; s < MAF_BATCH_SIZE; s++)
        dst[s] = src[s];
    }

    /* Store input to cache */
    memcpy(lcache->input, u_perm, D * MAF_BATCH_SIZE * sizeof(float));

    /* Forward pass */
    maf_made_forward(layer, ws, u_perm, feat_perm, mu, alpha);

    /* Store activations to cache */
    memcpy(lcache->h, ws->h,
           layer->hidden_units * MAF_BATCH_SIZE * sizeof(float));
    memcpy(lcache->mu, mu, D * MAF_BATCH_SIZE * sizeof(float));
    memcpy(lcache->alpha, alpha, D * MAF_BATCH_SIZE * sizeof(float));

    /* Transform: u = (u - mu) * exp(-alpha) */
    for (uint16_t i = 0; i < D; i++) {
      float *u_dst = &u[i * MAF_BATCH_SIZE];
      float *u_src = &u_perm[i * MAF_BATCH_SIZE];
      float *mu_vec = &mu[i * MAF_BATCH_SIZE];
      float *alpha_vec = &alpha[i * MAF_BATCH_SIZE];

      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        u_dst[s] = (u_src[s] - mu_vec[s]) * expf(-alpha_vec[s]);
        log_det -= alpha_vec[s];
      }
    }
  }

  /* Base distribution: N(0, I) */
  float base_logp = 0.0f;
  for (uint16_t i = 0; i < D; i++) {
    float *u_vec = &u[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      base_logp -= 0.5f * u_vec[s] * u_vec[s];
    }
  }
  base_logp -= 0.5f * D * MAF_BATCH_SIZE * logf(2.0f * M_PI);

  return base_logp + log_det;
}

/* =============================================================================
 * Model Initialization
 * ========================================================================== */

static float maf_rand_f() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

maf_model_t *maf_init_random_model(uint16_t n_flows, uint16_t param_dim,
                                   uint16_t feature_dim,
                                   uint16_t hidden_units) {
  int n = n_flows;
  int D = param_dim;
  int C = feature_dim;
  int H = hidden_units;

  /* Temporary buffers for initialization */
  float *t_M1 = malloc(n * H * D * sizeof(float));
  float *t_M2 = malloc(n * D * H * sizeof(float));
  uint16_t *t_perm = malloc(n * D * sizeof(uint16_t));
  uint16_t *t_inv_perm = malloc(n * D * sizeof(uint16_t));
  float *t_W1y = malloc(n * H * D * sizeof(float));
  float *t_W1c = malloc(n * H * C * sizeof(float));
  float *t_b1 = malloc(n * H * sizeof(float));
  float *t_W2 = malloc(n * 2 * D * H * sizeof(float));
  float *t_W2c = malloc(n * 2 * D * C * sizeof(float));
  float *t_b2 = malloc(n * 2 * D * sizeof(float));

  if (!t_M1 || !t_M2 || !t_perm || !t_inv_perm || !t_W1y || !t_W1c || !t_b1 ||
      !t_W2 || !t_W2c || !t_b2) {
    /* Simple error handling: free all and return NULL */
    free(t_M1);
    free(t_M2);
    free(t_perm);
    free(t_inv_perm);
    free(t_W1y);
    free(t_W1c);
    free(t_b1);
    free(t_W2);
    free(t_W2c);
    free(t_b2);
    return NULL;
  }

  /* Initialize proper MADE masks */
  for (int k = 0; k < n; k++) {
    /* 1. Create Permutation */
    for (int i = 0; i < D; i++)
      t_perm[k * D + i] = i;

    /* Shuffle permutation */
    for (int i = 0; i < D; i++) {
      int j = rand() % D;
      uint16_t tmp = t_perm[k * D + i];
      t_perm[k * D + i] = t_perm[k * D + j];
      t_perm[k * D + j] = tmp;
    }

    /* Compute inverse */
    for (int i = 0; i < D; i++)
      t_inv_perm[k * D + t_perm[k * D + i]] = i;

    /* 2. Assign Degrees */
    int *m_in = malloc(D * sizeof(int));
    int *m_h = malloc(H * sizeof(int));

    for (int i = 0; i < D; i++)
      m_in[i] = i + 1; /* 1-based degrees */

    for (int j = 0; j < H; j++) {
      if (D > 1)
        m_h[j] = (rand() % (D - 1)) + 1;
      else
        m_h[j] = 0;
    }

    /* 3. Build Masks */
    /* M1: H x D */
    for (int j = 0; j < H; j++) {
      for (int i = 0; i < D; i++) {
        if (m_in[i] <= m_h[j])
          t_M1[k * H * D + j * D + i] = 1.0f;
        else
          t_M1[k * H * D + j * D + i] = 0.0f;

        /* Initialize Weights */
        t_W1y[k * H * D + j * D + i] = maf_rand_f() * 0.01f;
      }

      /* Initialize Biases and Context Weights */
      for (int c = 0; c < C; c++)
        t_W1c[k * H * C + j * C + c] = maf_rand_f() * 0.01f;
      t_b1[k * H + j] = 0.0f;
    }

    /* M2: 2D x H */
    for (int d = 0; d < D; d++) {
      int m_out = d + 1;
      /* Iterate hidden units */
      for (int j = 0; j < H; j++) {
        float val = (m_h[j] < m_out) ? 1.0f : 0.0f;
        t_M2[k * D * H + d * H + j] = val;
        t_W2[k * 2 * D * H + d * H + j] = maf_rand_f() * 0.01f;
        t_W2[k * 2 * D * H + (D + d) * H + j] = maf_rand_f() * 0.01f;
      }

      /* Biases and Context */
      for (int c = 0; c < C; c++) {
        t_W2c[k * 2 * D * C + d * C + c] = maf_rand_f() * 0.01f;
        t_W2c[k * 2 * D * C + (D + d) * C + c] = maf_rand_f() * 0.01f;
      }
      t_b2[k * 2 * D + d] = 0.0f;
      t_b2[k * 2 * D + (D + d)] = 0.0f;
    }

    free(m_in);
    free(m_h);
  }

  maf_weights_t w;
  w.n_flows = n;
  w.param_dim = D;
  w.feature_dim = C;
  w.hidden_units = H;
  w.M1_data = t_M1;
  w.M2_data = t_M2;
  w.perm_data = t_perm;
  w.inv_perm_data = t_inv_perm;
  w.W1y_data = t_W1y;
  w.W1c_data = t_W1c;
  w.b1_data = t_b1;
  w.W2_data = t_W2;
  w.W2c_data = t_W2c;
  w.b2_data = t_b2;

  maf_model_t *model = maf_load_model(&w);

  /* Free temporary buffers */
  free(t_M1);
  free(t_M2);
  free(t_perm);
  free(t_inv_perm);
  free(t_W1y);
  free(t_W1c);
  free(t_b1);
  free(t_W2);
  free(t_W2c);
  free(t_b2);

  return model;
}

/* =============================================================================
 * Backward Pass
 * ========================================================================== */

static void
maf_layer_backward(const maf_layer_t *layer, const maf_layer_cache_t *lcache,
                   maf_layer_grad_t *lgrad,
                   const float *features,      /* [C][8] transposed */
                   const float *grad_from_top, /* delta_out [D][8] */
                   float *grad_to_bottom)      /* delta_in [D][8] */
{
  uint16_t D = layer->param_dim;
  uint16_t C = layer->feature_dim;
  uint16_t H = layer->hidden_units;

  /* Allocate temporary gradient buffers */
  float *d_out = (float *)calloc(2 * D * MAF_BATCH_SIZE, sizeof(float));
  float *d_h = (float *)calloc(H * MAF_BATCH_SIZE, sizeof(float));

  if (!d_out || !d_h) {
    free(d_out);
    free(d_h);
    return;
  }

  /* 1. Gradients wrt mu and alpha */
  /* u_out = (u_in - mu) * exp(-alpha) */
  for (uint16_t i = 0; i < D; i++) {
    const float *u_in = &lcache->input[i * MAF_BATCH_SIZE];
    const float *mu = &lcache->mu[i * MAF_BATCH_SIZE];
    const float *alpha = &lcache->alpha[i * MAF_BATCH_SIZE];
    const float *delta = &grad_from_top[i * MAF_BATCH_SIZE];

    float *d_mu = &d_out[i * MAF_BATCH_SIZE];
    float *d_alpha = &d_out[(D + i) * MAF_BATCH_SIZE];

    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      float exp_neg_alpha = expf(-alpha[s]);
      float u_out = (u_in[s] - mu[s]) * exp_neg_alpha;

      d_mu[s] = delta[s] * (-exp_neg_alpha);
      d_alpha[s] = delta[s] * (-u_out) + 1.0f;
    }
  }

  /* 2. Backprop through Output Layer (W2, b2) */
  /* out = (h @ W2.T) * M2 + (ctx @ W2c.T) + b2 */

  for (uint16_t i = 0; i < 2 * D; i++) {
    const float *delta = &d_out[i * MAF_BATCH_SIZE];

    /* Accumulate db2 */
    for (int s = 0; s < MAF_BATCH_SIZE; s++)
      lgrad->db2[i] += delta[s];

    /* Accumulate dW2c */
    for (uint16_t j = 0; j < C; j++) {
      const float *feat_vec = &features[j * MAF_BATCH_SIZE];
      float grad_sum = 0;
      for (int s = 0; s < MAF_BATCH_SIZE; s++)
        grad_sum += delta[s] * feat_vec[s];
      lgrad->dW2c[i * C + j] += grad_sum;
    }

    /* Backprop to h and Accumulate dW2 */
    uint16_t d_idx = i % D;
    for (uint16_t j = 0; j < H; j++) {
      float m2_val = layer->M2[d_idx * H + j];
      if (m2_val != 0.0f) {
        const float *h_vec = &lcache->h[j * MAF_BATCH_SIZE];
        float *dh_vec = &d_h[j * MAF_BATCH_SIZE];

        float w = layer->W2[i * H + j];
        float grad_sum = 0;

        for (int s = 0; s < MAF_BATCH_SIZE; s++) {
          float d = delta[s];
          dh_vec[s] += d * w * m2_val;
          grad_sum += d * h_vec[s] * m2_val;
        }
        lgrad->dW2[i * H + j] += grad_sum;
      }
    }
  }

  /* 3. Backprop through Tanh */
  for (uint16_t i = 0; i < H; i++) {
    const float *h_vec = &lcache->h[i * MAF_BATCH_SIZE];
    float *dh_vec = &d_h[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      dh_vec[s] *= (1.0f - h_vec[s] * h_vec[s]);
    }
  }

  /* 4. Backprop through Input Layer (W1, b1) */
  float *d_u_in = (float *)calloc(D * MAF_BATCH_SIZE, sizeof(float));
  if (!d_u_in) {
    free(d_out);
    free(d_h);
    return;
  }

  for (uint16_t i = 0; i < H; i++) {
    const float *delta = &d_h[i * MAF_BATCH_SIZE];

    /* Accumulate db1 */
    for (int s = 0; s < MAF_BATCH_SIZE; s++)
      lgrad->db1[i] += delta[s];

    /* Accumulate dW1c */
    for (uint16_t j = 0; j < C; j++) {
      const float *feat_vec = &features[j * MAF_BATCH_SIZE];
      float grad_sum = 0;
      for (int s = 0; s < MAF_BATCH_SIZE; s++)
        grad_sum += delta[s] * feat_vec[s];
      lgrad->dW1c[i * C + j] += grad_sum;
    }

    /* Backprop to u_in and Accumulate dW1y */
    for (uint16_t j = 0; j < D; j++) {
      float m1_val = layer->M1[i * D + j];
      if (m1_val != 0.0f) {
        const float *inp_vec = &lcache->input[j * MAF_BATCH_SIZE];
        float *du_vec = &d_u_in[j * MAF_BATCH_SIZE];

        float w = layer->W1y[i * D + j];
        float grad_sum = 0;

        for (int s = 0; s < MAF_BATCH_SIZE; s++) {
          float d = delta[s];
          du_vec[s] += d * w * m1_val;
          grad_sum += d * inp_vec[s] * m1_val;
        }
        lgrad->dW1y[i * D + j] += grad_sum;
      }
    }
  }

  /* 5. Combine gradients for u_in */
  for (uint16_t i = 0; i < D; i++) {
    float *du_vec = &d_u_in[i * MAF_BATCH_SIZE];
    const float *alpha = &lcache->alpha[i * MAF_BATCH_SIZE];
    const float *delta_top = &grad_from_top[i * MAF_BATCH_SIZE];

    for (int s = 0; s < MAF_BATCH_SIZE; s++) {
      du_vec[s] += delta_top[s] * expf(-alpha[s]);
    }
  }

  /* 6. Inverse Permutation to get grad_to_bottom */
  for (uint16_t i = 0; i < D; i++) {
    float *dst = &grad_to_bottom[layer->perm[i] * MAF_BATCH_SIZE];
    float *src = &d_u_in[i * MAF_BATCH_SIZE];
    for (int s = 0; s < MAF_BATCH_SIZE; s++)
      dst[s] = src[s];
  }

  free(d_out);
  free(d_h);
  free(d_u_in);
}

int maf_backward(const maf_model_t *model, const maf_cache_t *cache,
                 maf_grad_t *grad, const float *features, const float *params) {
  if (model == NULL || cache == NULL || grad == NULL || features == NULL ||
      params == NULL) {
    return -1;
  }

  uint16_t D = model->param_dim;
  uint16_t C = model->feature_dim;

  /* Allocate gradient buffers for flow [D][8] */
  float *delta = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  float *prev_delta = (float *)malloc(D * MAF_BATCH_SIZE * sizeof(float));
  float *feat_perm = (float *)malloc(C * MAF_BATCH_SIZE * sizeof(float));

  if (!delta || !prev_delta || !feat_perm) {
    free(delta);
    free(prev_delta);
    free(feat_perm);
    return -2;
  }

  /* Transpose features for backward pass */
  transpose_8_to_workspace(features, feat_perm, C);

  /* Initialize delta with gradient of base distribution */
  /* Recompute final u */
  {
    const maf_layer_cache_t *last_cache = &cache->layers[model->n_flows - 1];
    for (uint16_t i = 0; i < D; i++) {
      const float *u_in = &last_cache->input[i * MAF_BATCH_SIZE];
      const float *mu = &last_cache->mu[i * MAF_BATCH_SIZE];
      const float *alpha = &last_cache->alpha[i * MAF_BATCH_SIZE];
      float *d_vec = &delta[i * MAF_BATCH_SIZE];

      for (int s = 0; s < MAF_BATCH_SIZE; s++) {
        d_vec[s] = (u_in[s] - mu[s]) * expf(-alpha[s]);
      }
    }
  }

  /* Backpropagate through layers */
  for (int k = model->n_flows - 1; k >= 0; k--) {
    maf_layer_backward(&model->layers[k], &cache->layers[k], &grad->layers[k],
                       feat_perm, delta, prev_delta);

    /* Swap buffers for next iteration */
    memcpy(delta, prev_delta, D * MAF_BATCH_SIZE * sizeof(float));
  }

  free(delta);
  free(prev_delta);
  free(feat_perm);
  return 0;
}

void maf_sgd_step(maf_model_t *model, const maf_grad_t *grad, float lr) {
  if (model == NULL || grad == NULL)
    return;

  for (uint16_t k = 0; k < model->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];
    const maf_layer_grad_t *lgrad = &grad->layers[k];

    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    for (uint32_t i = 0; i < (uint32_t)H * D; i++)
      layer->W1y[i] -= lr * lgrad->dW1y[i];
    for (uint32_t i = 0; i < (uint32_t)H * C; i++)
      layer->W1c[i] -= lr * lgrad->dW1c[i];
    for (uint32_t i = 0; i < H; i++)
      layer->b1[i] -= lr * lgrad->db1[i];

    for (uint32_t i = 0; i < (uint32_t)2 * D * H; i++)
      layer->W2[i] -= lr * lgrad->dW2[i];
    for (uint32_t i = 0; i < (uint32_t)2 * D * C; i++)
      layer->W2c[i] -= lr * lgrad->dW2c[i];
    for (uint32_t i = 0; i < (uint32_t)2 * D; i++)
      layer->b2[i] -= lr * lgrad->db2[i];
  }
}

maf_adam_t *maf_create_adam(const maf_model_t *model, float lr, float beta1,
                            float beta2, float epsilon) {
  if (model == NULL)
    return NULL;

  maf_adam_t *adam = (maf_adam_t *)malloc(sizeof(maf_adam_t));
  if (!adam)
    return NULL;

  adam->n_flows = model->n_flows;
  adam->t = 0;
  adam->lr = lr;
  adam->beta1 = beta1;
  adam->beta2 = beta2;
  adam->epsilon = epsilon;

  adam->layers =
      (maf_layer_adam_t *)calloc(model->n_flows, sizeof(maf_layer_adam_t));
  if (!adam->layers) {
    free(adam);
    return NULL;
  }

  for (uint16_t k = 0; k < model->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];
    maf_layer_adam_t *ladam = &adam->layers[k];
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    /* Allocate first moments (m) */
    ladam->mW1y = (float *)calloc(H * D, sizeof(float));
    ladam->mW1c = (float *)calloc(H * C, sizeof(float));
    ladam->mb1 = (float *)calloc(H, sizeof(float));
    ladam->mW2 = (float *)calloc(2 * D * H, sizeof(float));
    ladam->mW2c = (float *)calloc(2 * D * C, sizeof(float));
    ladam->mb2 = (float *)calloc(2 * D, sizeof(float));

    /* Allocate second moments (v) */
    ladam->vW1y = (float *)calloc(H * D, sizeof(float));
    ladam->vW1c = (float *)calloc(H * C, sizeof(float));
    ladam->vb1 = (float *)calloc(H, sizeof(float));
    ladam->vW2 = (float *)calloc(2 * D * H, sizeof(float));
    ladam->vW2c = (float *)calloc(2 * D * C, sizeof(float));
    ladam->vb2 = (float *)calloc(2 * D, sizeof(float));

    if (!ladam->mW1y || !ladam->mW1c || !ladam->mb1 || !ladam->mW2 ||
        !ladam->mW2c || !ladam->mb2 || !ladam->vW1y || !ladam->vW1c ||
        !ladam->vb1 || !ladam->vW2 || !ladam->vW2c || !ladam->vb2) {
      maf_free_adam(adam);
      return NULL;
    }
  }

  return adam;
}

void maf_free_adam(maf_adam_t *adam) {
  if (adam == NULL)
    return;

  if (adam->layers) {
    for (uint16_t k = 0; k < adam->n_flows; k++) {
      maf_layer_adam_t *ladam = &adam->layers[k];
      free(ladam->mW1y);
      free(ladam->mW1c);
      free(ladam->mb1);
      free(ladam->mW2);
      free(ladam->mW2c);
      free(ladam->mb2);
      free(ladam->vW1y);
      free(ladam->vW1c);
      free(ladam->vb1);
      free(ladam->vW2);
      free(ladam->vW2c);
      free(ladam->vb2);
    }
    free(adam->layers);
  }
  free(adam);
}

static void adam_update_param(float *param, const float *grad, float *m,
                              float *v, uint32_t size, float lr, float beta1,
                              float beta2, float epsilon, float beta1_t,
                              float beta2_t) {
  for (uint32_t i = 0; i < size; i++) {
    float g = grad[i];

    /* Update moments */
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    /* Bias correction */
    float m_hat = m[i] / (1.0f - beta1_t);
    float v_hat = v[i] / (1.0f - beta2_t);

    /* Update parameter */
    param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
  }
}

void maf_adam_step(maf_model_t *model, maf_adam_t *adam,
                   const maf_grad_t *grad) {
  if (model == NULL || adam == NULL || grad == NULL)
    return;

  adam->t++;
  float beta1_t = powf(adam->beta1, adam->t);
  float beta2_t = powf(adam->beta2, adam->t);

  for (uint16_t k = 0; k < model->n_flows; k++) {
    maf_layer_t *layer = &model->layers[k];
    maf_layer_adam_t *ladam = &adam->layers[k];
    const maf_layer_grad_t *lgrad = &grad->layers[k];

    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    adam_update_param(layer->W1y, lgrad->dW1y, ladam->mW1y, ladam->vW1y, H * D,
                      adam->lr, adam->beta1, adam->beta2, adam->epsilon,
                      beta1_t, beta2_t);
    adam_update_param(layer->W1c, lgrad->dW1c, ladam->mW1c, ladam->vW1c, H * C,
                      adam->lr, adam->beta1, adam->beta2, adam->epsilon,
                      beta1_t, beta2_t);
    adam_update_param(layer->b1, lgrad->db1, ladam->mb1, ladam->vb1, H,
                      adam->lr, adam->beta1, adam->beta2, adam->epsilon,
                      beta1_t, beta2_t);

    adam_update_param(layer->W2, lgrad->dW2, ladam->mW2, ladam->vW2, 2 * D * H,
                      adam->lr, adam->beta1, adam->beta2, adam->epsilon,
                      beta1_t, beta2_t);
    adam_update_param(layer->W2c, lgrad->dW2c, ladam->mW2c, ladam->vW2c,
                      2 * D * C, adam->lr, adam->beta1, adam->beta2,
                      adam->epsilon, beta1_t, beta2_t);
    adam_update_param(layer->b2, lgrad->db2, ladam->mb2, ladam->vb2, 2 * D,
                      adam->lr, adam->beta1, adam->beta2, adam->epsilon,
                      beta1_t, beta2_t);
  }
}