# 7.3 BatchNorm의 선형대수

> "BatchNorm은 활성값을 평균 0, 분산 1의 부분공간으로 '정사영'한다."

---

## 1. 학습 목표

- **Batch Normalization**의 수식을 엄밀히 유도한다.
- **학습 중**과 **추론 중**의 동작 차이, running statistics의 역할을 이해한다.
- BatchNorm의 **선형대수적 해석** — 특정 부분공간으로의 정사영 + 학습 가능한 affine.
- BatchNorm의 **기울기 흐름**을 유도하고, 왜 훈련을 안정시키는지 설명한다.
- **LayerNorm, GroupNorm, InstanceNorm** 과의 축 차이 비교.

---

## 2. BatchNorm의 정의

### 2.1 문제 설정

입력 $X \in \mathbb{R}^{B \times D}$ (배치 크기 $B$, 특징 차원 $D$). 각 특징 $j$에 대해 배치 축 통계:

$$
\mu_j = \frac{1}{B}\sum_{i=1}^B X_{ij}, \quad \sigma_j^2 = \frac{1}{B}\sum_{i=1}^B (X_{ij} - \mu_j)^2
$$

정규화:

$$
\hat X_{ij} = \frac{X_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

Affine 변환:

$$
Y_{ij} = \gamma_j \hat X_{ij} + \beta_j
$$

$\gamma, \beta \in \mathbb{R}^D$는 학습 가능한 파라미터.

### 2.2 Conv에서의 BatchNorm

$X \in \mathbb{R}^{B \times C \times H \times W}$에 대해 **채널별**로 ($(B, H, W)$ 전체를 모아) 통계 계산.

### 2.3 학습 vs 추론

- **학습**: 각 미니배치의 $\mu, \sigma$ 사용, 동시에 지수이동평균(EMA)으로 **running stats** 유지:
$$
\mu_{\text{run}} \leftarrow \alpha \mu_{\text{run}} + (1 - \alpha) \mu_{\text{batch}}
$$
- **추론**: running stats로 대체 (배치 크기에 무관해짐).

---

## 3. 선형대수적 해석

### 3.1 정규화 = 정사영

$B$-차원 벡터 $x \in \mathbb{R}^B$ (하나의 특징에 대한 배치 값들)를 생각하자. 정규화는

$$
\hat x = \frac{x - \bar x \mathbf{1}}{\|x - \bar x \mathbf{1}\| / \sqrt{B}}
$$

(여기서 $\sigma = \|x - \bar x \mathbf{1}\| / \sqrt{B}$).

이는 **두 단계**:

1. $\mathbf{1}$-방향 성분 제거: $x \mapsto x - \bar x \mathbf{1}$. 이는 $\mathbf{1}^\perp$로의 정사영.
2. 정규화: 크기를 $\sqrt{B}$로 표준화.

### 3.2 정사영 행렬

$P_{\mathbf{1}^\perp} = I - \frac{1}{B}\mathbf{1}\mathbf{1}^T$. 그러면 $x - \bar x \mathbf{1} = P_{\mathbf{1}^\perp} x$.

### 3.3 $\hat x$의 살아있는 부분공간

$\hat x$는 항상 $\mathbf{1}^\perp$에 놓이며, 노름이 $\sqrt{B}$이다. 즉 $\hat x \in S^{B-2}(\sqrt B)$ (반경 $\sqrt B$의 $(B-1)$-차원 구).

### 3.4 $\gamma, \beta$의 역할

- **정규화**는 $x$에서 "배치 전체의 평균과 스케일"이라는 **2개 자유도**를 뺐다.
- **$\gamma, \beta$** 가 이를 **복원**한다 (특징별 독립).

BatchNorm은 결국 "불필요한 공변이(covariate) 2개를 학습 가능하게 다시 포함"시키는 **재파라미터화**.

---

## 4. Internal Covariate Shift와 BatchNorm의 효과

### 4.1 원래 동기 (Ioffe & Szegedy, 2015)

깊은 네트워크에서 각 층의 입력 분포가 훈련 중 변하는 "내부 공변이 이동". BatchNorm이 이를 줄여 훈련을 안정.

### 4.2 후속 연구: Santurkar et al. (2018)

실험적으로 internal covariate shift는 주 원인이 아님. 오히려 **손실 풍경을 매끈하게** 만들어 그래디언트가 **더 예측 가능**하게.

### 4.3 이론적 설명

Lipschitz 상수 개선:

$$
L_{\text{BN}} \le L_{\text{unnorm}}
$$

즉 BN 이후의 함수는 Lipschitz 상수가 더 작다. SGD가 더 큰 스텝을 사용 가능.

---

## 5. BatchNorm의 역전파

### 5.1 상류 그래디언트

$\partial L / \partial Y = G \in \mathbb{R}^{B \times D}$.

### 5.2 $\gamma, \beta$ 그래디언트

$$
\frac{\partial L}{\partial \gamma_j} = \sum_i G_{ij} \hat X_{ij}, \quad \frac{\partial L}{\partial \beta_j} = \sum_i G_{ij}
$$

### 5.3 $X$ 그래디언트

$X_{ij}$가 $\mu_j, \sigma_j, \hat X_{ij}$ 모두에 영향을 주기 때문에 복잡:

$$
\frac{\partial L}{\partial X_{ij}} = \frac{\gamma_j}{\sigma_j}\left[G_{ij} - \frac{1}{B}\sum_k G_{kj} - \hat X_{ij} \cdot \frac{1}{B}\sum_k G_{kj} \hat X_{kj}\right]
$$

**해석**: 배치 평균과 배치-활성값 상관관계를 **빼내는** 두 개의 항. 결과적으로 BN의 backward는 "배치 축에서 $\mathbf{1}$과 $\hat X$ 방향을 제거"하는 **정사영**.

### 5.4 Gram 행렬 관점

$B \times B$ 행렬 $M = \mathbf{1}\mathbf{1}^T / B$는 $\mathbf{1}$ 방향 정사영자. $N = \hat X_{\cdot j} \hat X_{\cdot j}^T / B$는 $\hat X$ 방향 정사영자. BN backward는

$$
\bar X = \frac{\gamma}{\sigma}(I - M - N_{\hat X}) G
$$

의 구조.

---

## 6. 다른 정규화 기법들

### 6.1 LayerNorm

배치가 아니라 **특징 축**에서 통계:

$$
\mu_i = \frac{1}{D}\sum_j X_{ij}, \quad \sigma_i^2 = \frac{1}{D}\sum_j (X_{ij} - \mu_i)^2
$$

각 샘플 독립. 배치 크기에 무관.

**Transformer의 기본**: batch 크기가 가변적이고 seq 간 배치가 이상적이지 않음.

### 6.2 InstanceNorm

각 샘플, 각 채널별로 통계 (H, W만 모음). 스타일 전이에 유용.

### 6.3 GroupNorm

채널을 $G$개 그룹으로 나눠 그룹별 정규화. 배치 크기 1에서도 안정.

### 6.4 축의 대조

| 정규화 | 통계 축 |
|---|---|
| BatchNorm | $(B, H, W)$, 각 $C$ 채널별 |
| LayerNorm | $(C, H, W)$, 각 샘플별 |
| InstanceNorm | $(H, W)$, 각 $(B, C)$ 별 |
| GroupNorm | $(C_{\text{group}}, H, W)$, 각 (sample, group) 별 |

같은 텐서에 대해 **어느 축을 평균내는가**의 차이.

---

## 7. BatchNorm의 불연속성과 문제점

### 7.1 배치 크기 의존

배치 크기 $B$가 작으면 $\mu, \sigma$의 추정이 불안정. $B = 1$이면 **정규화 후 0**.

### 7.2 학습/추론 불일치

Running stats가 잘못 추정되면 추론 성능 악화. EMA 모멘텀, warm-up 조심.

### 7.3 분산 훈련

데이터 병렬 시 각 GPU가 서로 다른 배치 → $\mu, \sigma$가 다름. **SyncBatchNorm**은 모든 GPU 간 동기화.

---

## 8. BatchNorm과 최적화

### 8.1 학습률 불변성 부분

$W \to cW$로 스케일해도 BN 뒤의 출력은 같다 (정규화 때문). 따라서 BN이 있는 층은 **weight decay와의 상호작용**이 복잡.

### 8.2 Weight Decay의 효과

$\|W\|^2$를 감소시키면 $\nabla W$의 **유효 학습률이 증가**하는 역설적 효과.

### 8.3 Batch Size vs Learning Rate

큰 배치는 BN 통계가 정확해지고, 학습률을 비례적으로 키울 수 있음 (linear scaling rule).

---

## 9. 거대 모델에서의 대체

### 9.1 RMSNorm

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

LayerNorm에서 평균 빼기를 생략. 더 단순하고 일부 이득. LLaMA 등에서 사용.

### 9.2 Pre-LN vs Post-LN

- **Post-LN**: $\text{LN}(x + f(x))$ (원래 Transformer)
- **Pre-LN**: $x + f(\text{LN}(x))$ (안정적, GPT 등)

Pre-LN이 더 안정적이나 약간 성능이 낮을 수 있음.

---

## 10. Python 실험

### 10.1 BatchNorm Forward/Backward 직접 구현

```python
import numpy as np

def batchnorm_forward(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    X_hat = (X - mu) / np.sqrt(var + eps)
    Y = gamma * X_hat + beta
    cache = (X, X_hat, mu, var, gamma, eps)
    return Y, cache

def batchnorm_backward(dY, cache):
    X, X_hat, mu, var, gamma, eps = cache
    B, D = X.shape
    sigma = np.sqrt(var + eps)
    
    dgamma = (dY * X_hat).sum(axis=0)
    dbeta = dY.sum(axis=0)
    
    dX_hat = dY * gamma
    mean_dY = dY.mean(axis=0)
    mean_dY_Xhat = (dY * X_hat).mean(axis=0)
    
    dX = gamma / sigma * (dY - mean_dY - X_hat * mean_dY_Xhat)
    
    return dX, dgamma, dbeta

np.random.seed(0)
B, D = 32, 10
X = np.random.randn(B, D) * 2 + 5
gamma = np.ones(D)
beta = np.zeros(D)

Y, cache = batchnorm_forward(X, gamma, beta)
print(f"정규화 후 평균: {Y.mean(axis=0)[:3]}")
print(f"정규화 후 표준편차: {Y.std(axis=0)[:3]}")

# 그래디언트 검증
dY = np.random.randn(*Y.shape)
dX, dgamma, dbeta = batchnorm_backward(dY, cache)

# 수치 검증
eps_num = 1e-6
dX_num = np.zeros_like(X)
for i in range(B):
    for j in range(D):
        orig = X[i, j]
        X[i, j] = orig + eps_num
        Y_p, _ = batchnorm_forward(X, gamma, beta)
        X[i, j] = orig - eps_num
        Y_m, _ = batchnorm_forward(X, gamma, beta)
        X[i, j] = orig
        dX_num[i, j] = ((Y_p - Y_m) * dY).sum() / (2 * eps_num)

print(f"dX 오차: {np.max(np.abs(dX - dX_num)):.2e}")
```

### 10.2 BN이 활성값의 분포를 안정시키는지

```python
import numpy as np

np.random.seed(0)
# 깊은 선형 네트워크
L = 20
D = 50
B = 64

Ws = [np.random.randn(D, D) * 2 for _ in range(L)]

def simulate(use_bn):
    x = np.random.randn(B, D)
    means, stds = [], []
    for W in Ws:
        x = x @ W
        if use_bn:
            mu = x.mean(axis=0)
            sig = x.std(axis=0) + 1e-5
            x = (x - mu) / sig
        x = np.maximum(0, x)
        means.append(x.mean())
        stds.append(x.std())
    return means, stds

m1, s1 = simulate(use_bn=False)
m2, s2 = simulate(use_bn=True)

print("No BN:")
print(f"  평균 범위: [{min(m1):.2e}, {max(m1):.2e}]")
print(f"  표준편차 범위: [{min(s1):.2e}, {max(s1):.2e}]")
print("With BN:")
print(f"  평균 범위: [{min(m2):.4f}, {max(m2):.4f}]")
print(f"  표준편차 범위: [{min(s2):.4f}, {max(s2):.4f}]")
```

### 10.3 LayerNorm vs BatchNorm

```python
import numpy as np

def layernorm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

def batchnorm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

np.random.seed(0)
B, D = 32, 128
X = np.random.randn(B, D) * np.arange(1, D+1)  # 차원별로 분산 다름
gamma = np.ones(D)
beta = np.zeros(D)

Y_bn = batchnorm(X, gamma, beta)
Y_ln = layernorm(X, gamma, beta)

print(f"BatchNorm: 차원별 평균 max={abs(Y_bn.mean(axis=0)).max():.2e}")
print(f"LayerNorm: 샘플별 평균 max={abs(Y_ln.mean(axis=-1)).max():.2e}")
print()
print("둘의 효과 차이 - BN은 차원별 정규화, LN은 샘플별 정규화")
```

### 10.4 SyncBN 시뮬레이션

```python
import numpy as np

def sync_batchnorm(X_per_gpu, gamma, beta, eps=1e-5):
    """여러 GPU의 배치를 합쳐 통계 계산 후 각자 정규화"""
    X_all = np.concatenate(X_per_gpu, axis=0)
    mu = X_all.mean(axis=0)
    var = X_all.var(axis=0)
    return [gamma * (x - mu) / np.sqrt(var + eps) + beta for x in X_per_gpu]

np.random.seed(0)
D = 10
gamma = np.ones(D)
beta = np.zeros(D)

# 4 GPU, 각 배치 크기 4
batches = [np.random.randn(4, D) for _ in range(4)]

# SyncBN
Y_sync = sync_batchnorm(batches, gamma, beta)

# 비교: GPU별 독립 BN
Y_local = [(b - b.mean(axis=0)) / (b.std(axis=0) + 1e-5) for b in batches]

# sync된 쪽의 배치 전체 평균은 0에 가깝다
Y_sync_all = np.concatenate(Y_sync, axis=0)
print(f"SyncBN 전체 평균: {abs(Y_sync_all.mean(axis=0)).max():.2e}")

# 각 GPU가 독립 BN이면 각 배치는 평균 0이지만 전체는 그렇지 않을 수 있음
Y_local_all = np.concatenate(Y_local, axis=0)
print(f"LocalBN 전체 평균: {abs(Y_local_all.mean(axis=0)).max():.2e}")
```

### 10.5 배치 크기 민감도

```python
import numpy as np

def batch_norm_estimate(B, D=50, n_trials=100):
    errs = []
    for _ in range(n_trials):
        X = np.random.randn(B, D)
        mu = X.mean(axis=0)
        var = X.var(axis=0)
        # 참값: 평균 0, 분산 1
        errs.append(np.linalg.norm(mu) + np.linalg.norm(var - 1))
    return np.mean(errs), np.std(errs)

for B in [2, 8, 32, 128]:
    mean, std = batch_norm_estimate(B)
    print(f"B={B:3d}: 통계 추정 오차 = {mean:.3f} ± {std:.3f}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 공식

| 요소 | 공식 |
|---|---|
| Forward | $\hat X = (X - \mu) / \sigma$, $Y = \gamma \hat X + \beta$ |
| 정사영 해석 | $\hat x \in \mathbf{1}^\perp$, $\|\hat x\| = \sqrt B$ |
| Backward | $\bar X = \frac{\gamma}{\sigma}(I - M_{\mathbf 1} - M_{\hat X}) \bar Y / B$ 구조 |
| 추론 시 | running stats로 $\mu, \sigma$ 대체 |

### 한 줄 요약

> **BatchNorm은 배치 평균과 분산이라는 2개 자유도를 빼고 $\gamma, \beta$로 다시 복원하는 재파라미터화이다.**

### 다음 절 예고

다음 절에서는 **Spectral Normalization**을 다룬다. 최대 특잇값을 1로 강제하여 Lipschitz 제약을 부여하는 기법으로, GAN과 기타 안정성 문제의 해답이다.

---

[◀ 02. 역전파](./02-backpropagation.md) | [📚 README](../README.md) | [04. Spectral Normalization ▶](./04-spectral-normalization.md)
