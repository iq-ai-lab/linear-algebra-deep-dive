# 6.4 텐서 분해: CP, Tucker, Tensor Train

> "SVD를 고차원으로 일반화하려는 시도가 텐서 분해의 전부다."

---

## 1. 학습 목표

- **CP 분해(CANDECOMP/PARAFAC)** 의 정의, 유일성, 계산 알고리즘을 익힌다.
- **Tucker 분해**와 고차 SVD(HOSVD)의 구성과 성질을 이해한다.
- **Tensor Train (TT) 분해**의 구조와 압축 효율을 본다.
- 텐서 랭크가 **NP-hard**임을 알고, 근사 알고리즘의 필요성을 인지한다.
- 추천 시스템, 신경망 압축, 신호 처리 응용을 전개한다.

---

## 2. 동기: 왜 텐서 분해인가?

### 2.1 행렬 SVD의 성과

$A \in \mathbb{R}^{m \times n}$는 SVD $A = U \Sigma V^T$로 완전히 이해된다:

- 랭크: 비영 특잇값 수
- 최적 저랭크 근사: Eckart-Young
- 주성분: $U, V$
- 수치 알고리즘: $O(mn \min(m, n))$

이 모든 걸 고차 텐서에서도 하고 싶다.

### 2.2 고차 텐서의 근본적 어려움

SVD의 단순성은 2차에 특유한 것이다. 3차 이상에서는

1. **텐서 랭크 결정**이 NP-hard (Håstad 1990)
2. **최적 저랭크 근사**가 존재하지 않을 수 있음 (border rank 문제)
3. **유일성 조건**이 모델마다 다름
4. **여러 비동치 랭크 개념**이 존재 (CP-rank, multilinear rank, tensor-train rank)

그럼에도 근사 분해는 실용적으로 매우 유용하다.

---

## 3. CP 분해 (CANDECOMP/PARAFAC)

### 3.1 정의

$T \in \mathbb{R}^{I_1 \times I_2 \times I_3}$을 rank-1 텐서들의 합으로 분해:

$$
T \approx \sum_{r=1}^R \lambda_r\, a_r \otimes b_r \otimes c_r
$$

또는 성분으로

$$
T_{ijk} \approx \sum_{r=1}^R \lambda_r\, a_{ir} b_{jr} c_{kr}
$$

### 3.2 행렬 형태

$A = [a_1 | \cdots | a_R] \in \mathbb{R}^{I_1 \times R}$, $B, C$ 마찬가지, $\Lambda = \text{diag}(\lambda)$. 그러면

$$
T_{(1)} = A \Lambda (C \odot B)^T
$$

여기서 $\odot$는 **Khatri-Rao 곱**: 열 단위 Kronecker.

### 3.3 CP 랭크

$T$의 **CP 랭크**는 분해에 필요한 최소 $R$. 텐서 랭크라고도 부른다.

- $3 \times 3 \times 3$ 텐서의 최대 CP 랭크: $5$ (필드에 따라 다름).
- 2차 텐서(행렬)의 CP 랭크 = 행렬 랭크 (SVD로부터 $R = \text{rank}(A)$).

### 3.4 유일성: Kruskal 조건

**정리 6.4.1 (Kruskal).** $A, B, C$의 $k$-rank를 $k_A, k_B, k_C$라 하자 ($k$-rank: 선형독립 열을 고를 수 있는 최대 열 수 + 1). CP 분해가 **본질적으로 유일**(순열, 스케일 제외)하려면

$$
k_A + k_B + k_C \ge 2R + 2
$$

**증명 개요.** 원래 Kruskal (1977)의 증명은 치밀한 경우 분석이 필요. 현대 증명은 대수기하학 도구를 사용.

### 3.5 중요한 성질

- CP 분해는 **회전 불변성이 없다**: 즉 $A' = AR, B' = BR$ 등으로 바꿀 수 없다 (SVD의 $UV^T$와 다름).
- 이 경직성이 오히려 **실질적 해석 가능성**을 준다 (심리학, 화학 등에서 CP가 선호됨).

### 3.6 ALS 알고리즘 (Alternating Least Squares)

$\|T - \hat T\|_F^2$를 최소화. $A, B, C$를 한 번에 최적화하는 건 비볼록이지만, **하나 고정하고 하나씩 업데이트**하면 각 단계가 볼록이다.

$A$ 업데이트 (다른 건 고정):

$$
A \leftarrow T_{(1)} (C \odot B) \left((C^T C) \ast (B^T B)\right)^{-1}
$$

여기서 $\ast$는 Hadamard 곱. 이는 **정규방정식**:

$$
A = \arg\min_A \|T_{(1)} - A (C \odot B)^T\|_F^2
$$

ALS는 각 단계에서 목적함수가 감소하지만 **전역 최적을 보장하지 않는다** (local minima, swamps).

---

## 4. Tucker 분해와 HOSVD

### 4.1 Tucker 정의

$$
T \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3
$$

여기서

- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$: **코어 텐서**
- $U_n \in \mathbb{R}^{I_n \times R_n}$: **인자 행렬** (직교 정규화 가능)
- $\times_n$: 모드-$n$ 곱

성분:

$$
T_{ijk} \approx \sum_{p, q, r} \mathcal{G}_{pqr}\, U_{1,ip} U_{2,jq} U_{3,kr}
$$

### 4.2 모드-$n$ 랭크와 Multilinear Rank

- $\text{rank}_n(T) = \text{rank}(T_{(n)})$ (모드-$n$ unfolding의 행렬 랭크)
- **Multilinear rank**: $(R_1, R_2, R_3) = (\text{rank}_1, \text{rank}_2, \text{rank}_3)$

CP 랭크와 달리 **정확한 분해의 최소 차원**이 각 모드별로 정의된다.

### 4.3 HOSVD (Higher-Order SVD)

**정의 6.4.2 (HOSVD, De Lathauwer 등).**

1. 각 모드에 대해 unfold 후 SVD: $T_{(n)} = U_n \Sigma_n V_n^T$.
2. 코어 $\mathcal{G} = T \times_1 U_1^T \times_2 U_2^T \times_3 U_3^T$.
3. 그러면 $T = \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3$ (정확한 분해).

HOSVD의 $U_n$은 직교하며, 코어 $\mathcal{G}$는 "모두 직교(all-orthogonal)" 성질을 만족:

$$
\langle \mathcal{G}_{i=p}, \mathcal{G}_{i=q}\rangle = 0\quad \text{for } p \ne q
$$

(각 모드의 고정 슬라이스 간 내적이 0).

### 4.4 Truncated HOSVD

특잇값을 상위 $R_n$개만 남기면 근사가 된다:

$$
\hat T = \mathcal{G}_{\text{trunc}} \times_1 U_1^{(R_1)} \times_2 U_2^{(R_2)} \times_3 U_3^{(R_3)}
$$

**Truncated HOSVD는 최적 Tucker 근사가 아니다** (다만 준최적이고, 오차는 특잇값으로 bound됨):

$$
\|T - \hat T\|_F^2 \le \sum_{n=1}^N \sum_{r > R_n} \sigma_{n, r}^2
$$

### 4.5 HOOI (Higher-Order Orthogonal Iteration)

최적 Tucker를 위한 반복 알고리즘. ALS와 유사하게 각 $U_n$을 번갈아 업데이트.

---

## 5. Tensor Train (TT) 분해

### 5.1 정의

$N$차 텐서 $T \in \mathbb{R}^{I_1 \times \cdots \times I_N}$에 대해

$$
T_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots, \alpha_{N-1}} G_1(i_1, \alpha_1) G_2(\alpha_1, i_2, \alpha_2) \cdots G_N(\alpha_{N-1}, i_N)
$$

- $G_1 \in \mathbb{R}^{I_1 \times r_1}$
- $G_k \in \mathbb{R}^{r_{k-1} \times I_k \times r_k}$ ($2 \le k \le N-1$)
- $G_N \in \mathbb{R}^{r_{N-1} \times I_N}$

### 5.2 TT-rank

$(r_1, r_2, \ldots, r_{N-1})$을 **TT-rank**라 한다. 각 $r_k$는 텐서를 $(i_1, \ldots, i_k)$ vs $(i_{k+1}, \ldots, i_N)$으로 이분할한 **unfolding 행렬**의 랭크.

### 5.3 저장 효율

$I_n = I$, $r_k = r$ 가정:

- 전체 텐서: $I^N$
- TT: $N I r^2$ (선형!)

이로 인해 $N$이 큰 고차 텐서에 특히 유용 (차원의 저주 극복).

### 5.4 TT-SVD 알고리즘

순차적 SVD로 구성:

```
W = T.reshape(I_1, I_2 * ... * I_N)
G_1, s, V = SVD(W); G_1 = top r_1 cols
W = diag(s) V^T  (reshape: r_1, I_2, ..., I_N)
W = W.reshape(r_1 * I_2, I_3 * ... * I_N)
G_2, s, V = SVD(W); truncate to r_2
...
```

### 5.5 응용

- **양자 다체 시스템** (MPS)
- **고차원 적분** (밀도 추정)
- **NN 가중치 압축** (Novikov et al. 2015)

---

## 6. 다른 분해

### 6.1 Block Term Decomposition (BTD)

CP와 Tucker의 중간: rank-$(L, L, 1)$ 블록들의 합. 뇌영상과 블라인드 신호 분리에 유용.

### 6.2 t-SVD (Tensor SVD, Kilmer-Martin)

3차 텐서에 대해 circular convolution 곱을 정의하고, 그 곱에 대한 SVD. 이미지 처리에 강점.

### 6.3 Tensor Ring (TR)

TT에서 양 끝의 rank를 연결하여 고리 구조. 추가 유연성.

### 6.4 Hierarchical Tucker (HT)

Tucker의 재귀 버전. 트리 구조의 코어.

---

## 7. 응용: 추천 시스템

### 7.1 3차 텐서: (User, Item, Time)

사용자의 상품 선호를 시간별로 기록:

$$
T[u, i, t] = \text{rating of user } u \text{ on item } i \text{ at time } t
$$

결측치가 대부분.

### 7.2 CP 모델로 예측

$$
\hat T_{uit} = \sum_{r=1}^R a_{ur} b_{ir} c_{tr}
$$

각 인자 $a_r, b_r, c_r$가 **잠재 선호 차원**을 대표.

### 7.3 Alternating Least Squares with Missing Data

관측 마스크 $\Omega$에 대해

$$
\min_{A, B, C} \sum_{(u, i, t) \in \Omega} \left(T_{uit} - \sum_r a_{ur} b_{ir} c_{tr}\right)^2 + \lambda (\|A\|^2 + \|B\|^2 + \|C\|^2)
$$

---

## 8. 응용: 신경망 압축

### 8.1 FC 계층 압축

Fully-connected 계층 $y = Wx$, $W \in \mathbb{R}^{m \times n}$.

$m, n$을 작은 요인으로 소인수분해해서 $W$를 $N$-way 텐서로 reshape (e.g., $W \in \mathbb{R}^{m_1 m_2 m_3 \times n_1 n_2 n_3}$), 그 후 TT 분해.

파라미터 수 감소: $mn \to N \cdot (\max m_i \cdot \max n_i) \cdot r^2$

### 8.2 Convolution 압축

Tucker 분해로 컨볼루션 커널 $K \in \mathbb{R}^{C_o \times C_i \times d \times d}$를

$$
K = \mathcal{G} \times_1 U_1 \times_2 U_2
$$

로 분해. $C_o, C_i$ 방향만 압축하고 공간 방향은 건드리지 않음. 이는 **depthwise-separable**의 일반화.

---

## 9. 응용: 과학 계산

### 9.1 양자역학

파동함수 $\psi(x_1, \ldots, x_N)$은 $N$차 텐서. 차원의 저주: $d^N$. **DMRG** (Density Matrix Renormalization Group)는 TT 기반 근사 알고리즘.

### 9.2 편미분방정식의 고차원

$d$-차원 확산 방정식의 해를 TT 형태로 저장하여 $d = 100$까지도 풀 수 있음 (Khoromskij 등).

### 9.3 강화학습 Q-함수

$Q(s, a, ...)$를 텐서로 취급하여 저랭크 근사로 학습. 큰 상태공간에서의 샘플 효율 개선.

---

## 10. Python 실험

### 10.1 CP 분해 (ALS)

```python
import numpy as np

def khatri_rao(A, B):
    """열 단위 Kronecker"""
    I, R = A.shape
    J, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(I*J, R)

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

def cp_als(T, R, n_iter=100):
    I = T.shape
    factors = [np.random.randn(i, R) for i in I]

    for it in range(n_iter):
        for n in range(T.ndim):
            # Khatri-Rao 곱 (n 제외)
            kr = None
            for m in reversed(range(T.ndim)):
                if m == n: continue
                kr = factors[m] if kr is None else khatri_rao(kr, factors[m])
            # 각 인자의 Gram 행렬
            G = np.ones((R, R))
            for m in range(T.ndim):
                if m == n: continue
                G *= factors[m].T @ factors[m]
            # 업데이트
            factors[n] = unfold(T, n) @ kr @ np.linalg.pinv(G)

    return factors

np.random.seed(0)
I, R_true = (10, 8, 6), 3

# Ground truth rank-R 텐서
A_t = np.random.randn(10, R_true)
B_t = np.random.randn(8, R_true)
C_t = np.random.randn(6, R_true)
T_true = np.einsum("ir,jr,kr->ijk", A_t, B_t, C_t)

# 추정
factors = cp_als(T_true, R=R_true, n_iter=200)
T_reconstructed = np.einsum("ir,jr,kr->ijk", *factors)

err = np.linalg.norm(T_true - T_reconstructed) / np.linalg.norm(T_true)
print(f"CP 복원 상대 오차: {err:.2e}")
```

### 10.2 HOSVD (Truncated)

```python
import numpy as np

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

def mode_n_product(T, U, mode):
    T_u = unfold(T, mode)
    out = U @ T_u
    new_shape = list(T.shape)
    new_shape[mode] = U.shape[0]
    return np.moveaxis(out.reshape([new_shape[mode]] + [s for i, s in enumerate(new_shape) if i != mode]), 0, mode)

def hosvd(T, ranks):
    factors = []
    for n in range(T.ndim):
        U, _, _ = np.linalg.svd(unfold(T, n), full_matrices=False)
        factors.append(U[:, :ranks[n]])
    core = T
    for n in range(T.ndim):
        core = mode_n_product(core, factors[n].T, n)
    return core, factors

np.random.seed(0)
T = np.random.randn(10, 8, 6)
core, factors = hosvd(T, ranks=(4, 4, 3))

# 복원
T_approx = core
for n, U in enumerate(factors):
    T_approx = mode_n_product(T_approx, U, n)

err = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
print(f"HOSVD 상대 오차: {err:.4f}")
print(f"코어 shape: {core.shape}")
```

### 10.3 TT 분해

```python
import numpy as np

def tt_svd(T, ranks):
    dims = T.shape
    N = len(dims)
    cores = []
    W = T.reshape(dims[0], -1)
    for k in range(N-1):
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        r = min(ranks[k], len(s))
        U = U[:, :r]
        if k == 0:
            cores.append(U)
        else:
            cores.append(U.reshape(cores[-1].shape[-1] if k == 1 else ranks[k-1], dims[k], r))
        s = s[:r]
        Vt = Vt[:r]
        W = (np.diag(s) @ Vt).reshape(r * dims[k+1], -1)
    cores.append(W.reshape(r, dims[-1]))
    return cores

np.random.seed(0)
T = np.random.randn(4, 5, 6, 7)
cores = tt_svd(T, ranks=[4, 10, 4])

# 파라미터 수
total_params = sum(c.size for c in cores)
print(f"TT 파라미터: {total_params}, 원래 텐서: {T.size}")

# 복원 (간단)
T_rec = cores[0]  # (I_1, r_1)
for k in range(1, len(cores)-1):
    T_rec = np.einsum("...a,abc->...bc", T_rec, cores[k])
T_rec = np.einsum("...a,ab->...b", T_rec, cores[-1])

print(f"복원 오차: {np.linalg.norm(T - T_rec) / np.linalg.norm(T):.2e}")
```

### 10.4 추천 시스템 (모의)

```python
import numpy as np

np.random.seed(0)
n_users, n_items, n_time = 50, 30, 10
R = 3

# Ground truth 인자
A_t = np.random.randn(n_users, R)
B_t = np.random.randn(n_items, R)
C_t = np.random.randn(n_time, R)
T_full = np.einsum("ur,ir,tr->uit", A_t, B_t, C_t)

# 30% 관측
mask = np.random.rand(*T_full.shape) < 0.3
T_obs = T_full * mask

# 간단한 ALS with mask
def cp_als_masked(T_obs, mask, R, n_iter=100, lam=1e-3):
    dims = T_obs.shape
    factors = [np.random.randn(d, R) for d in dims]
    for _ in range(n_iter):
        for n in range(3):
            # Gradient descent 단계 (간단화)
            T_pred = np.einsum("ur,ir,tr->uit", *factors)
            diff = (T_pred - T_obs) * mask
            # factor_n 업데이트 (간단한 gradient)
            if n == 0:
                grad = np.einsum("uit,ir,tr->ur", diff, factors[1], factors[2])
            elif n == 1:
                grad = np.einsum("uit,ur,tr->ir", diff, factors[0], factors[2])
            else:
                grad = np.einsum("uit,ur,ir->tr", diff, factors[0], factors[1])
            factors[n] -= 0.01 * (grad + lam * factors[n])
    return factors

factors = cp_als_masked(T_obs, mask.astype(float), R=R, n_iter=500)
T_pred = np.einsum("ur,ir,tr->uit", *factors)

err_obs = np.linalg.norm((T_pred - T_full) * mask) / np.linalg.norm(T_full * mask)
err_hidden = np.linalg.norm((T_pred - T_full) * (1 - mask)) / np.linalg.norm(T_full * (1 - mask))

print(f"관측 오차: {err_obs:.3f}")
print(f"예측 오차: {err_hidden:.3f}")
```

### 10.5 Tensorly 라이브러리 활용

```python
# pip install tensorly 필요
# import tensorly as tl
# from tensorly.decomposition import parafac, tucker, tensor_train

# np.random.seed(0)
# T = tl.tensor(np.random.randn(10, 8, 6))
# 
# # CP
# cp_factors = parafac(T, rank=3)
# 
# # Tucker
# tucker_core, tucker_factors = tucker(T, rank=(4, 4, 3))
# 
# # TT
# tt_cores = tensor_train(T, rank=[1, 4, 4, 1])
```

---

## 11. 요약 및 다음 절 예고

### 분해 비교

| 분해 | 파라미터 수 | 장점 | 단점 |
|---|---|---|---|
| CP | $NR$개 벡터 | 해석 가능, 유일성 | 랭크 결정 NP-hard |
| Tucker | 코어 + $N$개 행렬 | 유연, HOSVD | 코어 지수적 |
| TT | $N$개 3-텐서 | 고차원 효율, 선형 저장 | 모드 순서 의존 |
| TR | TT + 고리 | 순서 독립 | 복잡성 증가 |

### 한 줄 요약

> **CP는 해석을 위해, Tucker는 유연성을 위해, TT는 고차원을 위해 존재한다.**

### 다음 절 예고

다음 절은 Chapter 6의 마지막이자 가장 실용적인 주제 — **신경망 가중치의 텐서 구조**. 딥러닝이 어떻게 텐서 대수 위에 서 있는지를 명시적으로 드러낸다.

---

[◀ 03. einsum](./03-einsum.md) | [📚 README](../README.md) | [05. NN 가중치 텐서 ▶](./05-nn-weight-tensor.md)
