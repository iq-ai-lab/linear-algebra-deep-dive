# Ch3-03. Rayleigh 몫: 고윳값을 최적화로 보기

> "대칭 행렬의 고윳값은 단위구 위에서의 **Rayleigh 몫의 임계값**이다."

## 📌 학습 목표

- Rayleigh 몫 $R(\mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$의 성질을 유도한다.
- **임계점의 방정식**이 고유값 방정식 $A\mathbf{x} = \lambda \mathbf{x}$와 동일함을 증명한다.
- **Courant-Fischer min-max 정리**로 모든 고윳값을 최적화로 표현한다.
- **Weyl's interaction inequalities** (섭동 이론).

---

## 🎯 핵심 질문

> **질문 1**: 왜 Rayleigh 몫의 gradient를 0으로 놓으면 고유값 방정식이 되는가?
> **질문 2**: 중간 고윳값 $\lambda_k$는 어떻게 최적화로 표현하는가?
> **질문 3**: 섭동된 행렬의 고윳값 변화 한계는?

---

## 1. Rayleigh 몫의 정의와 기본 성질

### 정의 1.1

대칭 $A \in \mathbb{R}^{n \times n}$에 대해:

$$
R_A(\mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}, \quad \mathbf{x} \neq \mathbf{0}
$$

### 성질

- **Scale 불변**: $R_A(c\mathbf{x}) = R_A(\mathbf{x})$ ($c \neq 0$)
- **연속 미분 가능**: $\mathbf{x} \neq \mathbf{0}$에서
- **범위**: $R_A(\mathbf{x}) \in [\lambda_n, \lambda_1]$ (스펙트럼 정리로부터)

### 정리 1.2 (Gradient)

$$
\nabla R_A(\mathbf{x}) = \frac{2}{\mathbf{x}^T\mathbf{x}}(A\mathbf{x} - R_A(\mathbf{x}) \mathbf{x})
$$

### 증명

$$
\nabla_\mathbf{x}(\mathbf{x}^T A \mathbf{x}) = 2 A \mathbf{x}, \qquad \nabla_\mathbf{x}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}
$$

몫의 미분:

$$
\nabla R_A = \frac{2A\mathbf{x} \cdot \mathbf{x}^T\mathbf{x} - \mathbf{x}^T A\mathbf{x} \cdot 2\mathbf{x}}{(\mathbf{x}^T \mathbf{x})^2} = \frac{2(A\mathbf{x} - R_A(\mathbf{x})\mathbf{x})}{\mathbf{x}^T\mathbf{x}} \quad \blacksquare
$$

### 따름정리 1.3 (임계점 ⟺ 고유값)

$\nabla R_A(\mathbf{x}) = \mathbf{0} \iff A \mathbf{x} = R_A(\mathbf{x}) \mathbf{x}$.

**즉, 임계점은 정확히 고유벡터**이고, 그 때의 Rayleigh 값은 고윳값.

---

## 2. Rayleigh 원리 (극값 정리)

### 정리 2.1

대칭 $A$의 고윳값을 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$이라 하면:

$$
\lambda_1 = \max_{\mathbf{x} \neq \mathbf{0}} R_A(\mathbf{x}), \qquad \lambda_n = \min_{\mathbf{x} \neq \mathbf{0}} R_A(\mathbf{x})
$$

그리고 최대/최소는 각각 $\mathbf{x} = \mathbf{q}_1, \mathbf{q}_n$ (고유벡터)에서 달성.

### 증명

$A = Q\Lambda Q^T$, $\mathbf{y} = Q^T \mathbf{x}$. $\mathbf{x}^T \mathbf{x} = \mathbf{y}^T \mathbf{y}$:

$$
R_A(\mathbf{x}) = \frac{\sum \lambda_i y_i^2}{\sum y_i^2}
$$

$\lambda_i \leq \lambda_1$이므로 $\sum \lambda_i y_i^2 \leq \lambda_1 \sum y_i^2$, 따라서 $R_A \leq \lambda_1$. $\mathbf{y} = \mathbf{e}_1$ ($\mathbf{x} = \mathbf{q}_1$)에서 등호. 최소는 대칭적으로. $\blacksquare$

### 정리 2.2 (중간 고윳값, deflation)

$\mathbf{q}_1, \ldots, \mathbf{q}_{k-1}$에 직교하는 $\mathbf{x}$에 대해 $R_A$를 최대화하면 $\lambda_k$ 얻음:

$$
\lambda_k = \max_{\substack{\mathbf{x} \neq \mathbf{0} \\ \mathbf{x} \perp \mathbf{q}_1, \ldots, \mathbf{q}_{k-1}}} R_A(\mathbf{x})
$$

### 증명

$\mathbf{x} \perp \mathbf{q}_i$ ($i < k$)이면 $\mathbf{y} = Q^T\mathbf{x}$의 첫 $k-1$ 성분이 0. 따라서:

$$
R_A(\mathbf{x}) = \frac{\sum_{i \geq k} \lambda_i y_i^2}{\sum_{i \geq k} y_i^2} \leq \lambda_k
$$

$\mathbf{x} = \mathbf{q}_k$에서 등호. $\blacksquare$

**실용적 방법**: Power method로 $\lambda_1, \mathbf{q}_1$ 찾고, deflate $A' = A - \lambda_1 \mathbf{q}_1 \mathbf{q}_1^T$, 다시 power method.

---

## 3. Courant-Fischer Min-Max

### 정리 3.1 (Courant-Fischer)

$$
\lambda_k = \min_{\substack{S \subset \mathbb{R}^n \\ \dim S = n - k + 1}} \max_{\mathbf{x} \in S \setminus \{0\}} R_A(\mathbf{x}) = \max_{\substack{T \subset \mathbb{R}^n \\ \dim T = k}} \min_{\mathbf{x} \in T \setminus \{0\}} R_A(\mathbf{x})
$$

### 증명 (max-min 버전)

**(≥)** $T^* = \operatorname{span}(\mathbf{q}_1, \ldots, \mathbf{q}_k)$, $\dim T^* = k$. $\mathbf{x} \in T^*$면 $\mathbf{y} = Q^T \mathbf{x}$의 처음 $k$ 성분만 비영:

$$
R_A(\mathbf{x}) = \frac{\sum_{i=1}^k \lambda_i y_i^2}{\sum_{i=1}^k y_i^2} \geq \lambda_k
$$

**(≤)** 임의 $k$차원 $T$와 $n-k+1$차원 $S^* = \operatorname{span}(\mathbf{q}_k, \ldots, \mathbf{q}_n)$. $\dim T + \dim S^* = k + (n-k+1) = n+1 > n$, 따라서 $T \cap S^* \neq \{\mathbf{0}\}$. $\mathbf{x} \in T \cap S^*$이면 $R_A(\mathbf{x}) \leq \lambda_k$. 따라서:

$$
\min_{T} R_A(\mathbf{x}) \leq R_A(\mathbf{x}^*) \leq \lambda_k
$$

두 방향 결합. $\blacksquare$

### 의미

**고윳값은 부분공간 최적화**. 좌표계와 무관한 본질적 정의.

---

## 4. Weyl의 섭동 정리

### 정리 4.1 (Weyl's Inequalities)

대칭 $A, E \in \mathbb{R}^{n \times n}$, $\tilde{A} = A + E$. 고윳값을 $\lambda_1 \geq \cdots \geq \lambda_n$, $\tilde{\lambda}_1 \geq \cdots \geq \tilde{\lambda}_n$이라 하면:

$$
|\tilde{\lambda}_k - \lambda_k| \leq \|E\|_2 \quad \forall k
$$

### 증명

Courant-Fischer:

$$
\tilde{\lambda}_k = \max_{\dim T = k} \min_{\mathbf{x} \in T} R_{\tilde{A}}(\mathbf{x}) = \max_{\dim T = k} \min_{\mathbf{x} \in T} \frac{\mathbf{x}^T (A + E) \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$

$\frac{\mathbf{x}^T E \mathbf{x}}{\mathbf{x}^T \mathbf{x}} \in [-\|E\|_2, \|E\|_2]$이므로:

$$
R_A(\mathbf{x}) - \|E\|_2 \leq R_{\tilde{A}}(\mathbf{x}) \leq R_A(\mathbf{x}) + \|E\|_2
$$

양변 max-min 적용:

$$
\lambda_k - \|E\|_2 \leq \tilde{\lambda}_k \leq \lambda_k + \|E\|_2 \quad \blacksquare
$$

### 의미

**대칭 행렬의 고윳값은 섭동에 Lipschitz-1 연속**. (비대칭은 훨씬 나쁘게, Ch3-06에서.)

---

## 5. 일반화된 Rayleigh 몫

### 정의 5.1

$A, B$ 대칭, $B \succ 0$:

$$
R_{A, B}(\mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T B \mathbf{x}}
$$

### 정리 5.2

$R_{A, B}$의 임계점은 일반화 고유값 문제 $A\mathbf{x} = \lambda B \mathbf{x}$의 해.

### 증명

$\nabla R_{A,B} = \frac{2(A\mathbf{x} - R_{A,B}(\mathbf{x}) B \mathbf{x})}{\mathbf{x}^T B \mathbf{x}} = 0 \iff A\mathbf{x} = \lambda B\mathbf{x}$. $\blacksquare$

### 응용

- **LDA (Fisher discriminant)**: $\max \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$ (클래스간/내 분산 비).
- **CCA, GEV in 신호처리**: 일반화 고유값.
- **진동 문제**: $K\mathbf{x} = \omega^2 M \mathbf{x}$ (강성, 질량).

---

## 6. 응용: Iterative 고유값 개선 (Rayleigh Quotient Iteration)

### 6.1 알고리즘

1. 초기 벡터 $\mathbf{x}_0$, 노멀라이즈
2. $\mu_k = R_A(\mathbf{x}_k)$
3. $(A - \mu_k I) \mathbf{y}_{k+1} = \mathbf{x}_k$로 선형계 풀기
4. $\mathbf{x}_{k+1} = \mathbf{y}_{k+1}/\|\mathbf{y}_{k+1}\|$

### 정리 6.1 (수렴 속도)

대칭 $A$에 대해 RQI는 근방에서 **3차 수렴**. $\|\mathbf{x}_k - \mathbf{q}\| \sim \epsilon$이면 $\|\mathbf{x}_{k+1} - \mathbf{q}\| \sim \epsilon^3$.

(증명: Taylor 전개, Ostrowski 1959. Rayleigh 근사 오차가 이미 $O(\epsilon^2)$, shift-invert가 오차를 또 한 번 제곱.)

### 단점

각 단계마다 $(A - \mu_k I)$를 분해해야 함 → 단계당 $O(n^3)$. 실무에서는 Inverse Iteration with fixed shift → QR Algorithm (Ch3-05).

---

## 7. Python 실험

### 7.1 Rayleigh 몫 범위 확인

```python
import numpy as np

np.random.seed(42)
n = 5
M = np.random.randn(n, n)
A = (M + M.T) / 2

eigvals, Q = np.linalg.eigh(A)
lam_min, lam_max = eigvals[0], eigvals[-1]

vals = []
for _ in range(10000):
    x = np.random.randn(n)
    vals.append(x @ A @ x / (x @ x))
print(f"λ_min={lam_min:.3f}, sample min R={min(vals):.3f}")
print(f"λ_max={lam_max:.3f}, sample max R={max(vals):.3f}")
```

### 7.2 Weyl 부등식 검증

```python
E = 0.01 * np.random.randn(n, n)
E = (E + E.T) / 2
A_pert = A + E
eigvals_pert, _ = np.linalg.eigh(A_pert)

diffs = np.abs(eigvals_pert - eigvals)
print("max |λ_pert - λ|:", diffs.max())
print("||E||_2:", np.linalg.norm(E, 2))
# max diff ≤ ||E||_2
```

### 7.3 Rayleigh Quotient Iteration

```python
def rqi(A, x0, max_iter=20, tol=1e-12):
    x = x0 / np.linalg.norm(x0)
    n = A.shape[0]
    history = []
    for k in range(max_iter):
        mu = x @ A @ x
        history.append(mu)
        try:
            y = np.linalg.solve(A - mu * np.eye(n), x)
        except np.linalg.LinAlgError:
            break  # singular = converged
        x_new = y / np.linalg.norm(y)
        if np.linalg.norm(x_new - x) < tol and np.linalg.norm(x_new + x) < tol:
            break
        x = x_new
    return mu, x, history

x0 = np.random.randn(n)
mu, v, hist = rqi(A, x0)
print("Converged eigval:", mu)
print("Known eigvals:", eigvals)
print("Convergence (log-scale):")
for k, m in enumerate(hist):
    err = min(abs(m - e) for e in eigvals)
    print(f"  iter {k}: μ={m:.10f}, err={err:.2e}")
```

### 7.4 일반화 고유값 (LDA 스타일)

```python
# 두 클래스
n_samples = 100
class0 = np.random.randn(n_samples, 2) + np.array([2, 0])
class1 = np.random.randn(n_samples, 2) + np.array([-2, 0])

# Between-class scatter
mu_all = np.vstack([class0, class1]).mean(axis=0)
mu0 = class0.mean(axis=0); mu1 = class1.mean(axis=0)
S_B = n_samples * (np.outer(mu0 - mu_all, mu0 - mu_all) +
                   np.outer(mu1 - mu_all, mu1 - mu_all))
# Within-class scatter
S_W = np.cov(class0.T, bias=False) + np.cov(class1.T, bias=False)

# Generalized eig: S_B w = λ S_W w
from scipy.linalg import eigh
eigvals, W = eigh(S_B, S_W)
print("Fisher directions:", W[:, -1])  # dominant direction
```

---

## 8. 요약

| 결과                  | 공식                                                     |
| --------------------- | -------------------------------------------------------- |
| Rayleigh 몫           | $R_A(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}/\mathbf{x}^T\mathbf{x}$ |
| Gradient              | $\frac{2}{\|\mathbf{x}\|^2}(A\mathbf{x} - R_A(\mathbf{x})\mathbf{x})$ |
| 임계점                | 고유벡터                                                 |
| 최대/최소             | $\lambda_1, \lambda_n$                                   |
| Courant-Fischer       | $\lambda_k = \max_{T^k} \min_{\mathbf{x}\in T} R_A$      |
| Weyl                  | $\|\tilde\lambda_k - \lambda_k\| \leq \|E\|_2$           |
| Generalized RQ        | $A\mathbf{x} = \lambda B\mathbf{x}$                      |
| RQI 수렴              | 3차 (대칭)                                               |

---

## 9. 참고 문헌

- Horn & Johnson, *Matrix Analysis*, §4.2.
- Parlett, B. N. (1998). *The Symmetric Eigenvalue Problem*. SIAM.
- Stewart, G. W. (1990). *Matrix Perturbation Theory*.

---

## 10. 다음 문서

- **[04. Perron-Frobenius](./04-perron-frobenius.md)**: 비음 행렬의 spectral radius와 dominant eigenvector (PageRank의 이론적 기초).

---

## 11. 내비게이션

[◀ 02. 고윳값의 기하적 의미](./02-eigenvalue-geometry.md) | [📚 README](../README.md) | [04. Perron-Frobenius ▶](./04-perron-frobenius.md)
