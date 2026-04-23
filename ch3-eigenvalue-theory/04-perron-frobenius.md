# Ch3-04. Perron-Frobenius 정리: 비음 행렬의 지배 고유값

> "비음 행렬은 **양의 실수 고윳값 하나**가 모두를 지배하고, 그 고유벡터도 **양**이다."

## 📌 학습 목표

- **양행렬(positive)**과 **비음행렬(nonnegative)**을 구분하고 Perron, Perron-Frobenius 정리를 각각 증명한다.
- 기약(irreducible), 원시(primitive) 개념과 그래프 해석.
- Perron root와 dominant eigenvector의 기하적/확률적 의미.
- **PageRank**, 마르코프 체인 정상분포, Leslie 인구모델의 이론적 기초.

---

## 🎯 핵심 질문

> **질문 1**: 왜 양행렬에서 **spectral radius가 항상 고윳값**인가?
> **질문 2**: 기약 행렬에서 Perron eigenvector가 **유일**한 이유는?
> **질문 3**: 확률 행렬의 정상분포가 존재하는 **구조적 조건**은?

---

## 1. 기본 정의

### 정의 1.1

- $A \geq 0$ (**비음**): 모든 $A_{ij} \geq 0$
- $A > 0$ (**양**): 모든 $A_{ij} > 0$

벡터도 마찬가지: $\mathbf{x} \geq \mathbf{0}$, $\mathbf{x} > \mathbf{0}$.

### 정의 1.2 (Spectral Radius)

$$
\rho(A) = \max_i |\lambda_i(A)|
$$

### 정의 1.3 (기약성, Irreducibility)

$A \geq 0$이 **기약**이라 함은 대응하는 방향 그래프 $G_A$ (엣지 $i \to j \iff A_{ij} > 0$)가 **강연결(strongly connected)**이라는 뜻.

### 정의 1.4 (원시성, Primitivity)

$A \geq 0$이 **원시**라 함은 $A^k > 0$이 어떤 $k \geq 1$에 대해 성립한다는 뜻.

**동치**: 기약 + 자기 루프 있음 / 주기가 1인 그래프.

---

## 2. Perron 정리 (양행렬)

### 정리 2.1 (Perron, 1907)

$A > 0$이면:

1. $\rho(A)$는 $A$의 고윳값 (실수 양수)
2. 단순근 (기하적·대수적 중복도 모두 1)
3. 대응하는 고유벡터 $\mathbf{v}$는 $\mathbf{v} > 0$으로 잡을 수 있음
4. 모든 다른 고윳값 $\lambda$에 대해 $|\lambda| < \rho(A)$

### 증명 (핵심 아이디어)

**단계 1** ($\rho(A)$ 달성 방향 존재): Collatz-Wielandt 함수:

$$
r(\mathbf{x}) = \min_{i : x_i > 0} \frac{(A\mathbf{x})_i}{x_i}, \quad \mathbf{x} \geq \mathbf{0}, \mathbf{x} \neq \mathbf{0}
$$

단위 단체 $\Delta_n = \{\mathbf{x} \geq 0 : \sum x_i = 1\}$ 위에서 $r$을 최대화. $\Delta_n$은 컴팩트, $r$은 상반연속, max $r^*$ 달성. 이 $r^* > 0$은 양의 고윳값.

**단계 2** ($r^* = \rho(A)$): 임의 고윳값 $\lambda$, 고유벡터 $\mathbf{w}$ ($|\mathbf{w}|$를 잡으면):

$$
|\lambda| \cdot |\mathbf{w}| = |A \mathbf{w}| \leq A |\mathbf{w}|
$$

따라서 $|\lambda| \leq r(|\mathbf{w}|) \leq r^*$. 모든 $|\lambda| \leq r^*$, $r^*$는 고윳값이므로 $r^* = \rho(A)$.

**단계 3** (고유벡터 양수): $A\mathbf{v} = r^* \mathbf{v}$, $\mathbf{v} \geq 0$. $A > 0$이므로 $A\mathbf{v} > 0$, 따라서 $\mathbf{v} = A\mathbf{v}/r^* > 0$.

**단계 4** (단순성): 기하적 중복도가 2 이상이면 두 일차독립 고유벡터 $\mathbf{v}_1, \mathbf{v}_2$. 적절한 선형결합으로 성분 중 하나가 0이 되도록 만들 수 있고, $A$ 작용 후 $> 0$이 되어 모순.

**단계 5** (엄격한 도미넌스 $|\lambda| < \rho(A)$): 만약 $|\lambda| = \rho(A)$인 다른 $\lambda$가 있다면 $A |\mathbf{w}| = \rho |\mathbf{w}|$ (위의 부등식이 등식이 됨) → $|\mathbf{w}|$도 Perron 고유벡터 → 유일성 모순. $\blacksquare$

---

## 3. Perron-Frobenius (비음·기약)

### 정리 3.1 (Perron-Frobenius, 1912)

$A \geq 0$이 **기약**이면:

1. $\rho(A)$는 $A$의 고윳값 (양수)
2. 단순근
3. 고유벡터 $\mathbf{v} > 0$
4. 다른 $|\lambda| = \rho(A)$인 고윳값은 $\rho(A) e^{2\pi i k/h}$ ($k = 0, \ldots, h-1$), $h$는 **주기(period)**
5. 원시 ⟺ $h = 1$ ⟺ $|\lambda| < \rho(A)$ (다른 모든 $\lambda$)

### 증명 개요

$A$가 기약이지만 양은 아닐 수 있으므로, $A + \epsilon J$ ($J$는 모든 성분 1인 행렬)을 생각하고 Perron 정리 적용 후 $\epsilon \to 0$. 연속성으로 주요 성질 전이.

주기성: 원시가 아니면 주기 $h \geq 2$이 존재, 단위원 위 $h$번째 단위근 배치로 동일한 반지름 고윳값들이 나타남.

---

## 4. 기약성 판정

### 정리 4.1

다음은 동치:

1. $A \geq 0$이 기약
2. $(I + A)^{n-1} > 0$
3. 모든 $i, j$에 대해 $\exists k : (A^k)_{ij} > 0$

### 증명

(2 ⇒ 3): $(I+A)^{n-1}$는 길이 $\leq n-1$의 경로 개수의 양. 강연결 → $(I+A)^{n-1}_{ij} > 0$.

(3 ⇒ 1): 모든 $i, j$ 간 경로 존재 → 강연결.

(1 ⇒ 2): Cayley-Hamilton 및 Perron 정리로부터. $\blacksquare$

---

## 5. 확률 행렬과 정상분포

### 정의 5.1

**확률(행) 행렬** $P$: $P \geq 0$이고 각 행 합이 1 ($P \mathbf{1} = \mathbf{1}$).

### 정리 5.2

$P$가 확률 행렬이면 $\rho(P) = 1$, $\mathbf{1}$이 우측 고유벡터.

### 증명

$P \mathbf{1} = \mathbf{1}$로 1이 고윳값. 임의 $\lambda$, $P\mathbf{v} = \lambda\mathbf{v}$. 최대 성분 $v_{i^*}$에 대해:

$$
|\lambda v_{i^*}| = |(P\mathbf{v})_{i^*}| = \left| \sum_j P_{i^* j} v_j \right| \leq \sum_j P_{i^* j} |v_j| \leq |v_{i^*}|
$$

$|\lambda| \leq 1$. 1이 이미 고윳값이므로 $\rho = 1$. $\blacksquare$

### 정리 5.3 (정상분포)

$P$가 **기약**이고 **원시**(aperiodic)면 유일한 정상분포 $\boldsymbol{\pi}$가 존재:

$$
\boldsymbol{\pi}^T P = \boldsymbol{\pi}^T, \qquad \boldsymbol{\pi} > 0, \qquad \sum \pi_i = 1
$$

그리고 임의 초기 분포 $\mathbf{p}_0$에 대해 $\mathbf{p}_0^T P^k \to \boldsymbol{\pi}^T$.

### 증명

$P^T$에 Perron-Frobenius 적용. $\rho(P^T) = 1$의 좌측 고유벡터 (즉 $P$의 좌측 고유벡터) $\boldsymbol{\pi} > 0$이 유일 (단순성).

**수렴**: 원시 → 다른 모든 고윳값 $|\lambda| < 1$. Jordan 형태로 $P^k = \boldsymbol{\pi}\mathbf{1}^T + O(|\lambda_2|^k)$. $\blacksquare$

---

## 6. PageRank

### 문제

웹 그래프의 인접 행렬 $W$ (out-link), 각 페이지에서 링크 개수로 나눠 확률 행렬 $H_{ij} = W_{ij}/\text{out-degree}(i)$. 이로부터 **가장 중요한** 페이지 찾기.

### Damping

모든 출링크가 없는 페이지 (dangling), 사이클 문제 해결을 위해:

$$
G = \alpha H + (1 - \alpha) \frac{\mathbf{1}\mathbf{1}^T}{n}, \quad \alpha \approx 0.85
$$

$G$는 **모두 양수** → Perron 정리 직접 적용. Dominant left eigenvector $\boldsymbol{\pi}$가 PageRank 벡터.

### 알고리즘

Power iteration: $\boldsymbol{\pi}_{k+1}^T = \boldsymbol{\pi}_k^T G$. 수렴률 $\alpha \approx 0.85$ (매우 빠름).

---

## 7. Leslie Population Model

### 정의

연령 구조 개체 모델. $\mathbf{n}_t \in \mathbb{R}^k$는 연령대별 개체 수, Leslie 행렬:

$$
L = \begin{pmatrix}
f_1 & f_2 & \cdots & f_{k-1} & f_k \\
s_1 & 0 & \cdots & 0 & 0 \\
0 & s_2 & \cdots & 0 & 0 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & s_{k-1} & 0
\end{pmatrix}
$$

$f_i$: 출생률, $s_i$: 생존률.

### 정리

$L$이 원시이면 Perron-Frobenius로 dominant eigenvalue $\lambda_1 > 0$ (내재 성장률), 고유벡터가 **안정 연령 분포**.

$\lambda_1 > 1$: 증가, $\lambda_1 < 1$: 멸종, $\lambda_1 = 1$: 균형.

---

## 8. Python 실험

### 8.1 Perron 고유벡터

```python
import numpy as np

# 양 행렬
A = np.array([[0.5, 0.2, 0.3],
              [0.1, 0.7, 0.2],
              [0.4, 0.3, 0.3]])

eigvals, eigvecs = np.linalg.eig(A)
idx = np.argsort(np.abs(eigvals))[::-1]
eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]

print("Spectral radius:", np.abs(eigvals[0]))
print("Perron eigenvector (normalized positive):", eigvecs[:, 0] / eigvecs[:, 0].sum())
```

### 8.2 Power iteration

```python
def power_iteration(A, n_iter=100):
    n = A.shape[0]
    x = np.ones(n) / n
    for _ in range(n_iter):
        x = A @ x
        x /= np.linalg.norm(x)
    lam = (A @ x) @ x / (x @ x)
    return lam, x

lam, v = power_iteration(A, 50)
print("Perron root:", lam, "vs eig:", eigvals[0].real)
```

### 8.3 PageRank

```python
# 간단한 웹 그래프 (4 페이지)
# A → B, A → C
# B → C
# C → A, C → D
# D → A
W = np.array([[0, 1, 1, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 1],
              [1, 0, 0, 0]], dtype=float)
out_deg = W.sum(axis=1)
H = W / out_deg[:, None]

alpha = 0.85
n = 4
G = alpha * H + (1 - alpha) / n * np.ones((n, n))

# Left eigenvector (transpose)
pi = np.ones(n) / n
for _ in range(100):
    pi = pi @ G
    pi /= pi.sum()
print("PageRank:", pi)
# C와 A가 가장 중요한 편
```

### 8.4 마르코프 체인 정상분포

```python
# 날씨: 맑음/흐림/비
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])

# 정상분포는 P^T의 eigvector (eigval=1)
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigvals - 1))
pi = np.real(eigvecs[:, idx])
pi = pi / pi.sum()
print("Stationary:", pi)

# Power iteration
p = np.array([1.0, 0.0, 0.0])  # 오늘 맑음
for k in [1, 5, 20, 100]:
    p_k = p @ np.linalg.matrix_power(P, k)
    print(f"After {k} days:", p_k)
# 수렴 확인
```

### 8.5 기약성 체크

```python
def is_irreducible(A, tol=1e-10):
    n = A.shape[0]
    reachability = np.linalg.matrix_power(np.eye(n) + (A > tol).astype(float), n - 1)
    return np.all(reachability > 0)

# 기약: strongly connected
A1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # 순환
print("A1 irreducible?", is_irreducible(A1))  # True

# 비기약
A2 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
print("A2 irreducible?", is_irreducible(A2))  # False
```

---

## 9. 응용 요약

| 응용                | 핵심 결과                                  |
| ------------------- | ------------------------------------------ |
| PageRank            | Dominant eigenvector of damped Google matrix |
| 마르코프 체인       | 정상분포 = left Perron eigenvector        |
| Leslie 인구모델     | $\lambda_1$ = 내재 성장률                  |
| Input-Output 경제   | Leontief 모델의 해 존재성                  |
| 양자 통계역학       | 분배함수의 도미넌트 고유값                 |
| 그래프 이론         | 스펙트럼 갭으로 확장성, clustering         |

---

## 10. 요약

| 결과                          | 조건                     |
| ----------------------------- | ------------------------ |
| Perron                        | $A > 0$                  |
| Perron-Frobenius              | $A \geq 0$, 기약         |
| Perron 고윳값 **유일 도미넌트** | 원시 (= 기약 + aperiodic) |
| 고유벡터 양수                 | 기약 (항상), 원시 (strict) |
| 정상분포                      | 확률행렬 + 기약 + 원시   |

**핵심 함의**: 자연에 존재하는 비음 시스템(인구, 경제, 정보 흐름)은 구조상 Perron-Frobenius가 적용되어 **하나의 지배 모드**가 장기 거동을 결정한다.

---

## 11. 내비게이션

[◀ 03. Rayleigh 몫](./03-rayleigh-quotient.md) | [📚 README](../README.md) | [05. Power/QR 알고리즘 ▶](./05-power-qr-algorithm.md)
