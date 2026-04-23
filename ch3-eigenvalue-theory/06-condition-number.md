# Ch3-06. 조건수와 고유값의 민감도

> "비대칭 행렬의 고윳값은 **무한히 민감**할 수 있다."

## 📌 학습 목표

- 고윳값 문제의 **조건수**를 수학적으로 정의한다.
- Bauer-Fike 정리: 대각화 가능 행렬의 고윳값 민감도 한계.
- **비대칭 행렬의 민감도 폭발**을 Wilkinson's polynomial 예제로 본다.
- Pseudospectrum과 실제 고윳값과의 차이.

---

## 🎯 핵심 질문

> **질문 1**: 왜 대칭 고윳값은 Lipschitz인데 비대칭은 아닌가?
> **질문 2**: Wilkinson 다항식이 고전 수치해석의 "사건"인 이유?
> **질문 3**: Pseudospectrum은 실무에 어떤 도움을 주는가?

---

## 1. 대각화 가능 행렬의 고윳값 민감도

### 정리 1.1 (Bauer-Fike)

$A = P D P^{-1}$ (대각화 가능), $\tilde{A} = A + E$. $\tilde{A}$의 임의 고윳값 $\tilde{\mu}$에 대해 $A$의 어떤 고윳값 $\lambda$가 존재하여:

$$
|\tilde{\mu} - \lambda| \leq \kappa_p(P) \|E\|_p
$$

(any $p$-norm).

### 증명

$\tilde{\mu}$가 $\tilde{A}$의 고윳값, $\tilde{\mathbf{v}} \neq 0$ 고유벡터: $(A + E - \tilde{\mu} I)\tilde{\mathbf{v}} = 0$.

만약 $\tilde{\mu}$가 $A$의 고윳값이면 $|\tilde{\mu} - \tilde{\mu}| = 0$로 끝. 아니면 $A - \tilde{\mu} I$가 가역:

$$
(I + (A - \tilde{\mu} I)^{-1} E) \tilde{\mathbf{v}} = 0
$$

$(A - \tilde{\mu} I)^{-1} E \tilde{\mathbf{v}} = -\tilde{\mathbf{v}}$, $\|(A - \tilde{\mu} I)^{-1} E\| \geq 1$, 따라서:

$$
1 \leq \|(A - \tilde{\mu} I)^{-1}\| \|E\|
$$

$A = PDP^{-1}$로 $(A - \tilde{\mu} I)^{-1} = P (D - \tilde{\mu} I)^{-1} P^{-1}$:

$$
\|(A - \tilde{\mu}I)^{-1}\|_p \leq \|P\|_p \|P^{-1}\|_p \max_i \frac{1}{|\lambda_i - \tilde{\mu}|} = \kappa_p(P) \max_i \frac{1}{|\lambda_i - \tilde{\mu}|}
$$

부등식 대입:

$$
1 \leq \kappa_p(P) \frac{\|E\|}{\min_i |\lambda_i - \tilde{\mu}|}
$$

정리: $\min_i |\lambda_i - \tilde{\mu}| \leq \kappa_p(P) \|E\|$. $\blacksquare$

### 해석

- 대칭/에르미트: $P = U$ (유니타리), $\kappa_2(U) = 1$ → Weyl 부등식과 일치
- **$P$가 매우 비직교일수록** (거의 defective) 고윳값이 민감해짐

---

## 2. 개별 고윳값의 조건수

### 정의 2.1 (고윳값 조건수)

$A$의 단순 고윳값 $\lambda_i$, 우측 고유벡터 $\mathbf{v}_i$, 좌측 고유벡터 $\mathbf{u}_i$ ($\mathbf{u}_i^H A = \lambda_i \mathbf{u}_i^H$) 정규화 ($\|\mathbf{v}_i\| = \|\mathbf{u}_i\| = 1$):

$$
\operatorname{cond}(\lambda_i) = \frac{1}{|\mathbf{u}_i^H \mathbf{v}_i|}
$$

### 정리 2.2

1차 섭동 $A + \epsilon E$의 고윳값:

$$
\tilde{\lambda}_i = \lambda_i + \epsilon \frac{\mathbf{u}_i^H E \mathbf{v}_i}{\mathbf{u}_i^H \mathbf{v}_i} + O(\epsilon^2)
$$

따라서:

$$
|\tilde{\lambda}_i - \lambda_i| \leq \operatorname{cond}(\lambda_i) \cdot \|E\|_2 \cdot \epsilon + O(\epsilon^2)
$$

### 증명

$\tilde{\lambda}(\epsilon)$, $\tilde{\mathbf{v}}(\epsilon)$이 해석적이라 가정: $\tilde{\lambda}(0) = \lambda_i$, $\tilde{\mathbf{v}}(0) = \mathbf{v}_i$.

$(A + \epsilon E) \tilde{\mathbf{v}}(\epsilon) = \tilde{\lambda}(\epsilon) \tilde{\mathbf{v}}(\epsilon)$. $\epsilon$에 대해 미분, $\epsilon = 0$:

$$
A \dot{\mathbf{v}} + E \mathbf{v}_i = \dot{\lambda} \mathbf{v}_i + \lambda_i \dot{\mathbf{v}}
$$

$\mathbf{u}_i^H$를 좌측 곱:

$$
\mathbf{u}_i^H A \dot{\mathbf{v}} + \mathbf{u}_i^H E \mathbf{v}_i = \dot{\lambda} \mathbf{u}_i^H \mathbf{v}_i + \lambda_i \mathbf{u}_i^H \dot{\mathbf{v}}
$$

$\mathbf{u}_i^H A = \lambda_i \mathbf{u}_i^H$이므로 $\mathbf{u}_i^H A \dot{\mathbf{v}} = \lambda_i \mathbf{u}_i^H \dot{\mathbf{v}}$, 상쇄:

$$
\mathbf{u}_i^H E \mathbf{v}_i = \dot{\lambda} \mathbf{u}_i^H \mathbf{v}_i \implies \dot{\lambda} = \frac{\mathbf{u}_i^H E \mathbf{v}_i}{\mathbf{u}_i^H \mathbf{v}_i} \quad \blacksquare
$$

### 해석

- **대칭**: $\mathbf{u}_i = \mathbf{v}_i$, $\mathbf{u}_i^H \mathbf{v}_i = 1$, $\operatorname{cond} = 1$. 완벽히 안정.
- **거의 defective**: 우·좌 고유벡터가 직교에 가까워짐 → $\operatorname{cond} \to \infty$.

---

## 3. Wilkinson's Polynomial

### 정의 3.1

$$
W_{20}(x) = \prod_{k=1}^{20} (x - k) = (x-1)(x-2)\cdots(x-20)
$$

근은 $1, 2, \ldots, 20$ (아주 깨끗).

### 섭동 실험

$x^{19}$ 계수를 $10^{-7}$만큼 변화시킨 후 근을 계산.

**결과**: 일부 근이 **복소수 쌍**으로 변하고, 실수부가 $\sim 5$ 단위 이동. 계수의 $10^{-7}$ 섭동이 근을 엄청나게 이동시킴.

### 왜?

$W_{20}$의 companion matrix는 극단적으로 비직교. Companion 행렬의 좌·우 고유벡터 조건수:

$$
\operatorname{cond}(\lambda_k) = \prod_{j \neq k} \frac{|j|}{|j - k|}
$$

$k = 15, 16$ 근처에서 $\operatorname{cond} \sim 10^{13}$. 계수의 $10^{-7}$ 오차가 근에서 $10^6$ 증폭.

### 실무 교훈

**다항식의 근을 companion matrix의 고유값으로 계산하는 것은 위험할 수 있다**. 그러나 현대 LAPACK은 specialized balancing과 다중 shift로 많은 경우 처리.

---

## 4. Pseudospectrum

### 정의 4.1 ($\epsilon$-pseudospectrum)

$$
\Lambda_\epsilon(A) = \{z \in \mathbb{C} : \|(zI - A)^{-1}\| \geq \epsilon^{-1}\} = \bigcup_{\|E\| \leq \epsilon} \sigma(A + E)
$$

즉, $\epsilon$ 크기의 섭동으로 도달 가능한 모든 고유값의 합집합.

### 정리 4.2

다음은 동치:

1. $z \in \Lambda_\epsilon(A)$
2. $\|(zI - A)^{-1}\| \geq \epsilon^{-1}$
3. $\exists E, \|E\| \leq \epsilon$: $z \in \sigma(A + E)$
4. $\exists \mathbf{v}, \|\mathbf{v}\| = 1$: $\|(A - zI)\mathbf{v}\| \leq \epsilon$

### 해석

- **정규 행렬**: $\Lambda_\epsilon = \sigma(A) + B_\epsilon(0)$ (고유값 근방의 단순 $\epsilon$-이웃)
- **비정규 행렬**: $\Lambda_\epsilon$이 훨씬 크게 퍼짐, 고유값 영역이 왜곡

### 응용

- **ODE 안정성**: $\dot{\mathbf{x}} = A\mathbf{x}$에서 과도적 증폭은 pseudospectrum이 지배
- **반복법 수렴**: GMRES 수렴률이 fearless pseudospectrum에 따름
- **양자 역학**: 비-Hermitian 시스템의 관측 가능량 분석

---

## 5. Hot take: 수치 문제의 3계층

1. **조건이 좋은 경우**: 대칭/에르미트, 정규 → 알고리즘 관계없이 안정
2. **조건이 나쁘지만 알고리즘으로 극복**: Balancing, Householder, partial pivoting, QR shift
3. **본질적으로 민감**: Wilkinson poly 같은 비정상적 구조는 어떤 알고리즘도 구할 수 없음

---

## 6. Python 실험

### 6.1 Wilkinson 다항식

```python
import numpy as np

roots_true = np.arange(1, 21)
coeffs = np.poly(roots_true)  # monic

# 섭동: x^19 계수를 1e-7 증가
coeffs_pert = coeffs.copy()
coeffs_pert[1] += 1e-7

roots_pert = np.sort_complex(np.roots(coeffs_pert))
print("True roots:     ", roots_true)
print("Perturbed (real):", roots_pert.real.round(3))
# 중반부 근이 완전히 이동, 일부는 복소수가 됨
```

### 6.2 고유값 조건수

```python
A = np.array([[1.0, 1.0, 1.0],
              [0.0, 2.0, 1.0],
              [0.0, 0.0, 3.0]])

eigvals, right = np.linalg.eig(A)
_, left = np.linalg.eig(A.T)

# 좌/우 고유벡터 매칭 (eigenvalue 순서로)
# 실제로는 주의 필요 — 여기서는 unique eigvals 가정
# left eigvec u satisfies u^T A = λ u^T, 즉 A^T u = λ u

# np.linalg.eig on A.T gives left eigvecs
# conditioning: 1 / |u^T v|
for k in range(3):
    vi = right[:, k]
    vi /= np.linalg.norm(vi)
    # Find left eigvec with same λ
    idx = np.argmin(np.abs(eigvals[k] - np.linalg.eigvals(A.T)))
    ui = left[:, idx]
    ui /= np.linalg.norm(ui)
    cond_i = 1.0 / abs(ui @ vi)
    print(f"λ={eigvals[k]:.3f}, cond(λ)={cond_i:.2f}")
```

### 6.3 Pseudospectrum 시각화

```python
A = np.array([[-1.0, 10.0],
              [0.0, -1.0]])  # Non-normal

# Grid of complex plane
x = np.linspace(-3, 1, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

# Resolvent norm
res_norm = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        z = Z[i, j]
        try:
            M = z * np.eye(2) - A
            res_norm[i, j] = 1.0 / np.linalg.norm(M, -2)  # ≈ 1/σ_min
            # Actually want ||(zI-A)^{-1}|| = 1/σ_min(zI - A)
            res_norm[i, j] = 1.0 / np.linalg.svd(M, compute_uv=False)[-1]
        except:
            res_norm[i, j] = np.inf

import matplotlib.pyplot as plt
plt.contourf(X, Y, np.log10(res_norm), levels=20)
plt.plot(-1, 0, 'ro', markersize=10)  # true eigenvalue
plt.colorbar(label='log10 ||(zI-A)^-1||')
plt.title("Pseudospectrum of non-normal A")
plt.xlabel('Re'); plt.ylabel('Im')
```

### 6.4 Bauer-Fike 확인

```python
P = np.random.randn(5, 5)
D = np.diag(np.random.randn(5))
A = P @ D @ np.linalg.inv(P)

E = 0.001 * np.random.randn(5, 5)
A_pert = A + E

eigs_A = np.sort_complex(np.linalg.eigvals(A))
eigs_pert = np.sort_complex(np.linalg.eigvals(A_pert))
print("True:     ", eigs_A.real.round(3))
print("Perturbed:", eigs_pert.real.round(3))

max_diff = max(abs(e1 - e2) for e1, e2 in zip(eigs_A, eigs_pert))
cond_P = np.linalg.cond(P)
bound = cond_P * np.linalg.norm(E, 2)
print(f"max diff: {max_diff:.4f}, Bauer-Fike bound: {bound:.4f}")
```

---

## 7. 요약

| 행렬 클래스       | 조건수            | 섭동 감도                     |
| ----------------- | ----------------- | ----------------------------- |
| Normal (대칭 포함)| $\kappa(P) = 1$  | Lipschitz (Weyl)              |
| Diagonalizable    | $\kappa(P)$ finite | Bauer-Fike bound            |
| Defective (Jordan)| $\infty$         | Fractional ($\epsilon^{1/m}$)|

**실무 가이드**:
- 대칭/에르미트 문제는 `eigh` 사용 → 안전
- 비대칭은 조심: `np.linalg.cond(eigvecs)` 체크
- 극단 상황은 pseudospectrum 분석

---

## 8. 참고 문헌

- Bauer, F. L., & Fike, C. T. (1960). *Norms and exclusion theorems*. Numer. Math.
- Wilkinson, J. H. (1965). *The Algebraic Eigenvalue Problem*.
- Trefethen, L. N., & Embree, M. (2005). *Spectra and Pseudospectra*. Princeton.
- Horn & Johnson, *Matrix Analysis*, Ch 6.

---

## 9. 챕터 3 요약

본 챕터에서 다룬 내용:
1. 특성다항식과 Cayley-Hamilton
2. 고유값의 기하학적 의미 (확대/회전/전단)
3. Rayleigh 몫 (최적화 표현)
4. Perron-Frobenius (비음 행렬)
5. Power iteration / QR 알고리즘 (수치 계산)
6. 조건수와 pseudospectrum (민감도 분석)

→ Ch 4 (SVD)로 이어짐: 비정사각 / 특이 행렬의 "적절한" 스펙트럼.

---

## 10. 다음 챕터 예고

- Ch 4: **특이값 분해(SVD)** — 모든 행렬에 존재하는 "가장 일반적" 분해

---

## 11. 내비게이션

[◀ 05. Power/QR 알고리즘](./05-power-qr-algorithm.md) | [📚 README](../README.md) | [다음 챕터: SVD ▶](../ch4-svd/01-svd-geometric.md)
