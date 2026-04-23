# Ch4-03. Moore-Penrose Pseudoinverse

> "모든 행렬에 대한 '최적의' 역행렬: 존재하지 않을 때에도 정의된다."

## 📌 학습 목표

- Moore-Penrose pseudoinverse의 **4가지 공리적 조건**으로 정의.
- SVD를 통한 **구성적** 공식 $A^+ = V\Sigma^+ U^T$.
- 과결정/부족결정 시스템의 **최소 노름 최소 제곱** 해법.
- Full rank 경우의 공식 $A^+ = (A^T A)^{-1} A^T$ 유도.

---

## 🎯 핵심 질문

> **질문 1**: 역행렬이 없어도 "해"가 **유일**하게 정의되는 이유는?
> **질문 2**: 4가지 Moore-Penrose 조건의 기하적 의미는?
> **질문 3**: Pseudoinverse가 정규방정식과 어떻게 연결되는가?

---

## 1. 공리적 정의

### 정의 1.1 (Moore-Penrose)

$A \in \mathbb{R}^{m \times n}$에 대해 $A^+ \in \mathbb{R}^{n \times m}$이 다음 **네 조건**을 모두 만족하면 $A$의 **Moore-Penrose 역행렬**:

1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$ (대칭)
4. $(A^+ A)^T = A^+ A$ (대칭)

### 정리 1.2 (존재와 유일성)

모든 $A$에 대해 $A^+$가 **유일**하게 존재한다.

### 증명

**존재**: SVD를 이용해 구성 (§2).

**유일성**: $B, C$가 모두 조건 만족. $B = BAB = B(AB)^T = BB^T A^T$. 마찬가지 조작으로 $C = C C^T A^T$. 조건들을 반복 사용:

$$
B = BAB \overset{(3)}{=} B(AB)^T = B B^T A^T
$$

다른 식으로 $C = CAC$, $A^T = (AC)^T A^T = (CAC A)^T A^T$... 등 꽤 장황하지만 결국 $B = BAB = BAC = CAC = C$. (중간에 $AB = (AB)^T = (AC)^T (A B)^T \cdots$ 여러 단계.) 표준 증명 참조. $\blacksquare$

---

## 2. SVD 기반 공식

### 정리 2.1

$A = U \Sigma V^T$ (SVD). $\Sigma^+ \in \mathbb{R}^{n \times m}$를 다음으로 정의:

$$
(\Sigma^+)_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0 \\ 0 & \sigma_i = 0 \end{cases}
$$

그리고 $\Sigma^+$의 다른 성분은 0. 그러면:

$$
A^+ = V \Sigma^+ U^T
$$

### 증명

4가지 조건 직접 확인. $\Sigma \Sigma^+ = \operatorname{diag}(1, \ldots, 1, 0, \ldots, 0)$ ($r$개 1, 나머지 0, $m \times m$). 마찬가지로 $\Sigma^+ \Sigma$은 $n \times n$ 대각 $r$개 1.

1. $AA^+ A = U\Sigma V^T \cdot V\Sigma^+ U^T \cdot U\Sigma V^T = U \Sigma \Sigma^+ \Sigma V^T = U\Sigma V^T = A$. ✓
2. $A^+ A A^+ = V\Sigma^+ U^T U \Sigma V^T V\Sigma^+ U^T = V\Sigma^+\Sigma\Sigma^+ U^T = V\Sigma^+ U^T = A^+$. ✓
3. $(AA^+)^T = (U\Sigma\Sigma^+ U^T)^T = U\Sigma\Sigma^+ U^T = AA^+$ (대각은 대칭). ✓
4. 유사. ✓

따라서 $A^+ = V\Sigma^+ U^T$. $\blacksquare$

---

## 3. 특수한 경우들

### 3.1 $A$ 정칙 ($n \times n$, full rank)

$\sigma_i > 0$ for all $i$. $\Sigma^+ = \Sigma^{-1}$:

$$
A^+ = V \Sigma^{-1} U^T = A^{-1}
$$

### 3.2 $A$ full column rank ($m \geq n$, rank $n$)

$$
A^+ = (A^T A)^{-1} A^T
$$

### 증명

$A^T A = V \Sigma^T \Sigma V^T = V (\operatorname{diag}\sigma_i^2) V^T$. $(A^T A)^{-1} = V (\operatorname{diag}\sigma_i^{-2}) V^T$.

$$
(A^T A)^{-1} A^T = V (\operatorname{diag}\sigma_i^{-2}) V^T V \Sigma^T U^T = V \operatorname{diag}(\sigma_i^{-2} \sigma_i) U^T = V \Sigma^+ U^T = A^+ \quad \blacksquare
$$

### 3.3 $A$ full row rank ($m \leq n$, rank $m$)

$$
A^+ = A^T (A A^T)^{-1}
$$

(유사 증명.)

---

## 4. 최소 제곱과 최소 노름

### 4.1 과결정 ($m > n$, full column rank)

**문제**: $A\mathbf{x} = \mathbf{b}$에 정확해 없음. 대신 $\min \|A\mathbf{x} - \mathbf{b}\|^2$.

### 정리 4.1

$\mathbf{x}^* = A^+ \mathbf{b} = (A^T A)^{-1} A^T \mathbf{b}$이 유일한 최소 제곱 해.

### 증명 (정규방정식)

$f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2 = \mathbf{x}^T A^T A \mathbf{x} - 2\mathbf{b}^T A \mathbf{x} + \|\mathbf{b}\|^2$.

$\nabla f = 2A^T A \mathbf{x} - 2 A^T \mathbf{b} = \mathbf{0} \implies A^T A \mathbf{x}^* = A^T \mathbf{b}$.

Full column rank → $A^T A$ 정칙 → $\mathbf{x}^* = (A^T A)^{-1} A^T \mathbf{b}$. $\blacksquare$

### 4.2 부족결정 ($m < n$, full row rank)

**문제**: $A\mathbf{x} = \mathbf{b}$는 무한히 많은 해. 그 중 $\|\mathbf{x}\|$ 최소.

### 정리 4.2

$\mathbf{x}^* = A^+ \mathbf{b} = A^T (A A^T)^{-1} \mathbf{b}$.

### 증명 (Lagrangian)

Constraint $A\mathbf{x} = \mathbf{b}$ with objective $\min \|\mathbf{x}\|^2$. Lagrangian $L = \|\mathbf{x}\|^2 + 2\boldsymbol{\lambda}^T(\mathbf{b} - A\mathbf{x})$. KKT:

$$
2\mathbf{x} - 2 A^T \boldsymbol{\lambda} = 0 \implies \mathbf{x} = A^T \boldsymbol{\lambda}
$$

Constraint: $A A^T \boldsymbol{\lambda} = \mathbf{b} \implies \boldsymbol{\lambda} = (AA^T)^{-1} \mathbf{b}$. 따라서 $\mathbf{x}^* = A^T (AA^T)^{-1} \mathbf{b}$. $\blacksquare$

### 4.3 일반적 경우 (rank-deficient)

**문제**: $\min \|A\mathbf{x} - \mathbf{b}\|$. 해집합은 affine subspace. 그 중 $\|\mathbf{x}\|$ 최소.

### 정리 4.3

$\mathbf{x}^* = A^+ \mathbf{b}$는 다음을 만족:

1. $\mathbf{x}^*$는 residual 최소화: $\|A\mathbf{x}^* - \mathbf{b}\| \leq \|A\mathbf{x} - \mathbf{b}\|$ for all $\mathbf{x}$.
2. Residual 최소화 집합 내에서 $\|\mathbf{x}^*\|$ 최소.

### 증명 (SVD 기반)

$\mathbf{x} = V\mathbf{y}$, $\mathbf{b} = U\mathbf{c}$ + orthogonal residual $\mathbf{c}_\perp$. 그러면:

$$
A\mathbf{x} - \mathbf{b} = U\Sigma V^T \cdot V\mathbf{y} - U\mathbf{c} - \mathbf{c}_\perp = U(\Sigma \mathbf{y} - \mathbf{c}) - \mathbf{c}_\perp
$$

$\|A\mathbf{x} - \mathbf{b}\|^2 = \|\Sigma\mathbf{y} - \mathbf{c}\|^2 + \|\mathbf{c}_\perp\|^2$. 최소는:

- $i \leq r$: $\sigma_i y_i = c_i$, $y_i = c_i/\sigma_i$
- $i > r$ (null space): $y_i$ 자유 → 최소 노름 위해 $y_i = 0$

$\mathbf{y}^* = \Sigma^+ \mathbf{c}$, $\mathbf{x}^* = V\Sigma^+ U^T \mathbf{b} = A^+ \mathbf{b}$. $\blacksquare$

---

## 5. Pseudoinverse의 기하

### 5.1 투영 연산자

$AA^+$와 $A^+A$는 대칭 멱등 → **정사영(orthogonal projection)**:

- $A A^+$: $\mathbb{R}^m$에서 $R(A) = C(A)$ 위로의 사영
- $A^+ A$: $\mathbb{R}^n$에서 $R(A^T) = $ row space 위로의 사영

### 증명

$AA^+ AA^+ = AA^+$ (조건 1). 대칭 (조건 3). $R(AA^+) = R(A)$ ($AA^+ \mathbf{x} = A \cdot A^+ \mathbf{x} \in R(A)$).

### 5.2 분해

임의 $\mathbf{b} \in \mathbb{R}^m$:

$$
\mathbf{b} = \underbrace{AA^+ \mathbf{b}}_{\in C(A)} + \underbrace{(I - AA^+) \mathbf{b}}_{\in N(A^T)}
$$

유일한 최소 제곱 해는 $\mathbf{x}^* = A^+ \mathbf{b}$이고, residual은 $(I - AA^+) \mathbf{b}$ (left null space).

---

## 6. Pseudoinverse의 대수적 성질

### 정리 6.1

- $(A^+)^+ = A$
- $(A^T)^+ = (A^+)^T$
- $(cA)^+ = c^{-1} A^+$ ($c \neq 0$)
- 일반적으로 $(AB)^+ \neq B^+ A^+$!

### 6.2 특수 경우 $(AB)^+ = B^+ A^+$

다음 중 하나가 성립할 때:

- $A$ full column rank 그리고 $B$ full row rank
- $B = A^H$ (에르미트/전치)
- $A = B^H$

일반적으로 $AB$의 SVD는 $A, B$의 SVD로 간단히 표현되지 않음.

---

## 7. Tikhonov Regularization과의 관계

### 정의 7.1

**Ridge/Tikhonov 해**:

$$
\mathbf{x}_\alpha = \arg\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|^2 + \alpha \|\mathbf{x}\|^2 = (A^T A + \alpha I)^{-1} A^T \mathbf{b}
$$

### 정리 7.2

$$
\lim_{\alpha \to 0^+} \mathbf{x}_\alpha = A^+ \mathbf{b}
$$

### 증명

SVD로:

$$
\mathbf{x}_\alpha = V (\Sigma^T \Sigma + \alpha I)^{-1} \Sigma^T U^T \mathbf{b} = V \operatorname{diag}\left(\frac{\sigma_i}{\sigma_i^2 + \alpha}\right) U^T \mathbf{b}
$$

$\alpha \to 0$:

- $\sigma_i > 0$: $\sigma_i / (\sigma_i^2) = 1/\sigma_i$ (pseudoinverse 값)
- $\sigma_i = 0$: $0 / \alpha = 0$ (pseudoinverse도 0)

따라서 $\mathbf{x}_\alpha \to V \Sigma^+ U^T \mathbf{b} = A^+ \mathbf{b}$. $\blacksquare$

### 의미

Tikhonov는 작은 특이값에서 $1/\sigma_i \to \sigma_i/(\sigma_i^2 + \alpha)$로 완화 → 조건수 악화 방지.

---

## 8. Python 실험

### 8.1 기본 Pseudoinverse

```python
import numpy as np

# 과결정
A = np.random.randn(5, 3)
b = np.random.randn(5)

A_pinv_np = np.linalg.pinv(A)
x_ls = A_pinv_np @ b
print("x (pinv):", x_ls)
print("||A x - b||:", np.linalg.norm(A @ x_ls - b))

# SVD 기반 수동
U, s, Vt = np.linalg.svd(A, full_matrices=False)
s_inv = np.where(s > 1e-12, 1/s, 0)
A_pinv_svd = Vt.T @ np.diag(s_inv) @ U.T
print("||pinv - svd-pinv||:", np.linalg.norm(A_pinv_np - A_pinv_svd))
```

### 8.2 Moore-Penrose 조건 검증

```python
A = np.random.randn(4, 3)
A_pinv = np.linalg.pinv(A)

print("1. ||A A+ A - A||:", np.linalg.norm(A @ A_pinv @ A - A))
print("2. ||A+ A A+ - A+||:", np.linalg.norm(A_pinv @ A @ A_pinv - A_pinv))
print("3. ||(A A+)^T - A A+||:", np.linalg.norm((A @ A_pinv).T - A @ A_pinv))
print("4. ||(A+ A)^T - A+ A||:", np.linalg.norm((A_pinv @ A).T - A_pinv @ A))
```

### 8.3 최소 노름 해 (부족결정)

```python
A = np.random.randn(3, 5)   # 3 equations, 5 unknowns
b = np.random.randn(3)

x_min_norm = np.linalg.pinv(A) @ b
print("||A x - b||:", np.linalg.norm(A @ x_min_norm - b))   # ≈ 0
print("||x||:", np.linalg.norm(x_min_norm))

# 임의의 다른 해
x_other = np.linalg.lstsq(A, b, rcond=None)[0]  # same as pinv here
# Add null-space component
null_basis = np.linalg.svd(A)[2][3:]  # last 2 rows of V^T
x_alt = x_min_norm + null_basis[0] * 1.0
print("||A x_alt - b||:", np.linalg.norm(A @ x_alt - b))   # still 0
print("||x_alt||:", np.linalg.norm(x_alt))                 # larger
```

### 8.4 Tikhonov to Pseudoinverse

```python
# Ill-conditioned matrix
U = np.linalg.qr(np.random.randn(5, 5))[0]
V = np.linalg.qr(np.random.randn(3, 3))[0]
A = U[:, :3] @ np.diag([1, 0.01, 1e-6]) @ V.T  # small σ_3

b = np.random.randn(5)
for alpha in [1.0, 0.01, 1e-6, 0]:
    if alpha > 0:
        x_tik = np.linalg.solve(A.T @ A + alpha*np.eye(3), A.T @ b)
    else:
        x_tik = np.linalg.pinv(A) @ b
    print(f"α={alpha:.0e}, ||x||={np.linalg.norm(x_tik):.3f}, ||Ax-b||={np.linalg.norm(A@x_tik-b):.3f}")
```

---

## 9. 요약

| 상황                       | $A^+$ 공식                   |
| -------------------------- | ---------------------------- |
| $A$ 정칙                   | $A^{-1}$                     |
| Full column rank $m \geq n$| $(A^T A)^{-1} A^T$           |
| Full row rank $m \leq n$   | $A^T (AA^T)^{-1}$            |
| 일반                       | $V \Sigma^+ U^T$             |

**용법**:
- 과결정 → 최소 제곱
- 부족결정 → 최소 노름
- Rank-deficient → 둘 다 (최소 제곱 + 최소 노름)
- Regularization → Tikhonov, $\alpha \to 0$에서 $A^+$와 일치

---

## 10. 참고 문헌

- Penrose, R. (1955). *A generalized inverse for matrices*. Proc. Cambridge Philos. Soc.
- Ben-Israel & Greville (2003). *Generalized Inverses*. Springer.
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.

---

## 11. 내비게이션

[◀ 02. SVD 존재성](./02-svd-existence.md) | [📚 README](../README.md) | [04. Eckart-Young 정리 ▶](./04-eckart-young.md)
