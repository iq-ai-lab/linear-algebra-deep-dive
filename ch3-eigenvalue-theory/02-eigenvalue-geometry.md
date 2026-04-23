# Ch3-02. 고윳값의 기하적 의미: 변환의 불변 방향

> "고유벡터는 행렬이 **단순한 스칼라 곱처럼 보이는** 방향이다."

## 📌 학습 목표

- 고윳값과 고유벡터의 기하학적 의미: 불변 방향과 확대율을 이해한다.
- 2×2 행렬의 **회전/반사/확대**를 고윳값 패턴으로 분류한다.
- 복소 고윳값의 기하학적 의미 (실 2차원에서의 회전).
- Spectral radius와 선형 시스템의 장기 거동.
- 양의 정부호의 기하적 의미: 타원체.

---

## 🎯 핵심 질문

> **질문 1**: 복소 고윳값이 실행렬에서 무엇을 의미하는가?
> **질문 2**: 반복 $\mathbf{x}_{k+1} = A \mathbf{x}_k$의 장기 거동은 무엇이 결정하는가?
> **질문 3**: PD 행렬이 타원체를 그리는 이유는?

---

## 1. 기본 기하학: 불변 방향

### 정의 1.1

고유벡터 $\mathbf{v}$는 $A$에 의해 **방향이 보존**되는 벡터. $A$의 작용은 이 방향에서 순수한 **스칼라 곱** $\lambda \mathbf{v}$.

### 다이어그램적 해석

| $\lambda$         | 기하학적 행위                   |
| ----------------- | ------------------------------- |
| $\lambda > 1$     | 확대 (stretch)                  |
| $0 < \lambda < 1$ | 축소 (contract)                 |
| $\lambda = 1$     | 불변 (fixed)                    |
| $\lambda = 0$     | 영공간으로 붕괴 (collapse)      |
| $\lambda = -1$    | 반사 (reflection)               |
| $\lambda < 0$     | 반사 + 확축                      |
| 복소 $\lambda$    | 회전 + 확축 (2차원 실 invariant subspace) |

---

## 2. 2×2 행렬 분류

### 2.1 판별식으로 분류

$A \in \mathbb{R}^{2 \times 2}$, $p_A(\lambda) = \lambda^2 - \operatorname{tr}(A) \lambda + \det(A)$. 판별식 $\Delta = \operatorname{tr}(A)^2 - 4\det(A)$.

- $\Delta > 0$: 두 실수 고윳값 → 두 서로 다른 실 방향으로 독립 작용
- $\Delta = 0$: 중근 → 대각화 가능 (scalar × I) 또는 Jordan
- $\Delta < 0$: 복소 공액 쌍 $a \pm bi$ → 회전 + 확축

### 2.2 행렬 유형

**확대 행렬**: $A = \operatorname{diag}(a, b)$, $a, b > 0$, $a \neq b$. 두 주축이 고유벡터.

**회전 행렬**: $R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$. 고윳값 $e^{\pm i\theta}$. 실 고유벡터 **없음** ($\theta \neq 0, \pi$일 때).

**반사 행렬**: 예를 들어 $A = \operatorname{diag}(1, -1)$. 고윳값 $1, -1$. $x$축은 불변, $y$축은 반사.

**전단(shear)**: $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. 중근 1, 기하적 중복도 1 (defective). 고유벡터는 $x$축뿐.

---

## 3. 복소 고윳값의 실 해석

### 정리 3.1

실행렬 $A \in \mathbb{R}^{n \times n}$의 복소 고윳값 $\lambda = a + bi$ ($b \neq 0$)에 대한 복소 고유벡터 $\mathbf{v} = \mathbf{u} + i\mathbf{w}$ ($\mathbf{u}, \mathbf{w} \in \mathbb{R}^n$)가 있으면, 실 2차원 불변 부분공간 $\operatorname{span}(\mathbf{u}, \mathbf{w})$가 존재하고 이 위에서 $A$는:

$$
\begin{pmatrix} a & -b \\ b & a \end{pmatrix} = \sqrt{a^2+b^2} R_\theta \quad (\theta = \arg \lambda)
$$

**즉, $|\lambda|$ 배 확축 + 각 $\arg \lambda$만큼 회전**.

### 증명

$A\mathbf{v} = \lambda \mathbf{v}$를 실수부 허수부로 분해:

$$
A(\mathbf{u} + i\mathbf{w}) = (a + bi)(\mathbf{u} + i\mathbf{w}) = (a\mathbf{u} - b\mathbf{w}) + i(b\mathbf{u} + a\mathbf{w})
$$

실수부: $A\mathbf{u} = a\mathbf{u} - b\mathbf{w}$. 허수부: $A\mathbf{w} = b\mathbf{u} + a\mathbf{w}$.

행렬로:

$$
A [\mathbf{u} \mid \mathbf{w}] = [\mathbf{u} \mid \mathbf{w}] \begin{pmatrix} a & b \\ -b & a \end{pmatrix}
$$

(전치 주의: 실제로는 $[A\mathbf{u}, A\mathbf{w}] = [\mathbf{u}, \mathbf{w}] \begin{pmatrix} a & b \\ -b & a \end{pmatrix}$. $\blacksquare$)

### 응용

2차 선형 ODE $\ddot{x} + \omega^2 x = 0$을 $\dot{\mathbf{x}} = A\mathbf{x}$로 전환 시 $A = \begin{pmatrix} 0 & 1 \\ -\omega^2 & 0 \end{pmatrix}$. 고윳값 $\pm i\omega$, 원형 궤적 (진동).

---

## 4. Spectral Radius와 장기 거동

### 정의 4.1

**Spectral radius**:

$$
\rho(A) = \max_i |\lambda_i|
$$

### 정리 4.2 (Power method의 수렴 한계)

대각화 가능 $A$, $|\lambda_1| > |\lambda_2| \geq \cdots$면 거의 모든 $\mathbf{x}_0$에 대해:

$$
\frac{A^k \mathbf{x}_0}{\|A^k \mathbf{x}_0\|} \to \pm \mathbf{v}_1 \quad (k \to \infty)
$$

그리고 수렴 속도는 $\left|\frac{\lambda_2}{\lambda_1}\right|^k$.

### 증명

$\mathbf{x}_0 = \sum_i c_i \mathbf{v}_i$, $c_1 \neq 0$ (generic). $A^k \mathbf{x}_0 = \sum_i c_i \lambda_i^k \mathbf{v}_i = \lambda_1^k \left[c_1 \mathbf{v}_1 + \sum_{i \geq 2} c_i (\lambda_i/\lambda_1)^k \mathbf{v}_i\right]$. $|\lambda_i/\lambda_1| < 1$이므로 나머지 항 → 0. $\blacksquare$

### 정리 4.3 (스펙트럼 반지름 공식)

$$
\rho(A) = \lim_{k \to \infty} \|A^k\|^{1/k}
$$

임의 연산자 노름에 대해.

### 따름정리 4.4

$\mathbf{x}_{k+1} = A\mathbf{x}_k$의 장기 거동:

- $\rho(A) < 1$: $\mathbf{x}_k \to \mathbf{0}$ (안정)
- $\rho(A) > 1$: $\|\mathbf{x}_k\| \to \infty$ (불안정)
- $\rho(A) = 1$: 경계 (Jordan 블록 존재 여부에 따라)

동역학 시스템, 마르코프 체인, 반복 해법 모두 $\rho(A)$로 지배된다.

---

## 5. 양의 정부호와 타원체

### 정의 5.1

$A \succ 0$ (PD, 대칭). 이차형식 $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$의 레벨셋 $\{f = 1\}$은 $\mathbb{R}^n$에서 **타원체**.

### 정리 5.2 (타원체 주축 정리)

$A = Q \Lambda Q^T$ (스펙트럼 분해). 타원체 $\mathbf{x}^T A \mathbf{x} = 1$의 주축은 $Q$의 열 방향, 주축 길이는 $1/\sqrt{\lambda_i}$.

### 증명

$\mathbf{y} = Q^T \mathbf{x}$로 좌표변환. $\mathbf{x}^T A \mathbf{x} = \mathbf{y}^T \Lambda \mathbf{y} = \sum \lambda_i y_i^2 = 1$. 이것이 $\mathbf{y}$-좌표계에서 축-정렬된 타원. $i$번째 축 반지름은 $1/\sqrt{\lambda_i}$. $\blacksquare$

### 예시

$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. 고윳값 $3, 1$, 고유벡터 $(1,1)/\sqrt{2}, (1,-1)/\sqrt{2}$. 타원체:

- 주축 (1,1)/√2 방향, 길이 $1/\sqrt{3}$
- 주축 (1,-1)/√2 방향, 길이 $1/\sqrt{1} = 1$

## 6. 동역학적 해석: 상상자의 정리들

### 6.1 마르코프 체인

확률 행렬 $P$ ($P_{ij} \geq 0$, 각 행 합 1). 고윳값 1이 항상 존재하며 $\rho(P) = 1$ (Perron, Ch3-04에서 정식 다룸).

**정상분포**: $\boldsymbol{\pi}^T P = \boldsymbol{\pi}^T$, 즉 $P^T$의 고윳값 1 고유벡터.

### 6.2 연속 시간 시스템

$\dot{\mathbf{x}} = A\mathbf{x}$의 해 $\mathbf{x}(t) = e^{At} \mathbf{x}_0$. 장기 거동은 **실수부가 최대**인 고윳값이 지배:

- $\operatorname{Re} \lambda_i < 0$ 모두: 원점으로 수렴 (안정)
- $\operatorname{Re} \lambda_i > 0$ 하나라도: 발산

### 6.3 PageRank

Google의 랭킹은 인접 확률 행렬 $M$의 dominant eigenvector (Perron vector). Power iteration으로 계산.

---

## 7. Python 실험

### 7.1 2×2 행렬 유형 분류

```python
import numpy as np

def classify_2x2(A):
    tr = np.trace(A)
    det = np.linalg.det(A)
    disc = tr**2 - 4*det
    if disc > 1e-10:
        typ = "두 실수 고윳값 (확대/반사)"
    elif abs(disc) < 1e-10:
        typ = "중근 (Jordan 가능)"
    else:
        typ = "복소 고윳값 (회전+확축)"
    print(f"tr={tr:.3f}, det={det:.3f}, Δ={disc:.3f} → {typ}")
    print(f"eigvals = {np.linalg.eigvals(A)}")

classify_2x2(np.array([[2.0, 0.0], [0.0, 3.0]]))       # 확대
classify_2x2(np.array([[0.0, -1.0], [1.0, 0.0]]))      # 회전 90°
classify_2x2(np.array([[1.0, 1.0], [0.0, 1.0]]))       # shear (중근 1)
classify_2x2(np.array([[1.0, 0.0], [0.0, -1.0]]))      # 반사
```

### 7.2 복소 고윳값 → 실 2D 회전

```python
A = np.array([[0.5, -1.0], [1.0, 0.5]])
# 고윳값 0.5 ± i → 반경 √(0.25+1) = √1.25
eigs = np.linalg.eigvals(A)
print("Eigenvalues:", eigs, "|λ|:", np.abs(eigs))

# 반복 적용
x = np.array([1.0, 0.0])
traj = [x.copy()]
for _ in range(30):
    x = A @ x
    traj.append(x.copy())
traj = np.array(traj)

import matplotlib.pyplot as plt
plt.plot(traj[:, 0], traj[:, 1], 'o-')
plt.axis('equal'); plt.grid()
plt.title("Spiral: |λ|>1 then rotation")
# 회전하면서 지수적 확대
```

### 7.3 타원체 주축

```python
A = np.array([[2.0, 1.0], [1.0, 2.0]])
eigs, Q = np.linalg.eigh(A)
print("Principal axes:", Q)
print("Semi-axis lengths (1/sqrt(lambda)):", 1/np.sqrt(eigs))

# 타원 샘플링
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])
# 변환 x^T A x = 1 → x = Q Λ^{-1/2} y, y on unit circle
ellipse = Q @ np.diag(1/np.sqrt(eigs)) @ circle

plt.plot(ellipse[0], ellipse[1])
plt.axis('equal'); plt.grid()
plt.title("Ellipse x^T A x = 1")
```

### 7.4 Spectral radius 과 수렴

```python
# 안정 vs 불안정
A_stab = 0.9 * np.array([[np.cos(0.3), -np.sin(0.3)],
                          [np.sin(0.3),  np.cos(0.3)]])
A_unst = 1.1 * np.array([[np.cos(0.3), -np.sin(0.3)],
                          [np.sin(0.3),  np.cos(0.3)]])

for name, A in [("stable", A_stab), ("unstable", A_unst)]:
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    x = np.ones(2)
    for _ in range(100):
        x = A @ x
    print(f"{name}: ρ={rho:.3f}, ||x_100||={np.linalg.norm(x):.3e}")
```

---

## 8. 요약

| 고윳값 패턴            | 기하적 의미                         |
| ---------------------- | ----------------------------------- |
| 실수, 양수             | 확대/축소 방향                      |
| 실수, 음수             | 반사 + 확축                         |
| 복소 공액 쌍 $a \pm bi$| 2D 회전 ($\arg\lambda$) + $|\lambda|$배 확축 |
| 중근, 기하 < 대수      | Shear (대각화 불가)                 |
| $\|\lambda\| > 1$      | 확대 (반복 적용 시 발산)            |
| $\|\lambda\| < 1$      | 축소 (반복 적용 시 수렴)            |

**핵심 원리**:
- **Spectral radius** $\rho(A)$가 반복 시스템의 안정성 결정
- **PD** 행렬은 타원체 생성, 주축 = 고유벡터
- **복소** 고윳값은 실 부분공간에서 **회전**으로 해석

---

## 9. 참고 문헌

- Strang, G. (2006). *Linear Algebra and Its Applications*, Ch 6.
- Meyer, C. D. (2000). *Matrix Analysis and Applied Linear Algebra*, Ch 7.
- Hirsch, Smale, Devaney. *Differential Equations, Dynamical Systems, and an Introduction to Chaos*.

---

## 10. 다음 문서

- **[03. Rayleigh 몫](./03-rayleigh-quotient.md)**: 고윳값을 최적화 문제로 표현.

---

## 11. 내비게이션

[◀ 01. 특성다항식](./01-characteristic-polynomial.md) | [📚 README](../README.md) | [03. Rayleigh 몫 ▶](./03-rayleigh-quotient.md)
