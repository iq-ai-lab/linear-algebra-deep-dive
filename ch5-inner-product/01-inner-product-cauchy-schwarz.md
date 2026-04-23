# Ch5-01. 내적과 Cauchy-Schwarz 부등식

> "각도와 거리의 개념은 내적 하나에서 파생된다."

## 📌 학습 목표

- 내적의 공리적 정의와 기본 성질.
- **Cauchy-Schwarz 부등식**을 **세 가지 방법**으로 증명.
- 유도된 노름의 성질 (삼각부등식, 평행사변형 법칙).
- 내적 공간의 예들: $\mathbb{R}^n$, $\mathbb{C}^n$, $L^2$, 가중 내적, 행렬 Frobenius.

---

## 🎯 핵심 질문

> **질문 1**: 내적 공리의 최소한 **무엇이 빠지면** Cauchy-Schwarz가 무너지는가?
> **질문 2**: 두 벡터 사이의 각을 정의하려면 왜 부등식이 필요한가?
> **질문 3**: 함수 공간 $L^2$도 내적 공간인 이유?

---

## 1. 내적의 정의

### 정의 1.1 (실 내적)

실 벡터 공간 $V$ 위의 **내적(inner product)**은 사상 $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$로서 다음을 만족:

1. **대칭성**: $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$
2. **쌍선형성**: $\langle a\mathbf{x}_1 + b\mathbf{x}_2, \mathbf{y} \rangle = a\langle \mathbf{x}_1, \mathbf{y} \rangle + b\langle \mathbf{x}_2, \mathbf{y} \rangle$
3. **양의 정부호성**: $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$, $= 0 \iff \mathbf{x} = \mathbf{0}$

### 정의 1.2 (복소 내적, Hermitian)

복소 $V$ 위의 내적은 sesquilinear:

1. **켤레대칭**: $\langle \mathbf{x}, \mathbf{y} \rangle = \overline{\langle \mathbf{y}, \mathbf{x} \rangle}$
2. **첫 인자 선형, 둘째 인자 반선형** (또는 반대, 관례에 따름)
3. **양의 정부호**: $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$

(본 문서는 실 내적 중심.)

### 유도 노름

$$
\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}
$$

---

## 2. Cauchy-Schwarz 부등식

### 정리 2.1 (Cauchy-Schwarz)

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \|\mathbf{y}\|
$$

등호 조건: $\mathbf{x}, \mathbf{y}$가 선형종속.

### 증명 1 (판별식 방법)

$\mathbf{y} = \mathbf{0}$: 양변 0, 등호.

$\mathbf{y} \neq \mathbf{0}$: 모든 $t \in \mathbb{R}$에 대해:

$$
0 \leq \langle \mathbf{x} + t\mathbf{y}, \mathbf{x} + t\mathbf{y} \rangle = \|\mathbf{x}\|^2 + 2t\langle \mathbf{x}, \mathbf{y}\rangle + t^2 \|\mathbf{y}\|^2
$$

이차식 $\geq 0$ ⟹ 판별식 $\leq 0$:

$$
4\langle \mathbf{x}, \mathbf{y}\rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 \leq 0 \implies \langle \mathbf{x}, \mathbf{y}\rangle^2 \leq \|\mathbf{x}\|^2 \|\mathbf{y}\|^2 \quad \blacksquare
$$

### 증명 2 (투영)

$\mathbf{y} \neq \mathbf{0}$. $\mathbf{z} = \mathbf{x} - \frac{\langle \mathbf{x}, \mathbf{y}\rangle}{\|\mathbf{y}\|^2} \mathbf{y}$ ($\mathbf{y}$로의 투영 잔차).

$\langle \mathbf{z}, \mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{y}\rangle - \frac{\langle \mathbf{x}, \mathbf{y}\rangle}{\|\mathbf{y}\|^2} \|\mathbf{y}\|^2 = 0$, 즉 $\mathbf{z} \perp \mathbf{y}$.

Pythagoras:

$$
\|\mathbf{x}\|^2 = \|\mathbf{z}\|^2 + \frac{\langle \mathbf{x}, \mathbf{y}\rangle^2}{\|\mathbf{y}\|^2} \geq \frac{\langle \mathbf{x}, \mathbf{y}\rangle^2}{\|\mathbf{y}\|^2}
$$

양변 $\|\mathbf{y}\|^2$ 곱. $\blacksquare$

### 증명 3 (정규화 + AM-GM)

$\mathbf{x}, \mathbf{y}$가 단위 벡터라 가정 ($\|\mathbf{x}\| = \|\mathbf{y}\| = 1$). $0 \leq \|\mathbf{x} - \mathbf{y}\|^2 = 2 - 2\langle \mathbf{x}, \mathbf{y}\rangle$이므로 $\langle \mathbf{x}, \mathbf{y}\rangle \leq 1$.

$-\mathbf{y}$로도 같은 주장 → $|\langle \mathbf{x}, \mathbf{y}\rangle| \leq 1 = \|\mathbf{x}\|\|\mathbf{y}\|$. 일반 $\mathbf{x}, \mathbf{y}$는 단위화 후 스케일. $\blacksquare$

### 등호 조건

증명 2에서 $\|\mathbf{z}\| = 0 \iff \mathbf{z} = \mathbf{0} \iff \mathbf{x} = c\mathbf{y}$ (선형종속).

---

## 3. 노름과 거리

### 정리 3.1 (유도 노름의 성질)

$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x}\rangle}$는 다음을 만족:

1. $\|\mathbf{x}\| \geq 0$, $= 0 \iff \mathbf{x} = \mathbf{0}$
2. $\|c\mathbf{x}\| = |c| \|\mathbf{x}\|$
3. **삼각부등식**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### 증명 (삼각부등식)

$$
\|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{y}\|^2 \leq \|\mathbf{x}\|^2 + 2\|\mathbf{x}\|\|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\| + \|\mathbf{y}\|)^2
$$

(중간 부등식은 Cauchy-Schwarz.) 양변 양수이므로 제곱근. $\blacksquare$

### 거리

$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$. 거리 공리 성립 (대칭, 양수, 삼각).

---

## 4. 각도와 직교

### 정의 4.1 (각도)

$\mathbf{x}, \mathbf{y} \neq \mathbf{0}$에 대해:

$$
\cos\theta = \frac{\langle \mathbf{x}, \mathbf{y}\rangle}{\|\mathbf{x}\|\|\mathbf{y}\|}, \quad \theta \in [0, \pi]
$$

Cauchy-Schwarz로 $|\cos\theta| \leq 1$이므로 well-defined.

### 정의 4.2 (직교)

$$
\mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y}\rangle = 0
$$

### 정리 4.3 (Pythagoras)

$\mathbf{x} \perp \mathbf{y}$이면 $\|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$.

### 증명

$\|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$. $\blacksquare$

---

## 5. 평행사변형 법칙과 특성화

### 정리 5.1 (평행사변형 법칙)

$$
\|\mathbf{x} + \mathbf{y}\|^2 + \|\mathbf{x} - \mathbf{y}\|^2 = 2\|\mathbf{x}\|^2 + 2\|\mathbf{y}\|^2
$$

### 증명

직접 전개:

$$
\|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{y}\|^2
$$
$$
\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 - 2\langle\mathbf{x}, \mathbf{y}\rangle + \|\mathbf{y}\|^2
$$

합. $\blacksquare$

### 정리 5.2 (Jordan-von Neumann)

반대로, 노름이 평행사변형 법칙을 만족하면 **내적에서 유도**된 것. 구체적으로 (polarization identity):

$$
\langle \mathbf{x}, \mathbf{y}\rangle = \frac{1}{4}(\|\mathbf{x} + \mathbf{y}\|^2 - \|\mathbf{x} - \mathbf{y}\|^2)
$$

### 증명 개요

정의한 $\langle \cdot, \cdot\rangle$이 쌍선형인지 확인. 평행사변형 법칙이 additivity를 보장. 연속성으로 scalar linearity. $\blacksquare$

**의미**: $L^p$ 노름 중 $p = 2$만이 내적에서 유도 (나머지는 평행사변형 실패).

---

## 6. 예시들

### 6.1 $\mathbb{R}^n$ 표준 내적

$$
\langle \mathbf{x}, \mathbf{y}\rangle = \mathbf{x}^T \mathbf{y} = \sum_i x_i y_i
$$

### 6.2 $\mathbb{C}^n$ Hermitian 내적

$$
\langle \mathbf{x}, \mathbf{y}\rangle = \mathbf{x}^H \mathbf{y} = \sum_i \bar{x}_i y_i
$$

(주의: physics 관례는 $\langle \mathbf{x}, \mathbf{y}\rangle = \sum_i \bar y_i x_i$, 첫 인자 선형.)

### 6.3 가중 내적

대칭 PD $W$에 대해:

$$
\langle \mathbf{x}, \mathbf{y}\rangle_W = \mathbf{x}^T W \mathbf{y}
$$

$W = I$: 표준. $W = \Sigma^{-1}$: Mahalanobis (통계).

### 6.4 $L^2([a, b])$

함수 공간:

$$
\langle f, g\rangle = \int_a^b f(x) g(x)\, dx
$$

### 6.5 행렬 Frobenius 내적

$A, B \in \mathbb{R}^{m \times n}$:

$$
\langle A, B\rangle_F = \operatorname{tr}(A^T B) = \sum_{i, j} A_{ij} B_{ij}
$$

$\|A\|_F = \sqrt{\sum A_{ij}^2}$.

---

## 7. Cauchy-Schwarz의 응용

### 7.1 분산과 공분산

확률 변수 $X, Y$에 대해 $\operatorname{Cov}(X, Y)^2 \leq \operatorname{Var}(X) \operatorname{Var}(Y)$. 내적 $\langle X, Y\rangle = E[(X - EX)(Y - EY)]$에 Cauchy-Schwarz. 상관계수 $|\rho| \leq 1$.

### 7.2 Hölder 부등식 (일반화)

$\frac{1}{p} + \frac{1}{q} = 1$일 때 $\sum |a_i b_i| \leq (\sum|a_i|^p)^{1/p}(\sum|b_i|^q)^{1/q}$. $p = q = 2$가 Cauchy-Schwarz.

### 7.3 단순 결과

$\sum a_i \leq \sqrt{n} \sqrt{\sum a_i^2}$ ($a_i \geq 0$, $\mathbf{b} = \mathbf{1}$). 산술-제곱 평균 부등식의 변형.

### 7.4 적분

$\left|\int f g\right|^2 \leq \int f^2 \cdot \int g^2$. $L^2$ inner product의 직접 적용.

---

## 8. Python 실험

### 8.1 Cauchy-Schwarz 확인

```python
import numpy as np

for _ in range(5):
    x = np.random.randn(10)
    y = np.random.randn(10)
    lhs = abs(x @ y)
    rhs = np.linalg.norm(x) * np.linalg.norm(y)
    print(f"|<x,y>| = {lhs:.4f}, ||x||||y|| = {rhs:.4f}, ratio = {lhs/rhs:.4f}")
```

### 8.2 등호 조건

```python
x = np.random.randn(10)
y_parallel = 3.2 * x
print("Parallel case: ratio =", abs(x @ y_parallel) / (np.linalg.norm(x) * np.linalg.norm(y_parallel)))
# ≈ 1.0
```

### 8.3 각도 계산

```python
def angle(x, y):
    cos_t = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    cos_t = np.clip(cos_t, -1, 1)  # 수치 오차 방지
    return np.arccos(cos_t) * 180 / np.pi

print("angle([1,0], [0,1]):", angle(np.array([1,0]), np.array([0,1])))     # 90
print("angle([1,1], [1,0]):", angle(np.array([1,1]), np.array([1,0])))     # 45
```

### 8.4 L² 내적 (함수)

```python
from scipy.integrate import quad

def L2_inner(f, g, a=0, b=1):
    return quad(lambda x: f(x)*g(x), a, b)[0]

# f(x) = x, g(x) = x^2 on [0,1]
print("<f, g>:", L2_inner(lambda x: x, lambda x: x**2))   # 1/4
print("||f||:", np.sqrt(L2_inner(lambda x: x, lambda x: x)))  # sqrt(1/3)
print("||g||:", np.sqrt(L2_inner(lambda x: x**2, lambda x: x**2)))  # sqrt(1/5)
# Cauchy-Schwarz: 1/4 ≤ sqrt(1/3 * 1/5) ≈ 0.258
```

### 8.5 가중 내적 (Mahalanobis)

```python
Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

def mahalanobis(x, y):
    return (x - y) @ Sigma_inv @ (x - y)

x = np.array([1.0, 1.0])
y = np.array([0.0, 0.0])
print("Mahalanobis distance squared:", mahalanobis(x, y))
```

### 8.6 Frobenius 내적

```python
A = np.random.randn(3, 4)
B = np.random.randn(3, 4)
print("<A, B>_F:", np.trace(A.T @ B))
print("sum A*B:",  np.sum(A * B))  # elementwise sum
# 동일
```

---

## 9. 요약

| 개념                    | 정의/공식                                         |
| ----------------------- | ------------------------------------------------- |
| 내적 (실)               | 대칭, 쌍선형, 양정부                              |
| 유도 노름               | $\|\mathbf{x}\| = \sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$ |
| Cauchy-Schwarz          | $|\langle\mathbf{x},\mathbf{y}\rangle| \leq \|\mathbf{x}\|\|\mathbf{y}\|$ |
| 삼각부등식              | $\|\mathbf{x}+\mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ |
| 평행사변형 법칙         | $\|\mathbf{x}+\mathbf{y}\|^2 + \|\mathbf{x}-\mathbf{y}\|^2 = 2\|\mathbf{x}\|^2 + 2\|\mathbf{y}\|^2$ |
| Polarization identity   | $\langle\mathbf{x},\mathbf{y}\rangle = \frac{1}{4}(\|\mathbf{x}+\mathbf{y}\|^2 - \|\mathbf{x}-\mathbf{y}\|^2)$ |
| 각도                    | $\cos\theta = \langle\mathbf{x},\mathbf{y}\rangle/(\|\mathbf{x}\|\|\mathbf{y}\|)$ |

---

## 10. 내비게이션

[◀ 이전 챕터: Randomized SVD](../ch4-svd/06-randomized-svd.md) | [📚 README](../README.md) | [02. 정사영 ▶](./02-orthogonal-projection.md)
