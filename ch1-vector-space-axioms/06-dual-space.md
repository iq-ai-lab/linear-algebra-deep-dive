# 06. 이중공간(Dual Space)

## 🎯 핵심 질문

- **쌍대공간** $V^* = \mathcal{L}(V, \mathbb{R})$이란 무엇이고, 유한차원에서 왜 $\dim V^* = \dim V$인가?
- "자연동형" $V \cong V^*$은 왜 기저 선택에 의존하는가? 내적을 도입하면 왜 **자연**해지는가 (Riesz)?
- 왜 **행벡터 vs 열벡터**, **공변 vs 반변**이라는 구분이 본질적인가?
- Backpropagation의 gradient가 왜 "쌍대벡터"인지?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Gradient는 쌍대벡터**: $\nabla L \in V^*$로 보는 것이 더 자연스럽다. 손실함수 $L: V \to \mathbb{R}$이 선형 근사 $L(\mathbf{v} + h\mathbf{u}) \approx L(\mathbf{v}) + h \langle \nabla L, \mathbf{u}\rangle$ 할 때 $\nabla L$은 $V^*$의 원소로서 "입력을 스칼라로 보내는" 선형 범함수.
- **Vector-Jacobian Product**: PyTorch/JAX의 `backward()`가 계산하는 것은 $v^\top J$ (Jacobian의 왼쪽 곱), 이는 **쌍대공간의 pullback**.
- **Pull-back과 Push-forward**: 신경망 층을 통한 gradient 전달은 쌍대공간의 사상 $T^*: W^* \to V^*$. 이것이 역방향으로 흐르는 이유다.
- **좌표 변환의 공변-반변**: 데이터 좌표가 $P$로 변환되면 gradient는 $P^{-\top}$로 변환 (반변적). Information Geometry의 e-좌표·m-좌표 쌍대성이 여기서 시작.
- **Legendre 변환**: Convex optimization의 conjugate $f^*(\mathbf{p}) = \sup_\mathbf{x}(\mathbf{p}^\top\mathbf{x} - f(\mathbf{x}))$에서 $\mathbf{p} \in V^*$, $\mathbf{x} \in V$. 원공간-쌍대공간 변환.

"열벡터와 행벡터는 사실 다른 공간에 산다"는 관점이 고급 선형대수의 출발점이다.

---

## 📐 수학적 선행 조건

- [Ch1-01 벡터공간](./01-vector-space-8-axioms.md)
- [Ch1-03 선형변환](./03-linear-transformation-matrix.md)

---

## 📖 직관적 이해

### 쌍대벡터 = "선형 측정기"

$\phi \in V^*$는 벡터를 받아서 수를 내는 **선형 측정 도구**. 예:
- "3번째 좌표만 뽑아내는" $\phi(\mathbf{v}) = v_3$
- "모든 성분을 더하는" $\phi(\mathbf{v}) = \sum v_i$
- "$\mathbf{w}$와의 내적" $\phi(\mathbf{v}) = \mathbf{w}^\top \mathbf{v}$

### 행벡터 vs 열벡터

$\mathbf{v} \in \mathbb{R}^n$ (열벡터)와 $\phi \in (\mathbb{R}^n)^*$ (행벡터)는 **다른 공간의 원소**. 행렬곱 $\phi \cdot \mathbf{v}$ ($1 \times n \cdot n \times 1$ = 스칼라)로 자연스럽게 "계산"된다.

유한차원에서 $\mathbb{R}^n \cong (\mathbb{R}^n)^*$이지만, 이 동형은 **기저 선택에 의존**한다. 내적을 도입하면 자연스러워진다(Riesz).

> **비유**: 원공간 $V$는 "위치", 쌍대공간 $V^*$는 "속도의 측정 단위". 둘은 차원은 같지만 물리적으로 다른 대상이다.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — 쌍대공간

$V$가 $\mathbb{F}$-벡터공간일 때, **쌍대공간**:

$$V^* = \mathcal{L}(V, \mathbb{F}) = \{\phi: V \to \mathbb{F} \mid \phi \text{ 선형}\}.$$

$V^*$는 $\mathbb{F}$-벡터공간 (선형변환공간 자체가 벡터공간).

### 정의 6.2 — 쌍대기저 (Dual Basis)

$V$의 기저 $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$에 대해, **쌍대기저** $\{\mathbf{e}^1, \ldots, \mathbf{e}^n\}$을

$$\mathbf{e}^i(\mathbf{e}_j) = \delta^i_j = \begin{cases}1 & i = j\\ 0 & i \neq j\end{cases}$$

로 정의. (상첨자 = 쌍대, 하첨자 = 원공간. 텐서 표기의 관례.)

### 정의 6.3 — Pull-back (쌍대사상)

선형변환 $T: V \to W$가 주어지면 **쌍대사상** $T^*: W^* \to V^*$:

$$T^*(\phi)(\mathbf{v}) = \phi(T(\mathbf{v})), \quad \phi \in W^*, \mathbf{v} \in V.$$

### 정의 6.4 — 이중쌍대 (Double Dual)

$V^{**} = (V^*)^*$. 자연 사상 $\iota: V \to V^{**}$, $\iota(\mathbf{v})(\phi) = \phi(\mathbf{v})$.

---

## 🔬 정리와 증명

### 정리 6.1 — $\dim V^* = \dim V$

**명제**: 유한차원 $V$에서 쌍대기저 $\{\mathbf{e}^i\}$가 $V^*$의 기저이며, $\dim V^* = n = \dim V$.

**증명**:

**선형독립**: $\sum_i \alpha_i \mathbf{e}^i = 0$ (영 범함수)이라 하자. $\mathbf{e}_j$에 적용: $\sum_i \alpha_i \mathbf{e}^i(\mathbf{e}_j) = \alpha_j = 0$. 따라서 모든 $\alpha_i = 0$.

**생성**: 임의 $\phi \in V^*$. $\phi(\mathbf{e}_j) = c_j$라 하면 $\phi = \sum_j c_j \mathbf{e}^j$임을 보인다. 임의 $\mathbf{v} = \sum v^j \mathbf{e}_j$에 대해

$$\phi(\mathbf{v}) = \sum v^j \phi(\mathbf{e}_j) = \sum v^j c_j, \quad \left(\sum_j c_j \mathbf{e}^j\right)(\mathbf{v}) = \sum_j c_j \mathbf{e}^j(\mathbf{v}) = \sum_j c_j v^j.$$

둘이 같으므로 $\phi = \sum c_j \mathbf{e}^j$. $\square$

---

### 정리 6.2 — $V \cong V^*$ (기저 선택 의존)

**명제**: 유한차원에서 $V$와 $V^*$는 동형이다. 그러나 **"자연" 동형은 기저를 고르지 않으면 존재하지 않는다**.

**증명**: 정리 6.1로 $\dim V = \dim V^*$이고, 같은 차원의 두 유한차원 벡터공간은 동형. 기저 $\{\mathbf{e}_i\}$ 고정 시 사상 $\mathbf{e}_i \mapsto \mathbf{e}^i$이 동형.

**기저 의존성**: 다른 기저 $\{\mathbf{e}_i'\}$로 하면 쌍대기저도 달라지고, 자연 동형 대응도 달라진다. 내적 없이는 "어느 동형을 선택할지" 내재적 기준이 없다. $\square$

---

### 정리 6.3 — $V \cong V^{**}$ (자연 동형)

**명제**: 사상 $\iota: V \to V^{**}$, $\iota(\mathbf{v})(\phi) = \phi(\mathbf{v})$는 **자연 동형**이다 (기저 선택 무관).

**증명**:

**선형**: $\iota(\alpha \mathbf{u} + \mathbf{v})(\phi) = \phi(\alpha \mathbf{u} + \mathbf{v}) = \alpha\phi(\mathbf{u}) + \phi(\mathbf{v}) = (\alpha \iota(\mathbf{u}) + \iota(\mathbf{v}))(\phi)$.

**단사**: $\iota(\mathbf{v}) = 0$이면 모든 $\phi(\mathbf{v}) = 0$. 특히 어떤 기저 $\{\mathbf{e}_i\}$의 쌍대 $\{\mathbf{e}^i\}$로 평가하면 $\mathbf{e}^i(\mathbf{v}) = v^i = 0$ (모든 $i$). 따라서 $\mathbf{v} = \mathbf{0}$.

**$\dim$ 같음** (정리 6.1 두 번 적용) + 단사 → 전사. $\square$

이 정리가 중요한 이유: $\iota$의 정의가 **기저를 쓰지 않음**. 따라서 $V^{**}$는 $V$와 "자연스럽게 같다".

---

### 정리 6.4 — 쌍대사상과 행렬 전치

**명제**: $T: V \to W$, 각 공간의 기저를 $\{\mathbf{e}_j\}, \{\mathbf{f}_i\}$로 하면 $[T^*]_{\mathbf{e}^* \mathbf{f}^*} = [T]_{\mathbf{f}\mathbf{e}}^\top$.

**증명**: $T(\mathbf{e}_j) = \sum_i a_{ij} \mathbf{f}_i$. $T^*(\mathbf{f}^k)(\mathbf{e}_j) = \mathbf{f}^k(T(\mathbf{e}_j)) = \mathbf{f}^k\left(\sum_i a_{ij} \mathbf{f}_i\right) = a_{kj}$. 

즉 $T^*(\mathbf{f}^k) = \sum_j a_{kj} \mathbf{e}^j$, 그 행렬 성분 $(j, k)$는 $a_{kj}$ = 원래 행렬의 $(k, j)$. 따라서 **전치**. $\square$

> 이것이 **"전치의 기하학적 의미"**: 전치행렬은 쌍대공간 사이의 사상. Backprop의 $v^\top J$가 곧 $J^\top v$이므로 Jacobian의 쌍대사상으로 해석된다.

---

### 정리 6.5 — Riesz 표현정리 (유한차원판)

**명제**: $V$가 유한차원 내적공간이면 모든 $\phi \in V^*$에 대해 **유일한** $\mathbf{w}_\phi \in V$가 존재해

$$\phi(\mathbf{v}) = \langle \mathbf{w}_\phi, \mathbf{v}\rangle, \quad \forall \mathbf{v} \in V.$$

따라서 사상 $V \to V^*$, $\mathbf{w} \mapsto \langle \mathbf{w}, \cdot\rangle$이 **자연 동형** (내적을 고정하면).

**증명**: 정규직교 기저 $\{\mathbf{u}_i\}$ 고정. $\phi(\mathbf{u}_i) = c_i$라 하면 $\mathbf{w}_\phi := \sum c_i \mathbf{u}_i$가 조건을 만족:

$$\langle \mathbf{w}_\phi, \mathbf{u}_j\rangle = c_j = \phi(\mathbf{u}_j) \implies \phi(\mathbf{v}) = \langle \mathbf{w}_\phi, \mathbf{v}\rangle \text{ for all } \mathbf{v}.$$

**유일성**: $\mathbf{w}_1, \mathbf{w}_2$가 모두 조건 만족 시 $\langle \mathbf{w}_1 - \mathbf{w}_2, \mathbf{v}\rangle = 0$ for all $\mathbf{v}$ → $\mathbf{w}_1 = \mathbf{w}_2$ (내적의 비퇴화성). $\square$

**해석**: Gradient $\nabla L$은 원래 $L \in V^*$의 원소(쌍대)이지만 Riesz로 $V$의 원소와 동일시된다 — 내적을 고정했을 때. 내적이 바뀌면 gradient 표현도 달라짐(Natural Gradient의 출발점).

---

## 💻 NumPy 검증

```python
import numpy as np

# ─────────────────────────────────────────────
# 1. 쌍대기저의 Kronecker delta 성질
# ─────────────────────────────────────────────
n = 4
rng = np.random.default_rng(0)

# 임의의 기저 (열벡터들)
E = rng.standard_normal((n, n))
while np.linalg.matrix_rank(E) < n:
    E = rng.standard_normal((n, n))

# 쌍대기저 = E의 역행렬의 행들
E_dual = np.linalg.inv(E)

# δᵢⱼ 검증: dual_i · e_j = δᵢⱼ
I_check = E_dual @ E
print("E_dual · E (should be I):")
print(np.round(I_check, 10))

# ─────────────────────────────────────────────
# 2. V ≅ V* 동형 (기저 의존)
# ─────────────────────────────────────────────
# 임의의 선형 범함수 φ: R^n → R,  φ(v) = c^T v
c = rng.standard_normal(n)
# 표준기저의 쌍대 → c가 그대로 표현
# 다른 기저 E의 쌍대에서는 좌표가 달라짐
c_in_E_dual = E.T @ c     # 기저변경 후 좌표

v = rng.standard_normal(n)
# 표준 계산
phi_v_std = c @ v
# E 기저에서 v의 좌표 × 쌍대기저 좌표
v_in_E = np.linalg.solve(E, v)
phi_v_new = c_in_E_dual @ v_in_E
assert np.isclose(phi_v_std, phi_v_new)
print(f"\nφ(v) = {phi_v_std:.6f}  (표준기저)")
print(f"φ(v) = {phi_v_new:.6f}  (E 기저에서, 쌍대 좌표 변환 후)")

# ─────────────────────────────────────────────
# 3. T*의 행렬 = Tᵀ (정리 6.4)
# ─────────────────────────────────────────────
m = 3
T = rng.standard_normal((m, n))     # T: R^n → R^m

# T*(ψ)(v) = ψ(Tv) = (Tᵀψ)·v
psi = rng.standard_normal(m)        # ψ ∈ (R^m)*
v = rng.standard_normal(n)
lhs = psi @ (T @ v)
rhs = (T.T @ psi) @ v
assert np.isclose(lhs, rhs)
print(f"\nψ(Tv) = (Tᵀψ)·v :  {lhs:.6f} = {rhs:.6f} ✓")

# ─────────────────────────────────────────────
# 4. Riesz 표현: gradient가 쌍대 원소 (정리 6.5)
# ─────────────────────────────────────────────
# L(x) = ½ xᵀ M x + bᵀ x,  ∇L(x) = Mx + b ∈ V (Riesz 동일시)
M = rng.standard_normal((n, n))
M = (M + M.T) / 2
b = rng.standard_normal(n)

def L(x): return 0.5 * x @ M @ x + b @ x
def grad_L(x): return M @ x + b

x = rng.standard_normal(n)
h = 1e-5
u = rng.standard_normal(n)

# 방향미분 L'(x)[u] = ⟨∇L, u⟩ (Riesz)
lhs = (L(x + h*u) - L(x - h*u)) / (2 * h)
rhs = grad_L(x) @ u
print(f"\n방향미분 ≈ ⟨∇L, u⟩:  {lhs:.6f} ≈ {rhs:.6f} ✓")
```

---

## 🔗 AI/ML 연결

### Backprop = Pullback of Functionals

Forward: $\mathbf{x} \to \mathbf{y} = T(\mathbf{x}) \to L = \phi(\mathbf{y})$. $L$을 $\mathbf{x}$에 대해 미분하면 $\mathbf{x}$의 쌍대 원소:

$$\frac{\partial L}{\partial \mathbf{x}} = T^*\left(\frac{\partial L}{\partial \mathbf{y}}\right).$$

이것이 VJP: $v^\top J$이 **쌍대공간의 pullback** $T^*$. JAX `vjp()`은 이 연산을 직접 드러낸다.

### Natural Gradient와 Riesz

Euclidean gradient $\nabla L$은 **표준 내적 + Riesz**로 얻은 원공간 표현. Fisher 계량 $F$로 내적을 바꾸면 Riesz 대응이 달라져 $\tilde{\nabla} L = F^{-1} \nabla L$이 natural gradient. (Information Geometry 레포 Ch5).

### Legendre 변환

Convex conjugate $f^*(\mathbf{p}) = \sup_\mathbf{x}(\mathbf{p}^\top\mathbf{x} - f(\mathbf{x}))$에서 $\mathbf{p} \in V^*$, $\mathbf{x} \in V$. Exponential family의 canonical parameter $\theta \in V$와 expectation parameter $\eta \in V^*$의 쌍대가 이 구조.

### Co-/Contra-variance in DL

Feature normalizer가 좌표를 $P$로 변환하면 데이터는 $P$로, gradient는 $(P^\top)^{-1}$로 변환. 이 차이가 BatchNorm 이후 학습 역동학의 변화를 설명한다 (Ch7-03).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 유한차원 | 무한차원에서는 topological dual (연속 범함수)과 algebraic dual이 다름 |
| 기저 선택 | Natural iso 없음 — 좌표-자유 표현이 텐서 대수의 주제 (Ch6) |
| 내적 고정 (Riesz) | 다른 내적 → 다른 Riesz 대응 → 다른 gradient |

---

## 📌 핵심 정리

$$\boxed{V^* = \mathcal{L}(V, \mathbb{F}), \quad \dim V^* = \dim V, \quad V^{**} \cong V \text{ naturally}}$$

$$\boxed{T: V \to W \implies T^*: W^* \to V^*, \quad [T^*] = [T]^\top}$$

$$\boxed{\text{Riesz: } V \cong V^* \text{ via } \mathbf{w} \mapsto \langle \mathbf{w}, \cdot\rangle \text{ (내적 고정 후)}}$$

---

## 🤔 생각해볼 문제

**문제 1**: $\mathbb{R}^3$의 기저 $\{\mathbf{e}_1 = (1,1,0), \mathbf{e}_2 = (1,0,1), \mathbf{e}_3 = (0,1,1)\}$의 쌍대기저를 구하라.

<details>
<summary>해설</summary>

$E = \begin{pmatrix}1&1&0\\1&0&1\\0&1&1\end{pmatrix}$, $E^{-1}$의 행이 쌍대. $\det E = -2$, $E^{-1} = \frac{1}{2}\begin{pmatrix}-1&1&1\\1&1&-1\\1&-1&1\end{pmatrix}$. 따라서 $\mathbf{e}^1 = \frac{1}{2}(-1,1,1)$, $\mathbf{e}^2 = \frac{1}{2}(1,1,-1)$, $\mathbf{e}^3 = \frac{1}{2}(1,-1,1)$. 검증: $\mathbf{e}^i \cdot \mathbf{e}_j = \delta^i_j$.

</details>

**문제 2**: $V = C[0,1]$ (무한차원)에서 $\phi(f) = f(0.5)$, $\psi(f) = \int_0^1 f$는 쌍대공간의 원소인가?

<details>
<summary>해설</summary>

둘 다 선형 범함수이므로 algebraic dual의 원소. 하지만 **연속성**이 추가로 필요하면 (topological dual), 적절한 노름(예: sup-norm)에서 둘 다 연속 → continuous dual에도 속함. Dirac evaluation $\delta_{0.5}$는 $L^2$-노름에서는 연속이 아니다 (distribution이 된다). Function space의 dual은 Functional Analysis 레포에서 상세.

</details>

**문제 3** (AI 연결): JAX의 `jax.grad(f)`와 `jax.vjp(f, x)`는 쌍대공간 관점에서 어떻게 다른가? `vjp`이 왜 backprop의 기본 연산인지 설명하라.

<details>
<summary>해설</summary>

`grad(f)` (스칼라 $f$)은 Riesz 대응으로 $\nabla f \in V$를 반환 (내적 기반). `vjp(f, x)`는 $f: V \to W$에서 쌍대사상 $T^*: W^* \to V^*$을 반환 — 즉 함수로서 $w \mapsto J^\top w$. 스칼라 손실에서는 둘이 동일하지만 vector-output일 때 vjp이 더 근본. Backprop은 $L: \text{params} \to \mathbb{R}$의 gradient를 구하는데 중간 층들이 vector-valued이므로 **층마다 vjp 필요**. forward-mode(jvp)는 $V \to W$, reverse-mode(vjp)는 $W^* \to V^*$. 출력 차원 ≪ 입력 차원일 때 reverse-mode가 효율적인 이유 (딥러닝의 전형).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. 4개의 기본 부분공간](./05-four-fundamental-subspaces.md) | [📚 README](../README.md) | [Ch2-01. LU 분해 ▶](../ch2-matrix-decomposition/01-lu-decomposition.md) |

</div>
