# 04. Rank-Nullity 정리

## 🎯 핵심 질문

- 왜 $\dim(\ker T) + \dim(\text{im } T) = \dim V$가 **정확히** 성립하는가?
- "잃어버린 차원(kernel)"과 "보존된 차원(image)"의 합이 왜 원래 차원인가?
- $Ax = b$의 해 공간이 왜 **affine subspace** $x_p + \ker A$ 인가?
- 신경망에서 "effective rank"와 "degree of freedom"의 관계는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **과/미결정 시스템**: Least squares $\min\|Ax - b\|$에서 $A$의 rank가 시스템이 유일해·무한해·해 없음 중 어느 쪽인지 결정.
- **Hessian의 rank-deficient**: 신경망의 손실 Hessian $H$에서 $\ker H$의 방향은 **학습에 영향 없는 방향**(flat direction). Sagun et al. (2016)은 실제 Hessian이 **매우 rank-deficient**함을 관찰.
- **Dropout의 Effective Rank**: Dropout은 각 스텝에서 **임의의 부분공간으로 projection**한다. 이는 kernel의 무작위 확장.
- **LoRA**: $W + BA$ ($A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$)에서 $BA$의 rank는 최대 $r$. Rank-Nullity로 $\ker(BA)$의 차원은 $\geq d - r$.
- **Policy collapse**: RL에서 정책 분포가 dirac으로 수축되면 Jacobian rank가 떨어짐 → 탐험 차원 상실.

---

## 📐 수학적 선행 조건

- [Ch1-01~03](./01-vector-space-8-axioms.md)
- Subspace, 기저, 차원 (Ch1-02)
- 선형변환과 행렬 표현 (Ch1-03)

---

## 📖 직관적 이해

### Kernel = "선형변환이 0으로 보내는 입력들"

$\ker T = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$은 "$T$가 구별 못 하는 방향"의 집합. 이 방향을 아무리 움직여도 출력이 안 바뀐다.

### Image = "도달 가능한 출력"

$\text{im } T = \{T(\mathbf{v}) : \mathbf{v} \in V\} \subseteq W$은 "실제로 생성되는 출력"의 부분공간.

### 보존의 법칙

$\dim V = \dim(\ker T) + \dim(\text{im } T)$는 **차원의 에너지 보존**같은 것. "입력 공간의 차원은 (구별 못 하는 차원) + (구별되는 차원)"로 정확히 쪼개진다.

> **비유**: 함수 $f(x, y) = x + y$는 $\mathbb{R}^2 \to \mathbb{R}$. $\ker f = \{(x, -x)\}$ (1차원), $\text{im } f = \mathbb{R}$ (1차원). 합 = 2 = $\dim \mathbb{R}^2$. ✓

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Kernel (영공간)

선형변환 $T: V \to W$의 **kernel** 또는 **영공간**:

$$\ker T = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}_W\}.$$

$\ker T$는 $V$의 **부분공간**이다 (영벡터 포함, 덧셈·스칼라곱 닫힘 — 선형성에서).

### 정의 4.2 — Image (상공간)

$T$의 **image** 또는 **range**:

$$\text{im } T = T(V) = \{T(\mathbf{v}) : \mathbf{v} \in V\} \subseteq W.$$

이것도 $W$의 부분공간.

### 정의 4.3 — Nullity, Rank

$$\text{nullity}(T) = \dim(\ker T), \quad \text{rank}(T) = \dim(\text{im } T).$$

행렬 $A$의 rank는 $A$가 표현하는 선형변환의 rank, 즉 **열공간의 차원**.

---

## 🔬 정리와 증명

### 정리 4.1 — Rank-Nullity Theorem

**명제**: $V$가 유한차원이고 $T: V \to W$가 선형이면

$$\boxed{\dim(\ker T) + \dim(\text{im } T) = \dim V.}$$

**증명**: $\dim(\ker T) = k$, $\dim V = n$이라 하자.

**Step 1 — kernel의 기저 고정**: $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$를 $\ker T$의 기저로 잡는다.

**Step 2 — 기저 확장 (Ch1-02)**: $\ker T$의 기저를 $V$의 기저로 확장:

$$\{\mathbf{u}_1, \ldots, \mathbf{u}_k, \mathbf{v}_1, \ldots, \mathbf{v}_{n-k}\}.$$

**주장**: $\{T(\mathbf{v}_1), \ldots, T(\mathbf{v}_{n-k})\}$가 **$\text{im } T$의 기저**다. 이를 증명하면 $\text{rank}(T) = n - k$, 따라서 $k + (n-k) = n = \dim V$. 증명 완료.

**Step 3 — 생성**: 임의 $T(\mathbf{v}) \in \text{im } T$에 대해 $\mathbf{v} = \sum \alpha_i \mathbf{u}_i + \sum \beta_j \mathbf{v}_j$. 선형성으로

$$T(\mathbf{v}) = \sum \alpha_i T(\mathbf{u}_i) + \sum \beta_j T(\mathbf{v}_j) = \mathbf{0} + \sum \beta_j T(\mathbf{v}_j).$$

$T(\mathbf{u}_i) = \mathbf{0}$ (kernel)이므로 $\{T(\mathbf{v}_j)\}$가 $\text{im } T$를 생성.

**Step 4 — 선형독립**: $\sum \beta_j T(\mathbf{v}_j) = \mathbf{0}$이라 하자. 선형성으로 $T(\sum \beta_j \mathbf{v}_j) = \mathbf{0}$, 즉 $\sum \beta_j \mathbf{v}_j \in \ker T$. 따라서

$$\sum \beta_j \mathbf{v}_j = \sum \alpha_i \mathbf{u}_i \implies \sum \beta_j \mathbf{v}_j - \sum \alpha_i \mathbf{u}_i = \mathbf{0}.$$

$\{\mathbf{u}_i, \mathbf{v}_j\}$ 전체가 $V$의 기저(선형독립)이므로 모든 $\beta_j = 0$ (및 $\alpha_i = 0$). $\square$

---

### 따름정리 4.2 — 행렬의 Rank-Nullity

$A \in \mathbb{F}^{m \times n}$. 선형변환 $T_A: \mathbb{F}^n \to \mathbb{F}^m$, $\mathbf{x} \mapsto A\mathbf{x}$에 적용하면

$$\text{null}(A) + \text{rank}(A) = n.$$

여기서 $\text{null}(A) = \dim \ker A$, $\text{rank}(A) = \dim \text{Col}(A)$.

---

### 정리 4.3 — 단사·전사 판정

**명제**:
- $T$ 단사 $\iff \ker T = \{\mathbf{0}\}$
- $T$ 전사 $\iff \text{im } T = W$
- $\dim V = \dim W$일 때: $T$ 단사 $\iff$ 전사 $\iff$ 동형

**증명**: 
- 단사: $T(\mathbf{u}) = T(\mathbf{v}) \iff T(\mathbf{u} - \mathbf{v}) = \mathbf{0} \iff \mathbf{u} - \mathbf{v} \in \ker T$. $\ker T = \{\mathbf{0}\}$ 이면 $\mathbf{u} = \mathbf{v}$.
- $\dim V = \dim W = n$: 단사면 $\text{rank} T = n - 0 = n = \dim W$ → 전사. 역도 유사. $\square$

---

### 정리 4.4 — $Ax = b$의 해 구조

**명제**: $A\mathbf{x} = \mathbf{b}$가 해 $\mathbf{x}_p$를 가지면, 모든 해는 **affine subspace**

$$\{\mathbf{x}_p + \mathbf{v} : \mathbf{v} \in \ker A\}.$$

**증명**: $A\mathbf{x} = \mathbf{b}$이면 $A(\mathbf{x} - \mathbf{x}_p) = \mathbf{b} - \mathbf{b} = \mathbf{0}$, 즉 $\mathbf{x} - \mathbf{x}_p \in \ker A$. 역으로 $\mathbf{v} \in \ker A$면 $A(\mathbf{x}_p + \mathbf{v}) = \mathbf{b}$. $\square$

**해의 분류**:
- $\mathbf{b} \notin \text{Col}(A)$ → 해 없음
- $\mathbf{b} \in \text{Col}(A), \ker A = \{\mathbf{0}\}$ → 유일
- $\mathbf{b} \in \text{Col}(A), \ker A \neq \{\mathbf{0}\}$ → 무한해 ($\dim \ker A$ 차원)

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np

# ─────────────────────────────────────────────
# 1. Rank-Nullity 수치 검증
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
m, n, r = 4, 6, 3
# Rank-r 행렬 생성: A = U V^T (U: 4×r, V: 6×r)
U = rng.standard_normal((m, r))
V = rng.standard_normal((n, r))
A = U @ V.T                                   # shape (m, n), rank ≤ r

rank_A = np.linalg.matrix_rank(A)
null_A = n - rank_A
print(f"A shape {A.shape},  rank = {rank_A},  nullity = {null_A}")
assert rank_A + null_A == n
print(f"✓ rank + nullity = {rank_A} + {null_A} = {n} = dim V")

# ─────────────────────────────────────────────
# 2. Kernel 기저 찾기 (SVD 기반)
# ─────────────────────────────────────────────
# Null space는 V의 마지막 (n - rank) 열
Uu, Ss, Vt = np.linalg.svd(A)
null_basis = Vt[rank_A:].T                    # shape (n, n - rank)
print(f"\nNull space 기저 shape: {null_basis.shape}")

# 검증: A × (null basis) = 0
print(f"max|A @ null_basis| = {np.max(np.abs(A @ null_basis)):.2e}")

# ─────────────────────────────────────────────
# 3. Ax = b의 해 구조 (정리 4.4)
# ─────────────────────────────────────────────
b = A @ rng.standard_normal(n)                # b ∈ Col(A) 보장

# 특수해
x_p, *_ = np.linalg.lstsq(A, b, rcond=None)
print(f"\n특수해 Ax_p = b 검증: ‖Ax_p - b‖ = {np.linalg.norm(A @ x_p - b):.2e}")

# 일반해: x_p + Null(A)의 임의 원소
for _ in range(3):
    w = null_basis @ rng.standard_normal(null_A)
    x_general = x_p + w
    residual = np.linalg.norm(A @ x_general - b)
    print(f"일반해 x_p + w 의 잔차: {residual:.2e}")

# ─────────────────────────────────────────────
# 4. Col 공간 기저 vs rank
# ─────────────────────────────────────────────
col_basis = Uu[:, :rank_A]
print(f"\nCol space 기저 shape: {col_basis.shape} = (m, rank)")
```

---

## 🔗 AI/ML 연결

### Hessian의 Flat Directions

신경망 손실 $L(\theta)$의 헤시안 $H = \nabla^2 L$에서 $\ker H$의 방향은 "2차 근사에서 손실이 변하지 않는" 방향. Sagun et al. (2016)는 $H$의 대부분 고유값이 0 근방이라 관찰. **$\dim(\ker H) \gg 0$** 이 과매개변수화의 수학적 표현이다.

### LoRA와 Low-Rank Update

LoRA(Hu et al. 2021)의 $\Delta W = BA$에서 $\text{rank}(\Delta W) \leq r$. Rank-Nullity로 $\Delta W$가 표현하는 변환의 kernel이 $\geq d - r$차원임이 보장. 즉 "$d - r$차원만큼은 변경되지 않음"이 rank 제약의 기하학적 의미.

### Consistency와 Loss Landscape

$A\theta = b$ 형태의 constraint (예: satisfaction-based 최적화)에서 해의 존재는 $b \in \text{Col}(A)$. 이는 loss landscape의 "feasible manifold" 차원을 rank-nullity로 제공.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $V$ 유한차원 | 무한차원에서는 closed range theorem 등 추가 조건 필요 |
| $T$ 선형 | 비선형 맵에서는 "infinitesimal" rank (Jacobian의 rank)로 일반화 |
| Exact rank | 수치 계산에서는 numerical rank (tolerance) 사용 |

**수치적 주의**: $\sigma_r \approx 10^{-10}$이면 rank를 어떻게 셀 것인가? `np.linalg.matrix_rank`의 기본 tolerance는 $\max(m,n) \cdot \|A\|_2 \cdot \varepsilon$. 조건수가 매우 크면 수동 tolerance 권장.

---

## 📌 핵심 정리

$$\boxed{\dim(\ker T) + \dim(\text{im } T) = \dim V}$$

| 양 | 해석 |
|----|------|
| $\text{nullity}(T)$ | "구별 못 하는" 입력 방향 수 |
| $\text{rank}(T)$ | 도달 가능한 출력의 차원 |
| $Ax = b$ 해 공간 | $x_p + \ker A$ (affine) |

---

## 🤔 생각해볼 문제

**문제 1**: $T: \mathbb{R}^4 \to \mathbb{R}^3$, $A = \begin{pmatrix}1 & 2 & 0 & 1\\ 2 & 4 & 1 & 3\\ 0 & 0 & 1 & 1\end{pmatrix}$. rank, nullity, ker의 기저를 구하라.

<details>
<summary>해설</summary>

Row reduce하면 rank = 2 ($R_2 - 2R_1 = R_3$). Nullity = 4 - 2 = 2. Ker의 기저: $(−2, 1, 0, 0)^\top, (−1, 0, −1, 1)^\top$ (RREF로 파라미터화).

</details>

**문제 2** (심화): $T: V \to W$, $S: W \to U$. $\dim \ker(S \circ T) \leq \dim \ker T + \dim \ker S$임을 보여라.

<details>
<summary>해설</summary>

$\mathbf{v} \in \ker(ST)$면 $T(\mathbf{v}) \in \ker S$. 즉 $T|_{\ker(ST)}: \ker(ST) \to \ker S$의 상은 $\ker S$ 안. 이 제한된 사상에 Rank-Nullity: $\dim \ker(ST) = \dim \ker(T|_{\ker(ST)}) + \dim \text{im}(T|_{\ker(ST)}) \leq \dim \ker T + \dim \ker S$. ($T|$의 kernel은 $\ker T \cap \ker(ST) \subseteq \ker T$.)

</details>

**문제 3** (AI 연결): 신경망 $f(x) = W_L \sigma(W_{L-1} \cdots \sigma(W_1 x))$. 활성화를 $\sigma(x) = \max(x, 0)$의 선형 영역에서 Jacobian $J_f = W_L D_{L-1} \cdots W_1$. $W_\ell \in \mathbb{R}^{d \times d}$ 각각 rank $r < d$면 $\text{rank}(J_f) \leq r$임을 보여라. 딥러닝 표현력과의 연결은?

<details>
<summary>해설</summary>

행렬 곱의 rank: $\text{rank}(AB) \leq \min(\text{rank} A, \text{rank} B)$. 따라서 어떤 한 층이 rank $r$이면 전체 Jacobian rank가 $r$로 상한. "병목(bottleneck)" 층이 전체 표현력을 제한하는 이유. Autoencoder의 latent dim이 전체 표현력의 상한이 되는 현상.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 선형변환과 행렬 표현](./03-linear-transformation-matrix.md) | [📚 README](../README.md) | [05. 4개의 기본 부분공간 ▶](./05-four-fundamental-subspaces.md) |

</div>
