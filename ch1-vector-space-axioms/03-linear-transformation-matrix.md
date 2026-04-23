# 03. 선형변환의 정의와 행렬 표현

## 🎯 핵심 질문

- 선형변환 $T: V \to W$가 어떻게 **기저 선택 후 행렬**이 되는가?
- 왜 행렬곱이 선형변환의 **합성**에 대응하는가?
- 좌표계를 바꾸면 같은 $T$가 왜 다른 행렬이 되고, **유사행렬** $P^{-1}AP$의 의미는?
- 신경망의 각 층이 "행렬곱"인 이유는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **신경망의 모든 선형층**: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$는 **선형변환 + 이동(affine)**. 행렬 $W$는 $T: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$의 표현.
- **Attention의 $Q, K, V$ projection**: 각각 선형변환 $\mathbf{x} \mapsto W_Q \mathbf{x}$, $W_K \mathbf{x}$, $W_V \mathbf{x}$.
- **좌표 변환 = 표현 학습**: "representation learning"은 곧 **더 좋은 기저를 찾는 일**. $PCA \cdot autoencoder$는 원 좌표계 $\mathbf{x}$를 새 기저 $P\mathbf{x}$로 바꾼다.
- **Transformer의 head-mixing**: $W_O$는 다중 헤드 결과의 선형 결합으로, 하나의 선형변환.
- **Backpropagation**: 각 층의 Jacobian이 바로 **그 층의 선형변환 행렬**(활성화 선형화)이다 (Ch7-02).

"행렬곱으로 계산한다"는 구현 뒤에는, **추상 선형변환 → 기저 선택 → 행렬**이라는 수학적 계층이 있다.

---

## 📐 수학적 선행 조건

- [Ch1-01 벡터공간의 8공리](./01-vector-space-8-axioms.md)
- [Ch1-02 선형독립·기저·차원](./02-basis-dimension.md)

---

## 📖 직관적 이해

### 선형변환 = "덧셈과 스칼라곱을 보존하는 사상"

$T: V \to W$가 선형이라는 것은, $T$가 "벡터공간 구조를 존중"한다는 뜻. 즉 $T$가 $\mathbf{v}_1 + \mathbf{v}_2$를 $T(\mathbf{v}_1) + T(\mathbf{v}_2)$로 보내고, $\alpha\mathbf{v}$를 $\alpha T(\mathbf{v})$로 보낸다.

### 왜 행렬이 나오는가

기저 $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$이 고정되면, 모든 $\mathbf{v} = \sum x_i \mathbf{e}_i$. 선형성에 의해

$$T(\mathbf{v}) = \sum x_i T(\mathbf{e}_i).$$

즉 **$T$는 기저벡터의 상 $T(\mathbf{e}_i)$만 알면 완전히 결정**된다. 이 $n$개의 상을 $W$의 기저 $\mathcal{C}$로 표현한 계수들을 모은 것이 바로 행렬 $[T]_{\mathcal{C}\mathcal{B}}$.

> **비유**: 선형변환은 "원어 표현(추상 $T$)"이고, 기저 선택은 "번역 방식"이다. 같은 표현도 번역 방식(기저)이 다르면 다른 언어(행렬)로 나오지만, 본질은 같다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 선형변환

$V, W$가 $\mathbb{F}$-벡터공간일 때, 사상 $T: V \to W$가 **선형**이라는 것은

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (가법성)
2. $T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})$ (동차성)

모든 선형변환의 집합을 $\mathcal{L}(V, W)$로 적는다.

### 정의 3.2 — 행렬 표현

$V$의 기저 $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$, $W$의 기저 $\mathcal{C} = \{\mathbf{f}_1, \ldots, \mathbf{f}_m\}$. 각 $T(\mathbf{e}_j)$를 $\mathcal{C}$에 대해 전개:

$$T(\mathbf{e}_j) = \sum_{i=1}^m a_{ij} \mathbf{f}_i.$$

계수 $a_{ij}$를 $(i, j)$-성분으로 가지는 $m \times n$ 행렬을 $[T]_{\mathcal{C}\mathcal{B}}$라 하고, **$T$의 $\mathcal{B}, \mathcal{C}$ 기저에 대한 행렬 표현**이라 한다.

### 정의 3.3 — 기저 변환 행렬

$V$의 두 기저 $\mathcal{B} = \{\mathbf{e}_i\}$, $\mathcal{B}' = \{\mathbf{e}_i'\}$. $\mathbf{e}_j' = \sum_i p_{ij} \mathbf{e}_i$로 전개할 때, 행렬 $P = (p_{ij})$를 **$\mathcal{B}' \to \mathcal{B}$ 좌표 변환 행렬**이라 한다. $P$는 가역이다.

---

## 🔬 정리와 증명

### 정리 3.1 — 선형변환은 기저의 상으로 결정

**명제**: $V$의 기저 $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$에 대해, 어떤 $\mathbf{w}_1, \ldots, \mathbf{w}_n \in W$를 임의로 주면 $T(\mathbf{e}_i) = \mathbf{w}_i$를 만족하는 선형변환 $T: V \to W$가 **유일하게 존재**한다.

**증명**: 임의의 $\mathbf{v} \in V$는 $\mathbf{v} = \sum x_i \mathbf{e}_i$로 유일한 표현(정리 2.1). 

**정의**: $T(\mathbf{v}) := \sum x_i \mathbf{w}_i$.

**선형성**: $T(\mathbf{v} + \mathbf{v}') = \sum(x_i + x_i')\mathbf{w}_i = T(\mathbf{v}) + T(\mathbf{v}')$. 유사하게 $T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})$.

**유일성**: 다른 $T'$도 같은 조건을 만족하면 선형성과 $T'(\mathbf{e}_i) = \mathbf{w}_i$로 $T'(\mathbf{v}) = \sum x_i \mathbf{w}_i = T(\mathbf{v})$. $\square$

---

### 정리 3.2 — $T \leftrightarrow [T]_{\mathcal{C}\mathcal{B}}$는 동형

**명제**: 기저 $\mathcal{B}, \mathcal{C}$를 고정하면, 사상 $\Phi: \mathcal{L}(V, W) \to \mathbb{F}^{m \times n}$, $T \mapsto [T]_{\mathcal{C}\mathcal{B}}$는 **벡터공간 동형**이다.

**증명**: 

**선형**: $[T_1 + T_2]_{\mathcal{C}\mathcal{B}}$의 $j$번째 열은 $(T_1+T_2)(\mathbf{e}_j) = T_1(\mathbf{e}_j) + T_2(\mathbf{e}_j)$의 $\mathcal{C}$-좌표로, 이는 $[T_1]_{\mathcal{C}\mathcal{B}}$의 $j$열과 $[T_2]_{\mathcal{C}\mathcal{B}}$의 $j$열의 합. 따라서 $[T_1 + T_2] = [T_1] + [T_2]$. 유사하게 스칼라곱도 보존.

**단사**: $[T] = 0$이면 모든 $T(\mathbf{e}_j) = 0$이고 선형성으로 $T \equiv 0$.

**전사**: 임의의 $A \in \mathbb{F}^{m \times n}$에 대해 $T(\mathbf{e}_j) := \sum_i a_{ij}\mathbf{f}_i$로 선형변환 정의(정리 3.1). 이 $T$는 $[T] = A$. $\square$

**따름정리**: $\dim \mathcal{L}(V, W) = mn$ (여기서 $n = \dim V$, $m = \dim W$).

---

### 정리 3.3 — 행렬곱 = 선형변환의 합성

**명제**: $T: V \to W$, $S: W \to U$. $V, W, U$의 기저를 $\mathcal{B}, \mathcal{C}, \mathcal{D}$라 하면

$$[S \circ T]_{\mathcal{D}\mathcal{B}} = [S]_{\mathcal{D}\mathcal{C}} \cdot [T]_{\mathcal{C}\mathcal{B}}.$$

**증명**: $T(\mathbf{e}_j) = \sum_k a_{kj} \mathbf{f}_k$, $S(\mathbf{f}_k) = \sum_i b_{ik} \mathbf{g}_i$. 그러면

$$(S \circ T)(\mathbf{e}_j) = S\left(\sum_k a_{kj} \mathbf{f}_k\right) = \sum_k a_{kj} \sum_i b_{ik} \mathbf{g}_i = \sum_i \left(\sum_k b_{ik} a_{kj}\right) \mathbf{g}_i.$$

$(i, j)$-성분이 $\sum_k b_{ik} a_{kj}$, 이는 **행렬곱 $BA$의 $(i, j)$-성분**. $\square$

> **행렬곱의 정의가 "왜 그렇게 되는가"는 이 정리의 증명이 답이다.** 행렬곱은 선형변환 합성의 강제된 결과이지, 임의의 약속이 아니다.

---

### 정리 3.4 — 좌표 변환 공식 (Change of Basis)

**명제**: $T: V \to V$. $V$의 두 기저 $\mathcal{B}, \mathcal{B}'$와 변환 행렬 $P$ ($\mathcal{B}' \to \mathcal{B}$)에 대해

$$[T]_{\mathcal{B}'\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}\mathcal{B}} P.$$

**증명**: 좌표 항등 $[\mathbf{v}]_{\mathcal{B}} = P [\mathbf{v}]_{\mathcal{B}'}$를 이용. $\mathbf{w} = T(\mathbf{v})$라 하면

$$[T(\mathbf{v})]_{\mathcal{B}} = [T]_{\mathcal{B}\mathcal{B}} [\mathbf{v}]_{\mathcal{B}} \implies P [T(\mathbf{v})]_{\mathcal{B}'} = [T]_{\mathcal{B}\mathcal{B}} P [\mathbf{v}]_{\mathcal{B}'}$$

$$\implies [T(\mathbf{v})]_{\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}\mathcal{B}} P [\mathbf{v}]_{\mathcal{B}'},$$

즉 $[T]_{\mathcal{B}'\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}\mathcal{B}} P$. $\square$

**정의 3.4 — 유사행렬**: $A, B \in \mathbb{F}^{n \times n}$이 **유사(similar)** 하다는 것은 $B = P^{-1} A P$인 가역 $P$가 존재. 유사한 행렬은 **같은 선형변환의 다른 기저 표현**.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np

# ─────────────────────────────────────────────
# 1. 선형변환의 행렬 표현
# ─────────────────────────────────────────────
# T: R^2 → R^2, 30도 회전
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 표준기저에서 T(e1), T(e2) 계산
e1, e2 = np.array([1, 0]), np.array([0, 1])
print("T(e1) =", R @ e1)  # (cos θ, sin θ)
print("T(e2) =", R @ e2)  # (-sin θ, cos θ)
# 이 두 열을 모은 것이 R 자체 — 행렬 표현의 정의

# ─────────────────────────────────────────────
# 2. 행렬곱 = 선형변환 합성 (정리 3.3)
# ─────────────────────────────────────────────
# 회전 30° 후 다시 60° = 총 90°
R30 = R
R60 = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],
                [np.sin(np.pi/3),  np.cos(np.pi/3)]])
R90 = np.array([[0, -1], [1, 0]])

assert np.allclose(R60 @ R30, R90)
print("✓ R₆₀ ∘ R₃₀ = R₉₀  (행렬곱 = 합성)")

# ─────────────────────────────────────────────
# 3. 좌표 변환 (정리 3.4)
# ─────────────────────────────────────────────
# 표준기저 vs 45° 회전된 기저
P = np.array([[1,  1],
              [1, -1]]) / np.sqrt(2)   # 새 기저의 열

# T = 회전 30°의 새 기저에서의 표현
T_std = R30
T_new = np.linalg.inv(P) @ T_std @ P
print("\n표준기저에서 T:")
print(T_std)
print("새 기저에서 T = P⁻¹ T P:")
print(T_new)

# 한 벡터를 두 가지 좌표로 변환해도 같은 "추상 벡터"
v_std = np.array([2, 1])
v_new_coord = np.linalg.inv(P) @ v_std

# T를 두 좌표로 적용 후 다시 표준으로 돌려도 같음
Tv_std = T_std @ v_std
Tv_new = T_new @ v_new_coord
assert np.allclose(Tv_std, P @ Tv_new)
print("\n✓ 좌표계에 무관하게 T(v)는 같은 추상 벡터")

# ─────────────────────────────────────────────
# 4. 유사행렬의 불변량: 대각합, 판별식
# ─────────────────────────────────────────────
# 유사행렬은 trace, det, eigenvalues가 같다
A = np.array([[2, 1],
              [0, 3]])
Q = np.array([[1, 2],
              [3, 4]])
B = np.linalg.inv(Q) @ A @ Q

print(f"\ntr(A) = {np.trace(A):.4f}, tr(B) = {np.trace(B):.4f}")
print(f"det(A) = {np.linalg.det(A):.4f}, det(B) = {np.linalg.det(B):.4f}")
eigs_A = np.sort(np.linalg.eigvals(A))
eigs_B = np.sort(np.linalg.eigvals(B))
print(f"eig(A) = {eigs_A},  eig(B) = {eigs_B}")
assert np.allclose(eigs_A, eigs_B)
print("✓ 유사행렬은 trace·det·eigenvalue 동일")
```

---

## 🔗 AI/ML 연결

### 신경망 층 = 선형변환 + 비선형

$\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$에서 $W$는 선형변환 $T$의 기저 표현. "은닉 차원 256"은 $W \in \mathbb{R}^{256 \times d_{\text{in}}}$이라는 뜻이고, 정리 3.2에 의해 $\dim \mathcal{L}(\mathbb{R}^{d_{\text{in}}}, \mathbb{R}^{256}) = 256 d_{\text{in}}$개의 자유도가 있다.

### Representation Learning = 좌표 변환 학습

Autoencoder의 encoder $f: \mathbb{R}^d \to \mathbb{R}^k$는 "더 좋은 기저를 찾는다". Linear autoencoder (tied weights)의 최적해는 **PCA**로 환원된다(Baldi & Hornik 1989). 이는 **기저 변환 $P$를 직접 학습**하는 것.

### Transformer의 Weight Matrices

Multi-head attention의 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$와 $W_O \in \mathbb{R}^{hd_k \times d}$는 모두 선형변환. 합성 $W_O (\text{Attn}) W_V$에서 행렬곱이 반복되는데, 각 곱은 **변환의 합성**이라는 정리 3.3의 응용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $V, W$ 유한차원 | 무한차원에서는 연산자(operator)로 일반화, 연속성 조건 필요 |
| 기저 고정 | 기저 변경 시 행렬이 바뀜 — 좌표-자유 관점(텐서)이 더 근본적 (Ch6) |
| Affine이 아닌 순수 선형 | 편향 $\mathbf{b}$는 선형변환 아님. 동차좌표 $[\mathbf{x}, 1]$로 흡수 가능 |

---

## 📌 핵심 정리

$$\boxed{\;T \in \mathcal{L}(V, W),\; \mathcal{B}, \mathcal{C}\text{ 기저} \implies [T]_{\mathcal{C}\mathcal{B}} \in \mathbb{F}^{m \times n}\text{ 유일}\;}$$

$$\boxed{\;[S \circ T]_{\mathcal{D}\mathcal{B}} = [S]_{\mathcal{D}\mathcal{C}} [T]_{\mathcal{C}\mathcal{B}}, \quad [T]_{\mathcal{B}'\mathcal{B}'} = P^{-1}[T]_{\mathcal{B}\mathcal{B}} P\;}$$

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T: \mathbb{R}^2 \to \mathbb{R}^2$, $T(x, y) = (2x + y, x - 3y)$의 표준기저에 대한 행렬을 구하라. 또한 $\{\mathbf{v}_1, \mathbf{v}_2\} = \{(1,1), (1,-1)\}$ 기저에 대한 행렬은?

<details>
<summary>힌트 및 해설</summary>

표준: $T(\mathbf{e}_1) = (2, 1)$, $T(\mathbf{e}_2) = (1, -3)$ → $A = \begin{pmatrix}2 & 1\\ 1 & -3\end{pmatrix}$. 새 기저 행렬 $P = \begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$. $A' = P^{-1} A P = \frac{1}{2}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}\begin{pmatrix}2 & 1\\ 1 & -3\end{pmatrix}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix} = \begin{pmatrix}1/2 & 7/2\\ 5/2 & -3/2\end{pmatrix}$ (계산 확인).

</details>

**문제 2** (심화): 미분 연산자 $D: \mathbb{R}[x]_{\leq 3} \to \mathbb{R}[x]_{\leq 3}$, $D(p) = p'$. 기저 $\{1, x, x^2, x^3\}$에 대한 행렬을 구하고, $D^4 = 0$임을 보여라.

<details>
<summary>힌트 및 해설</summary>

$D(1) = 0$, $D(x) = 1$, $D(x^2) = 2x$, $D(x^3) = 3x^2$. 행렬 $[D] = \begin{pmatrix}0 & 1 & 0 & 0\\ 0 & 0 & 2 & 0\\ 0 & 0 & 0 & 3\\ 0 & 0 & 0 & 0\end{pmatrix}$. 상삼각이고 대각이 0이므로 **nilpotent**, $[D]^4 = 0$. 이는 4차 다항식을 4번 미분하면 0이 되는 사실에 대응. 이런 연산자는 고유값이 모두 0이고 대각화 불가능 — **Jordan Form**(Ch2-06)의 모범 예제.

</details>

**문제 3** (AI 연결): Attention의 $W_Q \in \mathbb{R}^{d \times d_k}$가 선형변환 $T_Q: \mathbb{R}^d \to \mathbb{R}^{d_k}$의 표현이라 하자. $d = 512, d_k = 64, h = 8$일 때 모든 헤드의 $W_Q^{(h)}$의 파라미터 수를 구하고, 이를 하나의 큰 변환 $\mathbb{R}^d \to \mathbb{R}^{hd_k}$로 합쳤을 때의 해석을 논하라.

<details>
<summary>힌트 및 해설</summary>

헤드 하나당 $512 \times 64 = 32768$ 파라미터, 8 헤드 총 $262144$. 이는 $\mathbb{R}^{512} \to \mathbb{R}^{512}$ ($8 \times 64 = 512$) 변환 하나의 파라미터 수와 같다 ($512 \times 512 = 262144$). 구현상 **$W_Q$를 $\mathbb{R}^{512 \times 512}$ 하나로 두고 결과를 reshape하여 헤드로 분리**하는 것이 일반적이며, 이는 "큰 선형변환 → 여러 부분공간으로 projection"이라는 관점이다. 각 헤드가 독립적으로 다른 부분공간을 탐색한다는 직관은 여기에서 온다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 선형독립·기저·차원](./02-basis-dimension.md) | [📚 README](../README.md) | [04. Rank-Nullity 정리 ▶](./04-rank-nullity.md) |

</div>
