# 05. 4개의 기본 부분공간

## 🎯 핵심 질문

- 왜 **Column·Row·Null·Left-Null**이 "4개 기본 부분공간"인가?
- 이 넷이 쌍으로 **직교 관계**를 이루는 이유는?
- $\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A)$, $\mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^\top)$ 분해의 기하학적 의미는?
- 이 구조가 **최소제곱·SVD·저랭크 근사**로 어떻게 이어지는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Least Squares의 기하**: $\min \|A\mathbf{x} - \mathbf{b}\|^2$의 해는 $\mathbf{b}$를 $\text{Col}(A)$로 **직교투영**하는 것. 잔차 $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}} \in \text{Null}(A^\top)$ (Ch5-03).
- **방정식의 가해성**: $A\mathbf{x} = \mathbf{b}$가 해를 가짐 $\iff \mathbf{b} \perp \text{Null}(A^\top)$.
- **SVD의 U, V 분해**: SVD의 왼쪽/오른쪽 특이벡터가 정확히 이 4개 공간의 **정규직교 기저**(Ch4-02).
- **Regularization**: Ridge regression은 $\text{Null}(A)$ 방향으로의 해의 자유도를 제어하고, Lasso는 sparse한 row space 원소를 선호.
- **Normal Equations**: $A^\top A\mathbf{x} = A^\top \mathbf{b}$에서 $A^\top \mathbf{b}$는 $\mathbf{b}$의 row space 성분. 이 구조가 **pseudoinverse**(Ch4-03)로 이어진다.

---

## 📐 수학적 선행 조건

- [Ch1-03 선형변환과 행렬 표현](./03-linear-transformation-matrix.md)
- [Ch1-04 Rank-Nullity](./04-rank-nullity.md)
- 내적 (잠정 — 이 문서에서는 $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$로 제한적 사용, 엄밀한 내적공간은 Ch5)

---

## 📖 직관적 이해

### 네 공간의 큰 그림

$A \in \mathbb{R}^{m \times n}$이 두 유클리드 공간을 잇는다:

```
        A
   ℝⁿ ─────► ℝᵐ
   
   Row(A) ─┐      ┌─ Col(A)
           │  A   │
   Null(A) ┘      └─ Null(Aᵀ)
```

- $\mathbb{R}^n$(정의역)은 $\text{Row}(A)$와 $\text{Null}(A)$로 **직교 쪼개짐**
- $\mathbb{R}^m$(공역)은 $\text{Col}(A)$와 $\text{Null}(A^\top)$로 **직교 쪼개짐**
- $A$는 $\text{Row}(A)$를 $\text{Col}(A)$로 **동형**으로 보냄 (rank $r$)
- $\text{Null}(A)$는 $\mathbf{0}$으로, $\text{Null}(A^\top)$는 도달 불가

### Strang의 "Big Picture"

$\text{Null}(A)$와 $\text{Row}(A)$의 **직교**는 내적의 정의에서 즉시:

$$\mathbf{x} \in \text{Null}(A) \iff A\mathbf{x} = \mathbf{0} \iff \text{모든 } A\text{의 행이 } \mathbf{x}\text{와 수직}.$$

이것이 "왜 row space와 null space가 직교하는가"의 본질이다.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — 4개의 기본 부분공간

$A \in \mathbb{R}^{m \times n}$에 대해:

| 공간 | 정의 | 어디에 사나 | 차원 |
|------|------|-------------|------|
| **Column space** $\text{Col}(A)$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $\mathbb{R}^m$ | $r$ |
| **Row space** $\text{Row}(A)$ | $\{A^\top \mathbf{y} : \mathbf{y} \in \mathbb{R}^m\} = \text{Col}(A^\top)$ | $\mathbb{R}^n$ | $r$ |
| **Null space** $\text{Null}(A)$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $\mathbb{R}^n$ | $n - r$ |
| **Left null space** $\text{Null}(A^\top)$ | $\{\mathbf{y} : A^\top \mathbf{y} = \mathbf{0}\}$ | $\mathbb{R}^m$ | $m - r$ |

여기서 $r = \text{rank}(A)$.

### 정의 5.2 — 직교보공간

부분공간 $W \subseteq \mathbb{R}^n$의 **직교보공간**:

$$W^\perp = \{\mathbf{v} \in \mathbb{R}^n : \mathbf{v} \cdot \mathbf{w} = 0, \forall \mathbf{w} \in W\}.$$

---

## 🔬 정리와 증명

### 정리 5.1 — Row-Null 직교

**명제**: $\text{Row}(A)^\perp = \text{Null}(A)$ (in $\mathbb{R}^n$). 동치로 $\text{Row}(A) \perp \text{Null}(A)$.

**증명**: 
$(\subseteq)$: $\mathbf{x} \in \text{Row}(A)^\perp$ ⟺ $\mathbf{x}$가 $A$의 모든 행과 수직 ⟺ $A\mathbf{x} = \mathbf{0}$ ⟺ $\mathbf{x} \in \text{Null}(A)$.
$(\supseteq)$: 역방향도 같은 논리. $\square$

---

### 정리 5.2 — Col-LeftNull 직교

**명제**: $\text{Col}(A)^\perp = \text{Null}(A^\top)$ (in $\mathbb{R}^m$).

**증명**: $\mathbf{y} \in \text{Col}(A)^\perp$ ⟺ $\mathbf{y} \cdot A\mathbf{x} = 0$ $\forall \mathbf{x}$ ⟺ $(A^\top \mathbf{y}) \cdot \mathbf{x} = 0$ $\forall \mathbf{x}$ ⟺ $A^\top \mathbf{y} = \mathbf{0}$ ⟺ $\mathbf{y} \in \text{Null}(A^\top)$. $\square$

---

### 정리 5.3 — 직교 분해

**명제**: 
$$\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A), \quad \mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^\top).$$
($\oplus$는 직교 직합)

**증명 (Lemma)**: 유한차원 $V$의 부분공간 $W$에 대해 $V = W \oplus W^\perp$. 증명: $W$의 정규직교기저 $\{\mathbf{w}_i\}$ 확장 + Gram-Schmidt로 $W^\perp$ 기저 구성 → 전체가 $V$ 생성. 차원: $\dim W + \dim W^\perp = \dim V$.

$\text{Row}(A) \oplus \text{Row}(A)^\perp = \mathbb{R}^n$이고 정리 5.1로 $\text{Row}(A)^\perp = \text{Null}(A)$. 따라서 $\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A)$. 두 번째도 같은 논리. $\square$

---

### 정리 5.4 — Row rank = Column rank

**명제**: $\dim \text{Row}(A) = \dim \text{Col}(A)$.

**증명**: Row-reduction은 행공간을 보존 ($A$와 $\text{rref}(A)$의 row space 동일). RREF의 row rank = pivot 개수 = column rank. $\square$

이것이 **"rank"가 유일한 개념**인 이유. $\text{rank}(A) = \text{rank}(A^\top)$.

---

### 정리 5.5 — 차원의 큰 그림 (Strang의 Big Picture)

$$\dim \text{Col}(A) + \dim \text{Null}(A^\top) = m$$
$$\dim \text{Row}(A) + \dim \text{Null}(A) = n$$
$$\dim \text{Col}(A) = \dim \text{Row}(A) = r$$

**증명**: 첫 둘은 정리 5.3 + 직합 차원. 셋째는 정리 5.4. 이들을 결합하면 Rank-Nullity (Ch1-04)의 재확인. $\square$

---

### 정리 5.6 — $A$는 Row(A)→Col(A)의 동형

**명제**: $A$를 $\text{Row}(A)$로 제한한 $A|_{\text{Row}(A)}: \text{Row}(A) \to \text{Col}(A)$는 **동형사상**이다.

**증명**: 
- **단사**: $\mathbf{x} \in \text{Row}(A) \cap \text{Null}(A) = \{\mathbf{0}\}$ (직교이고 $\mathbf{0}$ 외 공통 원소 없음). 따라서 $A|_{\text{Row}(A)}$의 kernel은 $\{\mathbf{0}\}$.
- **전사**: $\forall \mathbf{b} \in \text{Col}(A)$, $\exists \mathbf{x} \in \mathbb{R}^n$ s.t. $A\mathbf{x} = \mathbf{b}$. $\mathbf{x} = \mathbf{x}_r + \mathbf{x}_n$ ($\mathbf{x}_r \in \text{Row}(A), \mathbf{x}_n \in \text{Null}(A)$). $A\mathbf{x}_n = \mathbf{0}$이므로 $A\mathbf{x}_r = \mathbf{b}$. $\square$

> 이 정리가 **pseudoinverse**(Ch4-03)의 기하학적 정당성이다. $A^+$는 이 "$\text{Col}(A) \to \text{Row}(A)$" 동형의 역사상을 확장한 것.

---

## 💻 NumPy 검증

```python
import numpy as np

rng = np.random.default_rng(0)
m, n, r = 5, 7, 3

# Rank-r 행렬 만들기
U = rng.standard_normal((m, r))
V = rng.standard_normal((n, r))
A = U @ V.T

# SVD로 네 부분공간의 정규직교 기저 추출
Um, S, Vt = np.linalg.svd(A, full_matrices=True)
rank = np.sum(S > 1e-10)
print(f"rank(A) = {rank}")

col_basis      = Um[:, :rank]              # Col(A)
leftnull_basis = Um[:, rank:]              # Null(Aᵀ)
row_basis      = Vt[:rank, :].T            # Row(A)
null_basis     = Vt[rank:, :].T            # Null(A)

# ─────────────────────────────────────────────
# 직교 관계 검증
# ─────────────────────────────────────────────
# Row(A) ⊥ Null(A)
print(f"\nmax|Row ⋅ Null|     = {np.max(np.abs(row_basis.T @ null_basis)):.2e}")
# Col(A) ⊥ Null(Aᵀ)
print(f"max|Col ⋅ LeftNull| = {np.max(np.abs(col_basis.T @ leftnull_basis)):.2e}")

# ─────────────────────────────────────────────
# 기저로 원래 공간을 직합으로 덮는지
# ─────────────────────────────────────────────
# ℝⁿ = Row ⊕ Null
B_n = np.hstack([row_basis, null_basis])
assert np.allclose(B_n @ B_n.T, np.eye(n))
print(f"✓ Row ⊕ Null = ℝⁿ  (정규직교 합집합으로 ℝ^{n} 완성)")

# ℝᵐ = Col ⊕ LeftNull
B_m = np.hstack([col_basis, leftnull_basis])
assert np.allclose(B_m @ B_m.T, np.eye(m))
print(f"✓ Col ⊕ LeftNull = ℝᵐ")

# ─────────────────────────────────────────────
# 차원 공식 (정리 5.5)
# ─────────────────────────────────────────────
print(f"\ndim Col      = {col_basis.shape[1]}")
print(f"dim Null(Aᵀ) = {leftnull_basis.shape[1]}")
print(f"dim Row      = {row_basis.shape[1]}")
print(f"dim Null     = {null_basis.shape[1]}")
print(f"합: Col + LeftNull = {rank + (m - rank)} = m = {m}")
print(f"합: Row + Null     = {rank + (n - rank)} = n = {n}")

# ─────────────────────────────────────────────
# Ax = b의 가해성: b ⊥ LeftNull이어야 (정리 5.2)
# ─────────────────────────────────────────────
b_good = A @ rng.standard_normal(n)                  # Col(A) 원소
b_bad  = leftnull_basis[:, 0]                        # LeftNull 원소

res_good = np.linalg.lstsq(A, b_good, rcond=None)[1] if False else np.linalg.norm(A @ np.linalg.lstsq(A, b_good, rcond=None)[0] - b_good)
res_bad  = np.linalg.norm(A @ np.linalg.lstsq(A, b_bad,  rcond=None)[0] - b_bad)
print(f"\nb ∈ Col(A)인 경우 residual: {res_good:.2e}")
print(f"b ∈ LeftNull인 경우 residual: {res_bad:.2e}   (→ 해 존재 불가)")
```

---

## 🔗 AI/ML 연결

### Least Squares = Col(A)로의 투영

$\min \|A\mathbf{x} - \mathbf{b}\|^2$의 해석: $\mathbf{b} = \mathbf{b}_{\text{Col}} + \mathbf{b}_{\text{LN}}$ 직교 분해로 (정리 5.3) 잔차는 최소가 $\|\mathbf{b}_{\text{LN}}\|$, 이때 $A\hat{\mathbf{x}} = \mathbf{b}_{\text{Col}}$. 이는 Ch5-03에서 엄밀하게.

### Consistency 판정

$A\mathbf{x} = \mathbf{b}$가 해를 가짐 $\iff \mathbf{b} \in \text{Col}(A) \iff \mathbf{b} \perp \text{Null}(A^\top)$. 이는 Fredholm alternative의 유한차원 판이다.

### PCA: Row Space의 주축

데이터 행렬 $X \in \mathbb{R}^{n \times d}$ (샘플 $n$개, 특징 $d$개). $\text{Row}(X)$는 $\mathbb{R}^d$의 부분공간 (샘플들이 놓이는 affine subspace의 평행이동). PCA의 주성분은 이 row space의 **주축**.

### Transformer의 Output Projection

Attention의 최종 $W_O$는 head-concat된 값을 원래 차원으로 되돌린다. 이 변환의 $\text{Col}(W_O)$이 모델이 실제 사용하는 "output subspace"이고, $\text{Null}(W_O^\top)$은 무시되는 방향.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 표준 내적 | 다른 내적 사용 시 $W^\perp$가 달라짐 (weighted least squares) |
| 실수체 | 복소행렬은 $A^\top \to A^*$ (Hermitian adjoint)로 대체 |
| Exact rank | 노이즈 환경에서 numerical rank tolerance 필요 |

---

## 📌 핵심 정리

$$\boxed{\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A), \quad \mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^\top)}$$

$$\boxed{\dim \text{Col}(A) = \dim \text{Row}(A) = r, \quad \dim \text{Null}(A) = n - r, \quad \dim \text{Null}(A^\top) = m - r}$$

---

## 🤔 생각해볼 문제

**문제 1**: $A = \begin{pmatrix}1 & 2\\ 2 & 4\\ 3 & 6\end{pmatrix}$. 네 부분공간의 기저와 차원을 구하라.

<details>
<summary>해설</summary>

Rank = 1 (두 번째 열은 첫 번째의 2배). $\text{Col}(A) = \text{span}\{(1,2,3)^\top\}$ (1차원). $\text{Null}(A^\top) = \text{span}\{(2,-1,0)^\top, (3,0,-1)^\top\}$ (2차원). $\text{Row}(A) = \text{span}\{(1,2)\}$ (1차원). $\text{Null}(A) = \text{span}\{(2,-1)^\top\}$ (1차원). 합: 1+2=3, 1+1=2. ✓

</details>

**문제 2** (심화): $A^\top A$와 $A$의 null space가 같음을 증명하라.

<details>
<summary>해설</summary>

$(\supseteq)$: $A\mathbf{x} = 0 \Rightarrow A^\top A \mathbf{x} = 0$. $(\subseteq)$: $A^\top A \mathbf{x} = 0 \Rightarrow \mathbf{x}^\top A^\top A \mathbf{x} = \|A\mathbf{x}\|^2 = 0 \Rightarrow A\mathbf{x} = 0$. 이 사실이 Normal Equations와 Ridge의 성립의 기반.

</details>

**문제 3** (AI 연결): LoRA $\Delta W = BA$에서 $\text{Col}(\Delta W) \subseteq \text{Col}(B)$이므로 rank $\leq r$. 이것이 "$r$차원 subspace에서만 업데이트"임을 기저 언어로 설명하라.

<details>
<summary>해설</summary>

$\text{Col}(B)$는 출력공간 $\mathbb{R}^d$의 $r$차원 부분공간. LoRA 업데이트는 이 부분공간을 벗어나는 방향으로는 전혀 이동하지 않음. Pretrained 모델의 "기존 표현 방향"을 보존하면서 특정 subspace에서만 미세조정.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Rank-Nullity 정리](./04-rank-nullity.md) | [📚 README](../README.md) | [06. 이중공간(Dual Space) ▶](./06-dual-space.md) |

</div>
