# 7.2 역전파와 Vector-Jacobian Product

> "역전파는 'Jacobian을 구성하지 않고 그 전치를 왼쪽에서 곱하는' 선형대수 요법이다."

---

## 1. 학습 목표

- **체인 룰**을 벡터·텐서 형태로 엄밀히 쓴다.
- **Vector-Jacobian Product (VJP)** 와 **Jacobian-Vector Product (JVP)** 의 차이를 이해한다.
- 리버스 모드 자동미분이 왜 **파라미터 수 × 1** 시간에 $\nabla L$을 계산할 수 있는지 증명한다.
- 주요 연산(행렬곱, softmax, BatchNorm, Attention)의 VJP를 직접 유도한다.
- **메모리 vs 계산** 트레이드오프와 **gradient checkpointing**의 수학을 정리한다.

---

## 2. 체인 룰 복습

### 2.1 스칼라 체인 룰

합성함수 $y = f(g(x))$의 미분은

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

### 2.2 벡터 체인 룰

$f: \mathbb{R}^n \to \mathbb{R}^m$, $g: \mathbb{R}^m \to \mathbb{R}^p$, 합성 $h = g \circ f: \mathbb{R}^n \to \mathbb{R}^p$. Jacobian:

$$
J_h(x) = J_g(f(x)) \cdot J_f(x) \quad (\text{행렬곱})
$$

$J_f \in \mathbb{R}^{m \times n}$, $J_g \in \mathbb{R}^{p \times m}$, $J_h \in \mathbb{R}^{p \times n}$.

### 2.3 텐서 체인 룰

입력과 출력이 텐서 $X \in \mathbb{R}^{I_1 \times \cdots \times I_N}$, $Y \in \mathbb{R}^{J_1 \times \cdots \times J_M}$인 함수의 Jacobian은 $(M + N)$-차원 텐서:

$$
\frac{\partial Y_{j_1 \cdots j_M}}{\partial X_{i_1 \cdots i_N}}
$$

이를 전개한 행렬은 $\prod J_k \times \prod I_l$ 크기.

---

## 3. Forward Mode vs Reverse Mode

### 3.1 Forward Mode (JVP)

"입력 방향"으로 미분 전파. $\dot x \in \mathbb{R}^n$ (input tangent), 각 연산에 대해

$$
\dot y = J_f(x) \dot x
$$

계산. 최종 $\dot L = J \dot x$ (한 입력 방향에 대한 그래디언트 성분).

**복잡도**: 한 번의 forward pass와 비슷, $O(\text{forward time})$.

**장점**: 여러 출력에 대한 하나의 입력 방향 미분 (예: JVP for ODE tangent).

**단점**: 입력 수만큼 반복해야 전체 Jacobian.

### 3.2 Reverse Mode (VJP)

"출력 방향"으로 미분 전파. $\bar y \in \mathbb{R}^m$ (output cotangent), 각 연산에 대해

$$
\bar x = J_f(x)^T \bar y
$$

계산. 최종 $\bar x = J^T \bar y$.

**복잡도**: 한 번의 forward + backward, $O(\text{forward time})$.

**장점**: **하나의 출력(스칼라 loss)에 대한 모든 파라미터 그래디언트**를 한 번에.

**단점**: 중간 활성값을 저장해야 함 (메모리 부담).

### 3.3 언제 어느 것을?

- **딥러닝**: 출력 1개 (loss), 입력 $10^{11}$개 (파라미터) → **리버스 모드**.
- **과학 계산**: Jacobian이 시스템 자체 (ODE, PDE) → forward mode.
- **Mixed**: 고차 미분에 포워드+리버스 조합.

---

## 4. 연산별 VJP 공식

### 4.1 행렬곱 $Y = XW$

$X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, $Y \in \mathbb{R}^{B \times D_{\text{out}}}$.

$\bar Y$가 주어졌을 때:

$$
\bar X = \bar Y W^T, \quad \bar W = X^T \bar Y
$$

**유도.**

$Y_{ij} = \sum_k X_{ik} W_{kj}$.

$\bar X_{ik} = \sum_{j} \bar Y_{ij} \frac{\partial Y_{ij}}{\partial X_{ik}} = \sum_{j} \bar Y_{ij} W_{kj} = (\bar Y W^T)_{ik}$.

$\bar W_{kj} = \sum_{i} \bar Y_{ij} X_{ik} = (X^T \bar Y)_{kj}$. ∎

### 4.2 합 $y = \sum x_i$

$\bar x_i = \bar y$ (단순 broadcasting).

### 4.3 원소별 함수 $y_i = \sigma(x_i)$

$\bar x_i = \bar y_i \cdot \sigma'(x_i)$

- ReLU: $\sigma'(x) = \mathbb{1}[x > 0]$
- Sigmoid: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Tanh: $\sigma'(x) = 1 - \tanh^2(x)$

### 4.4 Softmax $y = \text{softmax}(x)$

**유도.** $y_i = \exp(x_i) / \sum_k \exp(x_k)$.

$$
\frac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)
$$

VJP:

$$
\bar x_j = \sum_i \bar y_i \cdot y_i(\delta_{ij} - y_j) = y_j \bar y_j - y_j \sum_i \bar y_i y_i = y_j (\bar y_j - \langle \bar y, y \rangle)
$$

### 4.5 Cross-entropy Loss

$L = -\sum_i t_i \log y_i$ ($t$: one-hot 타겟, $y$: softmax 확률).

$$
\bar y_i = -t_i / y_i
$$

**Softmax + cross-entropy 결합**: 많은 최적화를 포함. $y$를 계산하지 않고 직접

$$
\bar x = y - t
$$

로 합치는 게 수치적·계산적으로 우수.

### 4.6 LayerNorm

$y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$ ($\mu, \sigma$는 $x$에 의존).

VJP는 다소 복잡하지만 표준 공식:

$$
\bar x_i = \frac{1}{\sigma N}\left[N \bar y_i - \sum_j \bar y_j - \hat x_i \sum_j \bar y_j \hat x_j\right] \cdot \gamma
$$

여기서 $\hat x = (x - \mu)/\sigma$.

---

## 5. Attention의 VJP

### 5.1 Forward 재정리

$$
S = QK^T / \sqrt{d_k}, \quad A = \text{softmax}(S), \quad O = AV
$$

### 5.2 역방향 유도

$\bar O$가 주어지면:

1. $\bar V = A^T \bar O$
2. $\bar A = \bar O V^T$
3. $\bar S = \text{softmax}'(S) \cdot \bar A$ (4.4의 공식, 각 행 독립)
4. $\bar Q = \bar S K / \sqrt{d_k}$
5. $\bar K = \bar S^T Q / \sqrt{d_k}$

### 5.3 메모리 소요

$A \in \mathbb{R}^{S \times S}$를 저장해야 함 → $O(S^2)$ 메모리. 긴 시퀀스에서 병목.

### 5.4 FlashAttention의 해결

출력을 블록별로 계산하며 $A$를 **명시적으로 저장하지 않음**. 수학적으로는 동일하지만 메모리 접근 패턴을 재구성하여 HBM 트래픽을 줄이고 속도도 향상.

---

## 6. 계산 그래프와 토폴로지적 순서

### 6.1 DAG 구조

각 연산을 노드로, 데이터 의존을 간선으로 하는 DAG.

### 6.2 Forward: 토폴로지적 정렬순

입력에서 출력으로.

### 6.3 Backward: 역순

그래디언트를 출력에서 입력으로. 합성 연산이 **한 노드에서 여러 곳으로 퍼져나갔으면(fan-out)** 해당 노드의 그래디언트는 **모든 경로에서 들어오는 것의 합**.

### 6.4 예: $y = f(x) + g(x)$

$\frac{dy}{dx} = f'(x) + g'(x)$. $\bar x = \bar y \cdot f'(x) + \bar y \cdot g'(x)$. 합산 필요.

---

## 7. 메모리 관리

### 7.1 전체 활성값 저장 (Naive)

모든 중간 텐서를 저장. 메모리 = $O(L \times B \times d)$. 큰 모델에서 한계.

### 7.2 Gradient Checkpointing

일부 노드만 저장하고 나머지는 backward 때 **재계산**. $L$개 레이어를 $\sqrt L$개 그룹으로 나누면:

- 저장: $O(\sqrt L \cdot B \cdot d)$
- 재계산: $O(L \cdot B \cdot d)$ (backward에 추가)

메모리와 속도의 **파레토 효율**.

### 7.3 In-place 연산

ReLU처럼 원소별 연산은 입력을 덮어써도 backward에 문제없는 경우가 많다 (출력만 필요).

### 7.4 Activation Checkpointing in Transformers

Transformer 블록 (self-attn + FFN) 단위로 체크포인팅. Megatron, DeepSpeed 등에서 활용.

---

## 8. 자동미분 구현 원리

### 8.1 연산 테이프(Tape) 기반

- Forward: 각 연산을 테이프에 기록 (입력, 출력, 함수)
- Backward: 테이프를 역순으로 읽으며 VJP 적용

PyTorch의 `autograd`가 이 방식. TensorFlow eager도 마찬가지.

### 8.2 소스 변환 기반

컴파일러가 소스 코드를 파싱해서 backward 코드 생성. JAX의 `jit` 이후에 해당.

### 8.3 Dual Numbers (Forward mode)

$(x, \dot x)$ 쌍으로 숫자를 표현하고 연산 규칙

$$
(x, \dot x) + (y, \dot y) = (x+y, \dot x + \dot y)
$$
$$
(x, \dot x)(y, \dot y) = (xy, \dot x y + x \dot y)
$$

등을 정의. 함수를 그대로 호출하면 derivative가 자동 계산.

---

## 9. 정밀도, 수치 안정성

### 9.1 FP16의 underflow

작은 그래디언트가 FP16에서 0이 됨. **Loss scaling**:

$$
L_{\text{scaled}} = s \cdot L, \quad \nabla L_{\text{scaled}} = s \nabla L
$$

$s$를 크게 (예: $2^{16}$) 해서 backward 중 그래디언트가 representable 범위 내로. Update 시 $s$로 나눔.

### 9.2 Gradient Clipping

$\|\nabla L\| > c$면

$$
\nabla L \leftarrow \nabla L \cdot c / \|\nabla L\|
$$

학습 안정성을 위해 (특히 RNN, Transformer).

### 9.3 Mixed Precision

Forward/backward는 FP16, 파라미터·옵티마이저 상태는 FP32. NVIDIA AMP의 표준.

---

## 10. Python 실험

### 10.1 간단한 자동미분 엔진

```python
import numpy as np

class Node:
    def __init__(self, value, parents=(), grad_fn=None):
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self.grad_fn = grad_fn
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad
        if self.grad_fn is not None:
            self.grad_fn(grad)

def add(a, b):
    out = Node(a.value + b.value, parents=(a, b))
    def _back(g):
        a.backward(g)
        b.backward(g)
    out.grad_fn = _back
    return out

def mul(a, b):
    out = Node(a.value * b.value, parents=(a, b))
    def _back(g):
        a.backward(g * b.value)
        b.backward(g * a.value)
    out.grad_fn = _back
    return out

def matmul(A, B):
    out = Node(A.value @ B.value, parents=(A, B))
    def _back(g):
        A.backward(g @ B.value.T)
        B.backward(A.value.T @ g)
    out.grad_fn = _back
    return out

def relu(a):
    out = Node(np.maximum(0, a.value), parents=(a,))
    mask = (a.value > 0).astype(float)
    def _back(g):
        a.backward(g * mask)
    out.grad_fn = _back
    return out

def sum_all(a):
    out = Node(a.value.sum(), parents=(a,))
    def _back(g):
        a.backward(np.ones_like(a.value) * g)
    out.grad_fn = _back
    return out

# 테스트: 간단한 MLP
np.random.seed(0)
X = Node(np.random.randn(5, 3))
W1 = Node(np.random.randn(3, 4))
W2 = Node(np.random.randn(4, 1))

h = relu(matmul(X, W1))
y = matmul(h, W2)
loss = sum_all(y)
loss.backward()

print(f"W1 grad shape: {W1.grad.shape}")
print(f"W2 grad shape: {W2.grad.shape}")

# 수치 미분과 비교
eps = 1e-6
W1_numerical = np.zeros_like(W1.value)
for i in range(W1.value.shape[0]):
    for j in range(W1.value.shape[1]):
        orig = W1.value[i, j]
        W1.value[i, j] = orig + eps
        h_plus = np.maximum(0, X.value @ W1.value)
        loss_plus = (h_plus @ W2.value).sum()
        W1.value[i, j] = orig - eps
        h_minus = np.maximum(0, X.value @ W1.value)
        loss_minus = (h_minus @ W2.value).sum()
        W1.value[i, j] = orig
        W1_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

print(f"W1 그래디언트 오차: {np.max(np.abs(W1.grad - W1_numerical)):.2e}")
```

### 10.2 Softmax의 VJP

```python
import numpy as np

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def softmax_vjp(y, grad_y):
    """y = softmax(x), grad_y 주어졌을 때 grad_x"""
    return y * (grad_y - (grad_y * y).sum())

np.random.seed(0)
x = np.random.randn(5)
y = softmax(x)
grad_y = np.random.randn(5)

# Analytical VJP
grad_x_vjp = softmax_vjp(y, grad_y)

# Numerical
eps = 1e-6
grad_x_num = np.zeros_like(x)
for i in range(len(x)):
    xp = x.copy(); xp[i] += eps
    xm = x.copy(); xm[i] -= eps
    # scalar: <grad_y, softmax(x)>
    fp = (grad_y * softmax(xp)).sum()
    fm = (grad_y * softmax(xm)).sum()
    grad_x_num[i] = (fp - fm) / (2 * eps)

print(f"VJP:      {grad_x_vjp}")
print(f"Numerical: {grad_x_num}")
print(f"오차: {np.max(np.abs(grad_x_vjp - grad_x_num)):.2e}")
```

### 10.3 Attention의 Backward

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention_forward(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    A = softmax(S, axis=-1)
    O = A @ V
    return O, A, S

def attention_backward(grad_O, Q, K, V, A, S):
    d = Q.shape[-1]
    grad_V = A.T @ grad_O
    grad_A = grad_O @ V.T
    # softmax backward per row
    grad_S = np.zeros_like(S)
    for i in range(S.shape[0]):
        y = A[i]
        g = grad_A[i]
        grad_S[i] = y * (g - (g * y).sum())
    grad_Q = grad_S @ K / np.sqrt(d)
    grad_K = grad_S.T @ Q / np.sqrt(d)
    return grad_Q, grad_K, grad_V

np.random.seed(0)
S_len, d = 6, 8
Q = np.random.randn(S_len, d)
K = np.random.randn(S_len, d)
V = np.random.randn(S_len, d)

O, A, S = attention_forward(Q, K, V)
grad_O = np.random.randn(*O.shape)
grad_Q, grad_K, grad_V = attention_backward(grad_O, Q, K, V, A, S)

# 수치 검증 (V만)
eps = 1e-6
grad_V_num = np.zeros_like(V)
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        orig = V[i, j]
        V[i, j] = orig + eps
        O_p, _, _ = attention_forward(Q, K, V)
        V[i, j] = orig - eps
        O_m, _, _ = attention_forward(Q, K, V)
        V[i, j] = orig
        grad_V_num[i, j] = ((O_p - O_m) * grad_O).sum() / (2 * eps)

print(f"V 그래디언트 오차: {np.max(np.abs(grad_V - grad_V_num)):.2e}")
```

### 10.4 Gradient Checkpointing 시뮬레이션

```python
import numpy as np

def deep_network_forward(x, Ws, checkpoint_every=None):
    activations = [x]
    for i, W in enumerate(Ws):
        x = np.tanh(x @ W)
        if checkpoint_every is None or i % checkpoint_every == 0:
            activations.append(x)
    return x, activations

L = 12  # layer 수
np.random.seed(0)
x = np.random.randn(10, 8)
Ws = [np.random.randn(8, 8) * 0.5 for _ in range(L)]

# 모든 활성값 저장
y1, acts_full = deep_network_forward(x, Ws, checkpoint_every=1)
print(f"Full: 저장된 활성값 수 = {len(acts_full)}")

# sqrt(L) 간격으로만 저장
import math
step = int(math.sqrt(L))
y2, acts_check = deep_network_forward(x, Ws, checkpoint_every=step)
print(f"Checkpoint: 저장된 활성값 수 = {len(acts_check)}")
print(f"메모리 절감: {len(acts_full) / len(acts_check):.2f}x")
```

### 10.5 Forward vs Reverse Mode 비교

```python
import numpy as np

def forward_jvp(f, x, v):
    """dual numbers 시뮬레이션"""
    eps = 1e-6
    return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)

def reverse_vjp(f, x, u):
    """수치 VJP (성분별)"""
    n = x.size
    J_row = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1
        J_row[i] = u @ forward_jvp(f, x, e_i)
    return J_row

# 예: scalar output, vector input
def f_scalar(x):
    return np.sum(np.sin(x) * np.exp(x[::-1]))

x = np.array([1.0, 2.0, 3.0, 4.0])
grad = reverse_vjp(f_scalar, x, np.array([1.0]))
print(f"Gradient: {grad}")

# 비교: 유한차분
eps = 1e-6
grad_fd = np.array([
    (f_scalar(x + eps*np.eye(len(x))[i]) - f_scalar(x - eps*np.eye(len(x))[i])) / (2*eps)
    for i in range(len(x))
])
print(f"FD:       {grad_fd}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 공식

| 연산 | Forward | VJP |
|---|---|---|
| 행렬곱 $Y = XW$ | | $\bar X = \bar Y W^T$, $\bar W = X^T \bar Y$ |
| Softmax $y = \sigma(x)$ | | $\bar x = y(\bar y - \langle \bar y, y\rangle)$ |
| CE Loss | | $\bar x = y - t$ |
| Attention | $A V$ | $\bar V = A^T \bar O$, etc. |

### 한 줄 요약

> **리버스 모드 자동미분은 $J^T \bar y$를 $J$ 구성 없이 계산하는 선형대수 알고리즘이다.**

### 다음 절 예고

다음 절에서 **BatchNorm**의 선형대수적 분석을 다룬다. "정규화"의 정확한 수학과 역전파 중 발생하는 **경사 재분배** 현상을 해부한다.

---

[◀ 01. Attention 선형대수](./01-attention-linear-algebra.md) | [📚 README](../README.md) | [03. BatchNorm ▶](./03-batchnorm.md)
