# 6.5 신경망 가중치의 텐서 구조

> "딥러닝은 거대한 텐서 수축(contraction) 그래프이다."

---

## 1. 학습 목표

- 신경망의 각 계층(FC, Conv, RNN, Attention)이 **텐서 연산**으로 어떻게 표현되는지 이해한다.
- 파라미터 수, 계산량, 메모리 요구를 텐서 shape으로부터 계산한다.
- **텐서 분해를 통한 네트워크 압축** 기법 (SVD, Tucker, TT)을 본다.
- 브로드캐스팅, 배치 처리, GPU 메모리 배치의 수학적 근거를 정리한다.
- **자동미분 (Autodiff)** 을 텐서곱 사슬로 해석한다.

---

## 2. 완전연결(FC) 계층

### 2.1 수학적 정의

입력 $x \in \mathbb{R}^{n}$, 가중치 $W \in \mathbb{R}^{m \times n}$, 편향 $b \in \mathbb{R}^m$:

$$
y = Wx + b, \quad y \in \mathbb{R}^m
$$

활성함수 $\sigma$ 적용: $a = \sigma(y)$.

### 2.2 배치 입력

배치 크기 $B$: $X \in \mathbb{R}^{B \times n}$.

$$
Y = X W^T + b \quad (\text{broadcasting}), \quad Y \in \mathbb{R}^{B \times m}
$$

Einsum:

```python
Y = np.einsum("bi,mi->bm", X, W) + b
```

### 2.3 파라미터 수와 계산량

- 파라미터: $mn + m$
- FLOPs (forward, 샘플당): $2mn$ (곱 $mn$, 덧셈 $mn$)
- 배치 전체: $2 B m n$

### 2.4 메모리

- 가중치: $mn$
- 활성값: $Bm$ (역전파 때 저장해야 함)
- 그래디언트: $mn$ 추가

### 2.5 SVD 압축

$W = U \Sigma V^T$, top-$r$ 보존:

$$
W \approx U_r \Sigma_r V_r^T = (U_r \Sigma_r)(V_r^T) = W_1 W_2
$$

$W_1 \in \mathbb{R}^{m \times r}$, $W_2 \in \mathbb{R}^{r \times n}$. 파라미터: $r(m + n)$. $r \ll \min(m, n)$이면 큰 절약.

---

## 3. 컨볼루션 계층

### 3.1 수학적 정의

입력 $I \in \mathbb{R}^{C_i \times H \times W}$, 필터 $K \in \mathbb{R}^{C_o \times C_i \times K_h \times K_w}$, 출력 $O \in \mathbb{R}^{C_o \times H' \times W'}$:

$$
O[c_o, y, x] = \sum_{c_i, k_y, k_x} K[c_o, c_i, k_y, k_x] \cdot I[c_i, y + k_y, x + k_x]
$$

(stride, padding 포함하면 더 복잡).

### 3.2 4D 텐서

배치 포함:

- 입력: $(B, C_i, H, W)$
- 필터: $(C_o, C_i, K_h, K_w)$
- 출력: $(B, C_o, H', W')$

### 3.3 im2col 변환

컨볼루션을 행렬곱으로 변환하면 BLAS를 활용 가능:

- 입력을 $(C_i K_h K_w) \times (B H' W')$로 재배치
- 필터를 $(C_o) \times (C_i K_h K_w)$로 reshape
- 행렬곱 후 출력으로 reshape

### 3.4 파라미터 수와 FLOPs

- 파라미터: $C_o \cdot C_i \cdot K_h \cdot K_w$
- 출력 위치 당 FLOPs: $2 \cdot C_i \cdot K_h \cdot K_w$
- 전체 FLOPs: $2 \cdot B \cdot C_o \cdot H' \cdot W' \cdot C_i \cdot K_h \cdot K_w$

### 3.5 Depthwise Separable 분해

일반 컨볼루션을

$$
\text{Depthwise}: \text{채널별 } K_h \times K_w \text{ 컨볼루션} + \text{Pointwise}: 1 \times 1 \text{ 컨볼루션}
$$

으로 분해. 파라미터: $C_i K_h K_w + C_i C_o$ (대비 $C_o C_i K_h K_w$). **MobileNet, Xception**의 핵심.

### 3.6 Tucker 분해 압축

커널 $K$를 Tucker로 분해:

$$
K[c_o, c_i, k_y, k_x] \approx \sum_{p, q} \mathcal{G}[p, q, k_y, k_x]\, U_o[c_o, p]\, U_i[c_i, q]
$$

이는 "$1 \times 1$ conv $\to$ $K_h \times K_w$ conv $\to$ $1 \times 1$ conv" 세 단계로 구현 가능.

---

## 4. RNN / LSTM / GRU

### 4.1 기본 RNN

$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$

- $W_h \in \mathbb{R}^{H \times H}$, $W_x \in \mathbb{R}^{H \times D}$, $h_t \in \mathbb{R}^H$
- 시퀀스 길이 $T$: 모든 단계에서 같은 $W$가 반복 사용 (**파라미터 공유**)

### 4.2 BPTT와 텐서 펼치기

Time dimension을 포함한 "펼쳐진" 그래프:

$$
h_T = \sigma(W_h \sigma(W_h \cdots \sigma(W_h h_0 + W_x x_1) \cdots) + W_x x_T)
$$

그래디언트는 $\prod_t W_h \cdot \text{diag}(\sigma'(y_t))$를 포함 — **기울기 소실/폭발**의 근원.

### 4.3 LSTM의 4개 게이트

$$
\begin{pmatrix} i \\ f \\ g \\ o \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{pmatrix}\left(W \begin{pmatrix} x_t \\ h_{t-1} \end{pmatrix} + b\right)
$$

$W \in \mathbb{R}^{4H \times (D + H)}$: 하나의 큰 행렬에 4개 게이트를 쌓아 GPU 효율 극대화.

---

## 5. Transformer / Attention

### 5.1 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

텐서 shape:

- $Q, K: (B, H, S, d_k)$
- $V: (B, H, S, d_v)$
- 출력: $(B, H, S, d_v)$

### 5.2 Einsum으로

```python
scores = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(d_k)
A = softmax(scores, axis=-1)  # (B, H, S, S)
O = np.einsum("bhij,bhjd->bhid", A, V)
```

### 5.3 Multi-Head의 파라미터

- $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$ (각각 $d^2$)
- Output projection $W^O \in \mathbb{R}^{d \times d}$
- 총 $4d^2$ 파라미터 (헤드 수에 무관, 왜냐하면 $d_k = d / H$로 나누기 때문)

### 5.4 FFN (Feed-Forward Network)

Transformer block 내:

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

- $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$
- 파라미터 $8d^2$ — **Attention보다 2배 많음!**

### 5.5 전체 Transformer

$L$개 레이어, 차원 $d$, 헤드 $H$:

- Attention: $4d^2 L$
- FFN: $8d^2 L$
- LayerNorm, embedding, position: 상대적으로 적음

**GPT-3 (175B)**: $L = 96$, $d = 12288$, 파라미터 $\approx 12 \cdot d^2 \cdot L \approx 12 \cdot 1.5 \times 10^8 \cdot 96 \approx 1.7 \times 10^{11}$. ✓

---

## 6. Embedding과 Lookup

### 6.1 Word Embedding

$E \in \mathbb{R}^{V \times d}$ (단어 사전 크기 $V$). 단어 $i$의 임베딩은 $E[i, :]$.

- 파라미터: $Vd$ — 큰 $V$ (예: $V = 50000$, $d = 768$)에 대해 수천만 파라미터
- 참조: 매우 빠름 (배열 인덱싱)

### 6.2 Factorized Embedding (ALBERT)

$$
E = E_1 E_2, \quad E_1 \in \mathbb{R}^{V \times d'}, E_2 \in \mathbb{R}^{d' \times d}, d' \ll d
$$

파라미터: $V d' + d' d$.

---

## 7. Autodiff와 텐서 수축 그래프

### 7.1 Forward: 텐서 수축

각 계층은 이전 활성값과 가중치의 텐서 수축:

$$
a^{(l)} = f^{(l)}(a^{(l-1)}, W^{(l)})
$$

### 7.2 Backward: 체인 룰

손실 $L$을 $W^{(l)}$에 대해 미분:

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial W^{(l)}}
$$

각 항은 텐서. 체인 룰은 이들의 축약 연산.

### 7.3 VJP (Vector-Jacobian Product)

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 Jacobian $J \in \mathbb{R}^{m \times n}$. 역전파에서 필요한 건

$$
u^T J = \text{VJP}(u)
$$

$J$를 **명시적으로** 구성하지 않고 같은 결과를 얻는 연산. 이것이 리버스 모드 자동미분의 핵심.

### 7.4 예: 행렬곱의 VJP

$Y = XW$에서 $\partial L / \partial Y = G$가 주어졌을 때

- $\partial L / \partial X = G W^T$
- $\partial L / \partial W = X^T G$

둘 다 텐서 축약이며 forward와 같은 복잡도.

### 7.5 Forward Mode (JVP)

$J v = \text{JVP}(v)$. Parameter 수보다 output 수가 적을 때 유리. 과학 계산이나 고차 미분에 사용.

---

## 8. 배치 정규화 (BatchNorm)의 텐서 뷰

### 8.1 정의

입력 $X \in \mathbb{R}^{B \times D}$ (또는 $(B, C, H, W)$ conv 활성):

$$
\mu = \frac{1}{B}\sum_b x_b, \quad \sigma^2 = \frac{1}{B}\sum_b (x_b - \mu)^2
$$

$$
\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat x + \beta
$$

### 8.2 축 선택의 중요성

- **BatchNorm**: 배치 축 전체에서 통계
- **LayerNorm**: 특징 축 전체에서 통계 (각 샘플 독립)
- **GroupNorm**: 채널을 그룹으로 나눠 그룹별
- **InstanceNorm**: 각 샘플, 각 채널별

이들은 모두 "어떤 축을 평균내는지"의 차이이다. Einsum 관점에서 모두 통일적.

---

## 9. 메모리와 계산의 트레이드오프

### 9.1 Gradient Checkpointing

모든 활성값을 저장하지 않고 일부만 저장. 역전파에서 필요시 재계산. 메모리 $O(\sqrt{L})$로 절감 (L은 레이어 수), 계산량 약 1.3배 증가.

### 9.2 Mixed Precision

FP32 대신 FP16 (또는 BF16). 메모리 절반, 계산 2~8배 (전용 하드웨어). 수치 안정성을 위해 loss scaling과 master weights in FP32.

### 9.3 ZeRO와 모델 병렬화

거대 모델의 파라미터를 여러 GPU에 분산:

- **ZeRO stage 1**: 옵티마이저 상태 분산
- **ZeRO stage 2**: + 그래디언트
- **ZeRO stage 3**: + 파라미터
- **Pipeline**, **Tensor** 병렬화와 조합

---

## 10. Python 실험

### 10.1 FC 계층의 SVD 압축

```python
import numpy as np

np.random.seed(0)
m, n = 1024, 1024
W = np.random.randn(m, n) / np.sqrt(n)

# SVD 압축
U, s, Vt = np.linalg.svd(W, full_matrices=False)

for r in [16, 64, 256, 1024]:
    W_r = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
    err = np.linalg.norm(W - W_r) / np.linalg.norm(W)
    params_orig = m * n
    params_compressed = r * (m + n)
    ratio = params_compressed / params_orig
    print(f"r={r:4d}: err={err:.4f}, 파라미터 비율={ratio:.3f}x")
```

### 10.2 Conv 파라미터/FLOPs 계산

```python
def conv_stats(C_i, C_o, K, H, W, B=1):
    """Conv 레이어의 파라미터와 FLOPs 계산"""
    params = C_o * C_i * K * K + C_o  # 편향 포함
    flops = 2 * B * C_o * H * W * C_i * K * K
    return params, flops

# ResNet-50 첫 번째 conv
p, f = conv_stats(C_i=3, C_o=64, K=7, H=112, W=112)
print(f"Conv1 파라미터: {p:,}, FLOPs: {f:,}")

# ResNet bottleneck 3번째 conv
p, f = conv_stats(C_i=2048, C_o=512, K=1, H=7, W=7)
print(f"Bottleneck: 파라미터: {p:,}, FLOPs: {f:,}")
```

### 10.3 Attention 구현 (Multi-head)

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def multihead_attention(Q, K, V, num_heads):
    B, S, D = Q.shape
    d_h = D // num_heads
    
    # (B, S, D) -> (B, H, S, d_h)
    def split_heads(x):
        return x.reshape(B, S, num_heads, d_h).transpose(0, 2, 1, 3)
    
    Qh = split_heads(Q)
    Kh = split_heads(K)
    Vh = split_heads(V)
    
    scores = np.einsum("bhid,bhjd->bhij", Qh, Kh) / np.sqrt(d_h)
    A = softmax(scores, axis=-1)
    O = np.einsum("bhij,bhjd->bhid", A, Vh)
    
    # (B, H, S, d_h) -> (B, S, D)
    return O.transpose(0, 2, 1, 3).reshape(B, S, D)

np.random.seed(0)
B, S, D, H = 2, 10, 64, 4
Q = np.random.randn(B, S, D)
K = np.random.randn(B, S, D)
V = np.random.randn(B, S, D)

out = multihead_attention(Q, K, V, num_heads=H)
print(f"출력 shape: {out.shape}")
```

### 10.4 간단한 Autodiff

```python
import numpy as np

class Tensor:
    def __init__(self, data, parents=(), backward_fn=None):
        self.data = np.asarray(data)
        self.grad = None
        self.parents = parents
        self.backward_fn = backward_fn

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad
        if self.backward_fn is not None:
            self.backward_fn(grad)

def matmul(A, B):
    out = Tensor(A.data @ B.data, parents=(A, B))
    def _back(g):
        A.backward(g @ B.data.T)
        B.backward(A.data.T @ g)
    out.backward_fn = _back
    return out

def mse(pred, target):
    diff = pred.data - target.data
    loss = Tensor(np.mean(diff**2), parents=(pred, target))
    def _back(g):
        pred.backward(g * 2 * diff / diff.size)
    loss.backward_fn = _back
    return loss

np.random.seed(0)
X = Tensor(np.random.randn(10, 5))
W = Tensor(np.random.randn(5, 3))
target = Tensor(np.random.randn(10, 3))

Y = matmul(X, W)
loss = mse(Y, target)
loss.backward()

# 수치 미분과 대조
eps = 1e-6
numerical_grad = np.zeros_like(W.data)
for i in range(W.data.shape[0]):
    for j in range(W.data.shape[1]):
        orig = W.data[i, j]
        W.data[i, j] = orig + eps
        l_plus = np.mean((X.data @ W.data - target.data)**2)
        W.data[i, j] = orig - eps
        l_minus = np.mean((X.data @ W.data - target.data)**2)
        W.data[i, j] = orig
        numerical_grad[i, j] = (l_plus - l_minus) / (2 * eps)

print(f"Autodiff 그래디언트 / 수치 미분 오차: {np.max(np.abs(W.grad - numerical_grad)):.2e}")
```

### 10.5 Tucker 압축을 통한 Conv 레이어 근사

```python
import numpy as np

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

def tucker_conv_compress(K, r_out, r_in):
    """
    K: (C_o, C_i, kh, kw)
    Tucker 분해: K = G x_1 U_o x_2 U_i
    """
    C_o, C_i, kh, kw = K.shape
    
    # mode-1 SVD
    U_o, _, _ = np.linalg.svd(unfold(K, 0), full_matrices=False)
    U_o = U_o[:, :r_out]
    
    # mode-2 SVD
    U_i, _, _ = np.linalg.svd(unfold(K, 1), full_matrices=False)
    U_i = U_i[:, :r_in]
    
    # core
    core = np.einsum("oipq,or,is->rspq", K, U_o, U_i)
    
    # 복원
    K_rec = np.einsum("rspq,or,is->oipq", core, U_o, U_i)
    
    orig_params = K.size
    comp_params = U_o.size + U_i.size + core.size
    err = np.linalg.norm(K - K_rec) / np.linalg.norm(K)
    return K_rec, orig_params, comp_params, err

np.random.seed(0)
K = np.random.randn(256, 128, 3, 3)
_, orig, comp, err = tucker_conv_compress(K, r_out=64, r_in=32)
print(f"원래: {orig:,} 파라미터")
print(f"압축: {comp:,} 파라미터 ({comp/orig:.3f}x)")
print(f"상대 오차: {err:.4f}")
```

---

## 11. 요약 및 Chapter 6 마무리

### 계층별 텐서 요약

| 계층 | 주요 텐서 | 파라미터 | 주요 연산 |
|---|---|---|---|
| FC | $W \in \mathbb{R}^{m \times n}$ | $mn$ | 행렬곱 |
| Conv | $K \in \mathbb{R}^{C_o \times C_i \times k \times k}$ | $C_o C_i k^2$ | im2col + 행렬곱 |
| RNN | $W_h, W_x$ | $H^2 + HD$ | 시간 재귀 |
| Self-Attn | $W_{Q, K, V, O}$ | $4d^2$ | QK^T, softmax V |
| FFN | $W_1, W_2$ | $8d^2$ | 2-층 FC |
| Embedding | $E \in \mathbb{R}^{V \times d}$ | $Vd$ | 인덱싱 |

### 한 줄 요약

> **신경망은 거대한 텐서 수축 그래프이며, 압축·가속은 이 그래프의 구조를 이용한다.**

### Chapter 6 정리

우리는 제6장에서

1. **텐서의 정의** (다중선형 사상)
2. **Kronecker 곱과 vec** (행렬 대수로의 환원)
3. **Einstein 합 규약과 einsum** (표기와 계산의 통일)
4. **텐서 분해** (CP, Tucker, TT)
5. **신경망 가중치** (응용의 정점)

을 차례로 보았다. 추상에서 출발하여 실용에 도달했다.

### 다음 장 예고

마지막 장 **Chapter 7: AI/ML 응용**에서는 지금까지 쌓은 모든 선형대수 도구를 **최신 딥러닝 기법**에 적용한다: Attention의 선형대수적 분석, 역전파, BatchNorm, Spectral Normalization, RoPE, 랜덤 행렬 이론.

---

[◀ 04. 텐서 분해](./04-tensor-decomposition.md) | [📚 README](../README.md) | [7장: AI/ML 응용 ▶](../ch7-ml-applications/01-attention-linear-algebra.md)
