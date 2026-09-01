# Stage 2 HC/SZ Classification — giải thích module, tensor shape và model components

## 1. Mục tiêu của tài liệu

Tài liệu này mô tả chi tiết architecture Stage 2 dùng HC normative bank theo stimulus để phân loại subject thành:

- `HC = 0`: Healthy Control;
- `SZ = 1`: Schizophrenia.

Stage 2 nhận heatmap của toàn bộ stimulus panel của một subject, so sánh từng trial với HC normative representation của **đúng stimulus**, sau đó tổng hợp bằng attention ở cấp stimulus/category và đưa ra dự đoán ở cấp subject.

Tài liệu phân biệt rõ:

- **Base model**: trial-level normative bank, bắt buộc;
- **Token semantic model**: fused-token normative bank và serial local attention, tùy chọn;
- **Same-space heat-token model**: nhánh ablation tùy chọn, chỉ hợp lệ khi encoder được frozen và checkpoint khớp chính xác.

Sơ đồ đi kèm:

```text
ems-stage2-methodology-overview.png
ems-stage2-methodology-overview.md
ems-stage2-methodology-overview.mmd
```

## 2. Ký hiệu tensor

| Ký hiệu | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| `B` | thay đổi theo batch | số subject trong một batch |
| `S` | `100` | tổng số stimulus slot của một subject |
| `C` | `3` | số channel của heatmap |
| `H × W` | `48 × 64` | kích thước heatmap |
| `P` | `192` | số heatmap patch, vì `12 × 16 = 192` |
| `D` | `128` | embedding dimension chính |
| `K` | `4` | số stimulus category |
| `N` | `≤ B × S` | tổng số trial hợp lệ sau khi loại missing slots |
| `h` | `4` | số attention head mặc định |
| `d_head` | `32` | dimension mỗi head, `128 / 4` |

Bốn category của EMS:

| Category | Số stimulus |
|---|---:|
| Manipulated Images | 32 |
| Natural Scenes | 31 |
| Social Scenes | 22 |
| Synthetic Images | 15 |
| **Tổng** | **100** |

`N` không nên được giả định bằng `B × 100`, vì một số subject thiếu trial. Ví dụ:

```text
B = 4
số trial hợp lệ mỗi subject = [100, 98, 100, 88]
N = 100 + 98 + 100 + 88 = 386
```

## 3. Tổng quan data flow

```text
Offline bank construction
    training HC heatmaps + DINO stimulus features
    -> full Stage-1 model
    -> group representation by stimulus_index
    -> fold-specific HC normative bank

Stage-2 subject classification
    subject heatmap panel
    -> frozen Stage-1 heatmap encoder
    -> new Stage-2 query pooling
    -> matched-stimulus normative comparator
    -> optional token semantic branch
    -> category-balanced stimulus attention
    -> subject Transformer
    -> main HC/SZ classifier

Parallel interpretation path
    stimulus importance + signed contribution
    + normative deviation + semantic compatibility
    + optional token semantic map
```

Điểm phân tách quan trọng:

- full Stage-1 model và DINO được dùng để **xây bank offline**;
- query subject trong Stage 2 **không chạy DINO**, semantic adapter, Stage-1 fusion, decoder hoặc Stage-1 pooling head;
- Stage 2 chỉ chuyển `HeatmapPatchEncoder` và weight tương ứng từ Stage 1.

## 4. Bảng tổng hợp các module

| Module | Model component chính | Input | Output | Trạng thái base |
|---|---|---|---|---|
| Normative bank builder | full Stage-1 model, streaming statistics | HC trials | bank theo stimulus | offline, không gradient |
| Subject dataset/collator | masked subject panel | subject files | `[B,100,3,48,64]` | không train |
| Transferred heatmap encoder | Stage-1 `HeatmapPatchEncoder` | `[N,3,48,64]` | `[N,192,128]` | frozen |
| Query attention pooler | LN–MLP–softmax–weighted sum | `[N,192,128]` | `[N,128]` | trainable |
| Query projection | LN–Linear | `[N,128]` | `[N,128]` | trainable |
| Bank lookup | indexed tensor gather | stimulus indices `[N]` | mean/std/count | frozen bank |
| Bank mean adapter | LN–Linear | `[N,128]` | `[N,128]` | trainable |
| Bank uncertainty adapter | LN–MLP | std/count | `[N,128]` | trainable |
| Reliability head | small MLP–sigmoid | `[N,2]` | `[N,1]` | trainable |
| Trial relation encoder | explicit relation + MLP | `[N,770]` | `[N,128]` | trainable |
| Token semantic branch | adapters + Local MHA ×2 + CNN bridge | token tensors | `[N,128]` + map | optional |
| Trial fusion | concat–MLP–residual | `[N,256]` | `[N,128]` | token mode |
| Stimulus/category embedding | embedding tables | stimulus/category IDs | `[B,100,128]` | trainable |
| Category-balanced attention | gated attention | `[B,100,128]` | `[B,4,128]` | trainable |
| Subject Transformer | 1 Transformer layer | `[B,5,128]` | `[B,128]` | trainable |
| Main classifier | LN–MLP–Linear | `[B,128]` | `[B]` | trainable |
| Additive evidence head | shared trial MLP + weighted sum | `[B,100,128]` | `[B]` | trainable |
| Interpretation outputs | deterministic reductions | model intermediates | subject/stimulus scores | không phải classifier riêng |

## 5. Module A — fold-specific HC normative bank

### 5.1 Mục đích

Normative bank biểu diễn response điển hình của HC đối với từng stimulus. Thay vì hỏi “heatmap này nhìn giống HC nói chung không?”, model hỏi:

> Với stimulus `s` cụ thể, response của subject lệch khỏi HC norm của chính stimulus `s` như thế nào?

Điều này loại bớt variation do nội dung stimulus. Hai heatmap của hai ảnh khác nhau không nên bị so sánh trực tiếp như cùng một task.

### 5.2 Input khi xây bank

Cho outer fold `k`:

```text
training HC heatmaps           [n,3,48,64]
stimulus_index                 [n]
precomputed DINO features      shape theo Stage-1 manifest
subject_id                     length n
trial_uid                      length n
```

Chỉ HC thuộc training partition của fold `k` được phép đóng góp.

### 5.3 Full Stage-1 inference

Bank builder load toàn bộ Stage-1 model vì bank trial/token là representation **sau semantic fusion**:

```text
heatmap
-> Stage-1 HeatmapPatchEncoder
-> DINO semantic adapter
-> serial semantic cross-attention blocks
-> Stage-1 pooling
-> trial embedding [n,128]
```

Inference phải dùng:

```text
model.eval()
torch.inference_mode()
token_mask = None
```

Không dùng masked Stage-1 output để làm normative statistics.

### 5.4 Statistics theo stimulus

Với stimulus `s`, HC representation của subject `h` là `z_{h,s} ∈ R^128`:

\[
\mu_s=\frac{1}{n_s}\sum_{h=1}^{n_s}z_{h,s}
\]

\[
\sigma_s=
\sqrt{
\frac{1}{n_s}\sum_{h=1}^{n_s}(z_{h,s}-\mu_s)^2
+\epsilon
}
\]

Tích lũy bằng float64:

```text
sum_trial       float64 [100,128]
sumsq_trial     float64 [100,128]
count_trial     int64   [100]
```

Sau khi finalize:

```text
mu_trial        float32 [100,128]
sigma_trial     float32 [100,128]
count_trial     int32   [100]
```

Nếu token bank được bật:

```text
mu_token        float32 [100,192,128]
sigma_token     float32 [100,192,128]
```

Nếu same-space heat-token bank được bật:

```text
mu_heat_token       float32 [100,192,128]
sigma_heat_token    float32 [100,192,128]
```

Tất cả bank arrays là buffers/read-only tensors và luôn có:

```python
requires_grad = False
```

### 5.5 Full bank và crossfit bank

Mỗi outer fold có:

- `full bank`: dùng toàn bộ training HC của outer fold; dùng cho validation/test;
- `crossfit bank`: chia training subjects thành bốn split; subject thuộc split `j` dùng bank được xây từ HC không thuộc split `j`.

Việc crossfit tránh trường hợp training HC subject được so sánh với bank có chứa chính response của họ.

SZ training subjects cũng được gán `bank_split_id`, dù họ không đóng góp vào bank. Điều này đảm bảo số contributor của bank không trở thành tín hiệu ngầm cho class.

## 6. Module B — subject-level Dataset và DataLoader

### 6.1 Vì sao một sample phải là một subject

Diagnostic label thuộc về subject, không thuộc độc lập từng trial. Nếu biến 15,912 trial thành 15,912 diagnostic samples:

- cùng một subject xuất hiện lặp lại nhiều lần;
- statistical independence bị vi phạm;
- metric và effective sample size bị thổi phồng;
- model có thể học subject-specific shortcut.

Vì vậy:

```text
Dataset length = số subject
one dataset item = toàn bộ 100-slot panel của một subject
```

### 6.2 Một dataset item

```python
{
    "subject_id": str,
    "label": float,                  # scalar, HC=0, SZ=1
    "heatmaps": Tensor,              # [100,3,48,64]
    "trial_mask": BoolTensor,        # [100]
    "stimulus_indices": LongTensor,  # [100]
    "category_ids": LongTensor,      # [100]
    "trial_uids": list,              # length 100
    "bank_split_id": int | None,
}
```

Slot thiếu trial có thể chứa zero tensor để collate, nhưng `trial_mask=False` và không được phép đi vào encoder, softmax hoặc loss.

### 6.3 Collated batch

```text
labels                    [B]
heatmaps                  [B,100,3,48,64]
trial_mask                [B,100]
stimulus_indices          [B,100]
category_ids              [B,100]
bank_split_ids            [B] hoặc None
```

### 6.4 Flatten valid trials

Chỉ lấy vị trí `trial_mask=True`:

```text
valid_heatmaps            [N,3,48,64]
flat_stimulus_indices     [N]
flat_category_ids         [N]
flat_subject_slots        [N]
flat_stimulus_slots       [N]
```

Hai index `flat_subject_slots` và `flat_stimulus_slots` được giữ lại để scatter representation về `[B,100,...]` sau này.

## 7. Module C — transferred Stage-1 HeatmapPatchEncoder

### 7.1 Mục đích

Module này chuyển heatmap thành spatial gaze tokens. Đây là phần duy nhất của query encoder được chuyển từ Stage 1.

Base model load đúng các key:

```text
heatmap_encoder.*
```

Không load:

```text
DINO adapter
semantic fusion
decoder
Stage-1 pooling head
```

### 7.2 Model components và shape

| Operation | Component | Output shape |
|---|---|---:|
| Input | heatmap tensor | `[N,3,48,64]` |
| Patch projection | `Conv2d(3,128,kernel=4,stride=4)` | `[N,128,12,16]` |
| Normalization | `GroupNorm(32)` | `[N,128,12,16]` |
| Activation | `GELU` | `[N,128,12,16]` |
| Flatten spatial grid | row-major `12×16 -> 192` | `[N,192,128]` |
| Position encoding | fixed 2-D sine/cosine | `[N,192,128]` |
| Spatial residual block 1 | residual token/spatial block | `[N,192,128]` |
| Spatial residual block 2 | residual token/spatial block | `[N,192,128]` |
| Output | heat tokens `H` | `[N,192,128]` |

Stage 2 luôn gọi:

```python
H = heatmap_encoder(valid_heatmaps, token_mask=None)
```

### 7.3 Frozen policy

Trong base model:

```text
encoder parameters             requires_grad=False
encoder mode                   eval()
output H                       vẫn đi vào trainable Stage-2 modules
```

Gradient không cần được giữ qua encoder nếu toàn bộ encoder frozen. Tuy nhiên không được detach những Stage-2 tensors phía sau query projection.

Named ablation `unfreeze_last_block` chỉ mở final residual block với learning rate nhỏ hơn khoảng 10 lần và bật anchor loss.

## 8. Module D — Stage-2 Query Attention Pooler

### 8.1 Mục đích

Heat encoder trả 192 patch tokens. Không phải patch nào cũng đóng góp như nhau cho normative comparison. Query pooler học trọng số patch riêng cho Stage 2.

Không tái sử dụng Stage-1 trial pooling head vì:

- Stage-1 pooling được tối ưu cho objective Stage 1;
- Stage-2 query chỉ có heat information;
- Stage-2 cần align query mới với frozen normative bank.

### 8.2 Model components

```text
H [N,192,128]
-> LayerNorm(128)
-> Linear(128,64)
-> GELU hoặc tanh
-> Linear(64,1)
-> logits [N,192]
-> softmax over 192 patches
-> patch_attention [N,192]
-> weighted sum of H
-> q0 [N,128]
```

\[
a_{i,t}=\operatorname{softmax}_t
\left(w_2^\top\phi(W_1\operatorname{LN}(H_{i,t}))\right)
\]

\[
q^0_i=\sum_{t=1}^{192}a_{i,t}H_{i,t}
\]

Mọi patch của một trial hợp lệ đều tham gia softmax. Patch có zero gaze density vẫn là patch thật, không phải missing patch.

### 8.3 Query projection

```text
q0 [N,128]
-> LayerNorm(128)
-> Linear(128,128)
-> q [N,128]
```

`q` là query representation dùng cho learned normative alignment.

## 9. Module E — stimulus-matched bank lookup

### 9.1 Indexed gather

Với `flat_stimulus_indices [N]`, lấy đúng hàng bank:

```text
mu_raw       = mu_trial[stimulus_indices]       [N,128]
sigma_raw    = sigma_trial[stimulus_indices]    [N,128]
count        = count_trial[stimulus_indices]    [N]
```

Không được:

- lấy average bank chung cho 100 stimulus trong base model;
- sort theo filename thay vì canonical `stimulus_index`;
- so sánh trial của stimulus `s` với bank của `s'`;
- dùng validation subject để cập nhật bank.

### 9.2 Tại sao cần independent adapters

Hai representation không nằm sẵn trong cùng không gian:

```text
Stage-2 q0           heat-only, pre-fusion
mu_trial             Stage-1 post-semantic-fusion trial embedding
```

Vì vậy các thao tác sau là không hợp lệ nếu dùng trực tiếp:

```text
q0 - mu_raw
cosine(q0, mu_raw)
q0 / sigma_raw
```

Model phải học hai phép chiếu độc lập và dùng matching objective để tạo comparable latent space.

## 10. Module F — bank mean, uncertainty và reliability adapters

### 10.1 Mean adapter

```text
mu_raw [N,128]
-> LayerNorm(128)
-> Linear(128,128)
-> n_mu [N,128]
```

`n_mu` là HC normative center sau learned alignment.

### 10.2 Uncertainty adapter

Đầu tiên:

```text
log_sigma = log(sigma_raw + epsilon)           [N,128]
log_count = log1p(count)                       [N,1]
```

Sau đó:

```text
LayerNorm(log_sigma)                           [N,128]
concat with log_count                          [N,129]
Linear(129,256)
GELU
Dropout
Linear(256,128)
uncertainty_context                            [N,128]
```

Module này không coi mọi dimension/stimulus có độ tin cậy bằng nhau. `sigma` lớn hoặc `count` thấp có thể làm normative evidence yếu đi.

### 10.3 Reliability head

```text
mean(-log_sigma, dim=feature)                   [N,1]
log_count                                      [N,1]
concat                                         [N,2]
Linear(2,32) -> GELU -> Linear(32,1) -> sigmoid
rho                                            [N,1]
```

`rho` gần 1 biểu diễn normative evidence đáng tin hơn; gần 0 làm giảm ảnh hưởng của deviation score. Đây là learned reliability, không phải calibrated probability.

## 11. Module G — explicit Normative Relation Encoder

### 11.1 Mục đích

Module này tạo representation mô tả cả:

- query response của subject;
- HC normative center;
- uncertainty của bank;
- hướng và độ lớn deviation;
- query–norm compatibility;
- reliability của normative estimate.

### 11.2 Relation tensor

| Thành phần | Shape | Dimension |
|---|---:|---:|
| Query `q` | `[N,128]` | 128 |
| Normative center `n_mu` | `[N,128]` | 128 |
| Uncertainty context | `[N,128]` | 128 |
| Signed difference `q - n_mu` | `[N,128]` | 128 |
| Absolute difference `abs(q - n_mu)` | `[N,128]` | 128 |
| Elementwise interaction `q * n_mu` | `[N,128]` | 128 |
| Learned-space cosine | `[N,1]` | 1 |
| Reliability `rho` | `[N,1]` | 1 |
| **Concatenated relation** | **`[N,770]`** | **770** |

Kiểm tra dimension:

\[
6\times128+1+1=770.
\]

### 11.3 Relation MLP

```text
relation [N,770]
-> Linear(770,256)
-> GELU
-> Dropout(0.25)
-> Linear(256,128)
-> add residual projection of q
-> LayerNorm(128)
-> z_trial [N,128]
```

Residual query giữ lại subject-specific heat information nếu normative branch chưa ổn định ở đầu training.

### 11.4 Base semantic outputs

Sau khi scatter về subject panel:

\[
\text{compatibility}_{i,s}=\cos(q_{i,s},n_{\mu,s})
\]

\[
\text{deviation}_{i,s}=\rho_{i,s}
\left[1-\cos(q_{i,s},n_{\mu,s})\right]
\]

```text
semantic_compatibility        [B,100]
normative_deviation           [B,100]
```

Trong base model, “semantic compatibility” có nghĩa là compatibility trong learned space được align với **post-fusion HC bank**. Nó không phải raw DINO similarity và chưa phải object-level explanation.

## 12. Module H — optional fused-token semantic branch

### 12.1 Khi nào bật

Chỉ bật khi bank metadata xác nhận có:

```text
mu_token       [100,192,128]
sigma_token    [100,192,128]
```

Config ví dụ:

```yaml
model:
  bank_mode: trial_and_fused_token
bank:
  require_fused_token_bank: true
```

### 12.2 Token lookup

Sau stimulus-index gather:

```text
query heat tokens H          [N,192,128]
mu_token                     [N,192,128]
sigma_token                  [N,192,128]
```

Query token là heat-only pre-fusion, trong khi bank token là post-fusion. Vì vậy vẫn phải dùng independent token adapters.

### 12.3 Token adapters

```text
H -> query token projection                  Q       [N,192,128]
mu_token -> bank token mean adapter          N_mu    [N,192,128]
sigma_token -> uncertainty adapter           N_ctx   [N,192,128]
sigma/count -> token reliability             rho_t   [N,192,1]
```

### 12.4 Per-token relation

Tại mỗi patch `t`:

```text
Q                                  [N,192,128]
N_mu                               [N,192,128]
abs(Q-N_mu)                        [N,192,128]
Q*N_mu                             [N,192,128]
cosine(Q,N_mu)                     [N,192,1]
rho_t                              [N,192,1]
```

Concatenate:

\[
4\times128+1+1=514.
\]

```text
token relation                     [N,192,514]
MLP 514 -> 256 -> 128
R0                                 [N,192,128]
```

### 12.5 Local 3×3 normative windows

Mỗi query patch chỉ attend vào 3×3 neighborhood tương ứng của normative token grid:

```text
local K/V                          [N,192,9,128]
neighbor_valid_mask                [192,9]
relative_position_bias             [4,9]
```

Với bốn attention heads:

```text
attention weights                  [N,4,192,9]
```

Không được wrap từ cột 15 của một row sang cột 0 của row tiếp theo.

### 12.6 Serial topology

Topology bắt buộc là nối tiếp:

```text
R0 [N,192,128]
-> LocalMHA1 + residual + LayerNorm
H1 [N,192,128]
-> spatial bridge + gated residual + LayerNorm
Hb [N,192,128]
-> LocalMHA2 + residual + LayerNorm
H2 [N,192,128]
```

Hai Local MHA không share parameter. `LocalMHA2` phải nhận `Hb`, không được quay lại nhận `R0`.

### 12.7 Spatial bridge components

```text
Hb input before bridge       [N,192,128]
reshape                      [N,128,12,16]
DepthwiseConv2d              128 groups, kernel=3, padding=1
PointwiseConv2d              128 -> 256
GELU + Dropout
PointwiseConv2d              256 -> 128
reshape back                 [N,192,128]
gated residual + LayerNorm   [N,192,128]
```

Spatial bridge cho phép information truyền giữa neighboring heatmap regions bằng convolutional inductive bias trước attention layer thứ hai.

### 12.8 Token pooling và trial fusion

Pool `H2`:

```text
H2                                 [N,192,128]
token attention pooling
d_token                            [N,128]
```

Fuse với base trial representation:

```text
concat(z_trial,d_token)            [N,256]
Linear(256,128)
GELU + Dropout
add residual z_trial
LayerNorm(128)
z_extended                        [N,128]
```

Subject aggregator nhận `z_trial` trong base mode và `z_extended` trong token mode.

### 12.9 Semantic patch map

Một map có thể kết hợp stimulus importance, normalized attention, token reliability và token mismatch:

\[
M_{i,s,t}
=I_{i,s}\bar A_{i,s,t}\rho_{s,t}
\left[1-c_{i,s,t}\right].
\]

```text
per-trial token map                [N,192]
scatter                            [B,100,192]
reshape                            [B,100,12,16]
```

Map này trả lời patch nào tạo ra matched normative compatibility/deviation mạnh hơn. Không nên gọi nó là causal explanation hoặc object-level semantic explanation nếu chưa kiểm chứng với image regions.

## 13. Module I — optional same-space heat-token branch

Nhánh này dùng bank được xây từ chính output của frozen heatmap encoder:

```text
mu_heat_token          [100,192,128]
sigma_heat_token       [100,192,128]
```

Chỉ trong trường hợp:

- bank và query encoder có cùng checkpoint SHA-256;
- encoder hoàn toàn frozen;
- preprocessing và token ordering giống hệt;

thì direct standardized residual mới hợp lệ:

\[
Z^{heat}=\operatorname{clip}
\left(
\frac{H-\mu^{heat}}{\sigma^{heat}+\epsilon},-5,5
\right).
\]

Đây là named ablation, không phải base model. Nếu unfreeze encoder, branch phải fail hoặc tự disable với lỗi rõ ràng.

## 14. Module J — scatter trial features về subject panel

Sau comparison/fusion:

```text
flat trial representation        [N,128]
flat_subject_slots               [N]
flat_stimulus_slots              [N]
```

Scatter:

```text
Z                              [B,100,128]
trial_mask                     [B,100]
```

Missing slots:

```text
Z[b,s] = 0
trial_mask[b,s] = False
```

Zero tensor chỉ là storage placeholder; mọi downstream attention/reduction vẫn phải dùng mask.

## 15. Module K — stimulus và category embeddings

### 15.1 Components

```text
stimulus embedding table        [100,128]
category embedding table        [4,128]
```

Với valid slot `(b,s)`:

\[
\tilde z_{b,s}
=\operatorname{LN}
\left(z_{b,s}+e^{stim}_s+e^{cat}_{k(s)}\right).
\]

Output:

```text
enriched trial panel            [B,100,128]
```

Stimulus embedding cho model biết identity của từng stimulus; category embedding cung cấp coarse semantic grouping. Normative bank vẫn được chọn bằng explicit index, không phải bằng learned embedding nearest neighbor.

## 16. Module L — category-balanced gated attention

### 16.1 Vì sao không dùng một softmax trên 100 stimulus

Các category có số stimulus không bằng nhau: 32, 31, 22 và 15. Một global softmax có thể cho category lớn nhiều cơ hội tích lũy attention hơn.

Category-balanced attention:

1. tính score từng stimulus;
2. softmax riêng trong từng category;
3. tạo bốn category tokens;
4. chia global importance đều cho các category hiện diện.

### 16.2 Gated attention components

```text
z panel                            [B,100,128]
Linear V: 128 -> 64 + tanh         [B,100,64]
Linear U: 128 -> 64 + sigmoid      [B,100,64]
elementwise gate                   [B,100,64]
Linear w: 64 -> 1                  [B,100,1]
score                              [B,100]
```

\[
g_{i,s}=w^\top
\left[
\tanh(Vz_{i,s})\odot\sigma(Uz_{i,s})
\right].
\]

### 16.3 Within-category attention

Với category `k`:

\[
\alpha^{(k)}_{i,s}
=\frac{\exp(g_{i,s})}
{\sum_{j\in\mathcal S_k,M_{i,j}=1}\exp(g_{i,j})}.
\]

```text
alpha within category              [B,100]
category-present mask              [B,4]
```

Missing trial score được set `-inf` trước softmax.

### 16.4 Category tokens và global importance

\[
c_{i,k}=\sum_{s\in\mathcal S_k}
\alpha^{(k)}_{i,s}z_{i,s}.
\]

```text
category tokens                    [B,4,128]
```

Nếu subject có `K_i` category hiện diện:

\[
I_{i,s}=\frac{\alpha^{(k(s))}_{i,s}}{K_i}.
\]

```text
stimulus importance I              [B,100]
sum over valid stimuli             1 cho mỗi subject
total mass mỗi present category    1/K_i
```

`I` trả lời stimulus nào được model sử dụng nhiều hơn, nhưng attention weight một mình không chứng minh causal importance.

## 17. Module M — Subject Transformer

### 17.1 Input sequence

Prepend một learned subject token:

```text
subject token                      [B,1,128]
four category tokens               [B,4,128]
Transformer sequence               [B,5,128]
```

### 17.2 Default components

```text
d_model                            128
num_heads                          4
head_dim                           32
FFN hidden                         256
layers                             1
dropout                            0.25
activation                         GELU
normalization                      pre-LN hoặc repo-consistent
```

Output:

```text
Transformer output                [B,5,128]
subject embedding u               output subject token [B,128]
```

Transformer học interaction giữa bốn loại stimulus response. Ví dụ, deviation ở Social Scenes có thể chỉ hữu ích khi được xét cùng Natural hoặc Manipulated Scenes.

Category không hiện diện phải được mask hoặc thay bằng learned missing-category token kèm mask; không được trình bày như observed response.

## 18. Module N — Main HC/SZ classifier

Contract bắt buộc:

```text
subject embedding u               [B,128]
main_logit                        [B]
```

Recommended component cụ thể:

```text
LayerNorm(128)
Linear(128,64)
GELU
Dropout(0.25)
Linear(64,1)
squeeze last dimension
main_logit [B]
```

Probability:

\[
p_i=\sigma(\ell_i^{main}).
\]

Không đặt sigmoid bên trong loss nếu dùng `BCEWithLogitsLoss`.

## 19. Module O — Additive Evidence Head

### 19.1 Mục đích

Main Transformer classifier có thể chính xác nhưng khó quy trách nhiệm cho từng stimulus. Auxiliary head buộc model xây một decomposition cộng được:

```text
subject auxiliary logit
= bias + tổng signed contribution của các stimulus
```

### 19.2 Shared evidence MLP

Recommended component:

```text
trial representation              [B,100,128]
LayerNorm(128)
Linear(128,64)
GELU hoặc tanh
Dropout(0.25)
Linear(64,1)
evidence e                        [B,100]
```

### 19.3 Contribution và auxiliary logit

\[
C_{i,s}=I_{i,s}e_{i,s}.
\]

\[
\ell_i^{aux}=b+\sum_{s:M_{i,s}=1}C_{i,s}.
\]

```text
stimulus evidence                 [B,100]
stimulus contribution             [B,100]
auxiliary logit                   [B]
```

Theo convention `SZ=1`:

- contribution dương hỗ trợ SZ;
- contribution âm hỗ trợ HC;
- magnitude lớn biểu diễn ảnh hưởng mạnh hơn trong auxiliary additive head.

Implementation phải test:

```text
aux_logit - bias == masked_sum(contribution)
```

## 20. Module P — interpretation outputs

### 20.1 Stimulus importance

```text
stimulus_importance = I           [B,100]
```

Cho biết model phân bổ attention thế nào, đã cân bằng tổng mass giữa các category hiện diện.

### 20.2 Signed contribution

```text
stimulus_contribution = I * e     [B,100]
```

Đây là output phù hợp hơn attention thuần khi hỏi stimulus nào đẩy dự đoán về HC hoặc SZ.

### 20.3 Semantic compatibility

```text
semantic_compatibility            [B,100]
```

Là cosine trong learned aligned query–bank space. Score cao nghĩa là query response tương thích hơn với HC normative representation của stimulus đó.

### 20.4 Normative deviation

```text
normative_deviation
= reliability * (1 - compatibility)       [B,100]
```

Weighted version:

```text
weighted_normative_deviation
= importance * normative_deviation        [B,100]
```

### 20.5 Token semantic map

Chỉ có trong optional token model:

```text
semantic_patch_map               [B,100,12,16]
```

Khi overlay lên stimulus image, phải resize map sau inference. Original image pixels không phải Stage-2 model input.

### 20.6 Cách xác định “stimulus quan trọng” đáng tin hơn

Không chỉ rank bằng attention. Nên báo cáo đồng thời:

1. mean absolute contribution theo subject;
2. attention stability qua folds/seeds;
3. leave-one-stimulus-out hoặc leave-one-category-out prediction change;
4. subject-level bootstrap confidence interval;
5. consistency giữa main head và auxiliary evidence head.

## 21. Loss modules

Mọi diagnostic BCE sử dụng:

```text
logits        [B]
labels        [B]
```

Không áp dụng HC/SZ BCE độc lập trên tensor `[B,100]`.

### 21.1 Main classification loss

Chạy aggregator trên full panel và hai category-stratified subsets `A`, `B`, mỗi subset giữ khoảng 50–80% valid stimuli:

\[
L_{cls}
=BCE(\ell^{full},y)
+\frac12
\left[
BCE(\ell^A,y)+BCE(\ell^B,y)
\right].
\]

Shapes:

```text
main logits full/A/B              [B]
labels                            [B]
L_cls                             scalar
```

Frozen heat encoder chỉ chạy một lần; subsets được tạo ở trial representation/aggregation level.

### 21.2 Auxiliary evidence loss

\[
L_{aux}=BCE(\ell^{aux},y).
\]

```text
auxiliary logits                  [B]
labels                            [B]
L_aux                             scalar
```

Loss này làm signed contributions có liên hệ trực tiếp với subject diagnosis.

### 21.3 Trial-bank matching loss

Chỉ dùng HC training trials. Với correct stimulus norm `n_pos` và wrong same-category stimulus norm `n_neg`:

\[
s^+=\cos(q^{match},n^{pos}),
\qquad
s^-=\cos(q^{match},n^{neg}).
\]

\[
L_{trialmatch}
=\operatorname{mean}_{HC}(1-s^+)
+\operatorname{mean}_{HC}
\max(0,m+s^- -s^+).
\]

```text
positive cosine                   [N_HC]
negative cosine                   [N_HC]
margin m                          0.2 mặc định
L_trialmatch                      scalar
```

Loss này dạy independent query/bank projections tạo comparable latent space.

### 21.4 Comparator bank-ranking loss

Chạy comparator với đúng bank và wrong-stimulus bank:

\[
L_{bankrank}
=\operatorname{mean}_{HC}
\max(0,m_b-r^+ +r^-).
\]

`r+` phải cao hơn `r-` ít nhất margin. Negative được lấy từ stimulus khác trong cùng category để tránh task quá dễ.

### 21.5 Optional token matching loss

\[
L_{tokenmatch}
=\operatorname{mean}_{HC,t}
\rho_{s,t}\omega_{i,s,t}
\left[
1-\cos(Q_t,N_t)
+\max(0,m_t+\cos(Q_t,N_{t^-})-\cos(Q_t,N_t))
\right].
\]

```text
token cosine                      [N_HC,192]
token reliability                 [N_HC,192]
L_tokenmatch                      scalar
```

### 21.6 Combined matching objective

Base:

\[
L_{match}=L_{trialmatch}+0.5L_{bankrank}.
\]

Token model:

\[
L_{match}
=L_{trialmatch}+0.5L_{bankrank}+0.25L_{tokenmatch}.
\]

### 21.7 Subset consistency loss

Latent consistency:

\[
L_{latent}
=\frac1B\sum_i
\left[1-\cos(u_i^A,u_i^B)\right].
\]

Probability consistency:

\[
L_{prob}
=JSD\left(
Bern(p^A),Bern(p^B)
\right).
\]

\[
L_{cons}=L_{latent}+L_{prob}.
\]

```text
subject embeddings A/B            [B,128]
probabilities A/B                 [B]
L_cons                            scalar
```

Mục tiêu là prediction không thay đổi quá mạnh khi chỉ quan sát một category-stratified subset hợp lệ.

### 21.8 Early attention anti-collapse loss

\[
L_{ent}
=\frac1B\sum_i
\max\left(0,H_{min}-H(I_i)\right).
\]

Loss chỉ active trong early epochs rồi anneal về 0. Nó ngăn attention sụp quá sớm vào một vài stimulus trước khi normative comparator học ổn định.

### 21.9 Optional encoder anchor loss

Chỉ dùng khi final encoder block được unfreeze:

\[
L_{anchor}
=\left\|\theta_h-\theta_h^{Stage1}\right\|_2^2.
\]

Frozen base model có:

```text
lambda_anchor = 0
L_anchor = differentiable zero hoặc zero scalar
```

### 21.10 Total loss

Starting configuration:

\[
L_{total}
=L_{cls}
+0.3L_{aux}
+0.1L_{match}
+0.1L_{cons}
+0.01L_{ent}
+\lambda_{anchor}L_{anchor}.
\]

Các weight là starting values, không phải universal optimum.

## 22. Training phases và trainable components

### 22.1 Stage 2A — HC bank-alignment warm-up

| Thành phần | Trạng thái |
|---|---|
| Data | outer-training HC subjects |
| Heatmap encoder | frozen |
| Normative bank | frozen |
| Query pooler/projection | trainable |
| Bank adapters/reliability | trainable |
| Relation encoder | trainable |
| Optional token branch | trainable nếu enabled |
| Subject classifier | không cần train trong phase này |
| Objective | `L_match` |
| Epoch khởi đầu | khoảng 5–15 |

### 22.2 Stage 2B — HC/SZ diagnostic training

| Thành phần | Trạng thái |
|---|---|
| Data | outer-training HC + SZ subjects |
| Heatmap encoder | frozen trong base |
| Normative bank | luôn frozen |
| Stage-2 comparison modules | trainable |
| Category attention | trainable |
| Subject Transformer | trainable |
| Main/auxiliary heads | trainable |
| Objective | `L_total` |

### 22.3 Optional final-block fine-tuning

```text
Stage-2 module learning rate       1e-4
encoder final-block learning rate  1e-5
lambda_anchor                      > 0
```

Đây là named ablation. Không dùng làm base mặc định.

## 23. End-to-end tensor example

Ví dụ:

```text
B = 4 subjects
valid trials = [100,98,100,88]
N = 386
```

| Bước | Tensor | Shape |
|---:|---|---:|
| 1 | Subject heatmaps | `[4,100,3,48,64]` |
| 2 | Trial mask | `[4,100]` |
| 3 | Flatten valid heatmaps | `[386,3,48,64]` |
| 4 | Heat tokens | `[386,192,128]` |
| 5 | Query patch attention | `[386,192]` |
| 6 | Query vector | `[386,128]` |
| 7 | Gathered bank mean/std | `[386,128]` mỗi tensor |
| 8 | Bank count | `[386]` |
| 9 | Bank mean/uncertainty adapters | `[386,128]` mỗi tensor |
| 10 | Reliability | `[386,1]` |
| 11 | Explicit relation | `[386,770]` |
| 12 | Base trial representation | `[386,128]` |
| 13 | Optional token relation | `[386,192,514]` |
| 14 | Optional serial-token output | `[386,192,128]` |
| 15 | Optional token pooled vector | `[386,128]` |
| 16 | Final flat trial representation | `[386,128]` |
| 17 | Scatter subject panel | `[4,100,128]` |
| 18 | Within-category attention | `[4,100]` |
| 19 | Four category tokens | `[4,4,128]` |
| 20 | Transformer sequence | `[4,5,128]` |
| 21 | Subject embedding | `[4,128]` |
| 22 | Main logits | `[4]` |
| 23 | Auxiliary logits | `[4]` |
| 24 | Stimulus contributions | `[4,100]` |
| 25 | Optional semantic patch map | `[4,100,12,16]` |

Missing positions trong `[4,100,...]` outputs phải bằng 0 hoặc masked value theo contract, luôn kèm `trial_mask`.

## 24. Recommended model configuration

```yaml
model:
  d_model: 128
  encoder_source: stage1_heatmap_encoder
  freeze_encoder: true
  query_pooling: attention
  query_pool_hidden: 64
  bank_mode: trial
  relation_hidden: 256
  attention_heads: 4
  category_attention_hidden: 64
  subject_transformer_layers: 1
  subject_transformer_ffn: 256
  dropout: 0.25
  auxiliary_evidence_head: true

  # Optional fused-token branch
  token_local_window: 3
  token_attention_layers: 2
  token_spatial_bridge: residual_dwconv_ffn

loss:
  lambda_aux: 0.3
  lambda_match: 0.1
  lambda_cons: 0.1
  lambda_entropy: 0.01
  lambda_anchor: 0.0
  match_margin: 0.2
  bank_rank_margin: 0.2
```

## 25. Forward output contract

Model nên trả typed output thay vì positional tuple:

```python
Stage2ForwardOutput(
    main_logit,                    # [B]
    auxiliary_logit,               # [B]
    subject_embedding,             # [B,128]
    trial_embeddings,              # [B,100,128]
    trial_mask,                    # [B,100]
    query_patch_attention,         # [B,100,192]
    stimulus_attention,            # [B,100]
    stimulus_importance,           # [B,100]
    stimulus_evidence,             # [B,100]
    stimulus_contribution,         # [B,100]
    semantic_compatibility,        # [B,100]
    normative_deviation,           # [B,100]
    weighted_normative_deviation,  # [B,100]
    semantic_patch_map,            # [B,100,12,16] or None
    diagnostics,                   # dict/dataclass
)
```

Loss function nên trả:

```python
Stage2LossOutput(
    total,
    cls,
    aux,
    match,
    trialmatch,
    bankrank,
    tokenmatch,
    cons,
    latent_cons,
    prob_cons,
    entropy,
    anchor,
    n_hc_match_trials,
    matched_cosine_mean,
    wrong_cosine_mean,
    bank_rank_accuracy,
)
```

## 26. Base model và optional modes

| Mode | Trial bank | Fused-token bank | Heat-token bank | Encoder |
|---|---:|---:|---:|---|
| `base` | bắt buộc | không | không | frozen |
| `trial_and_fused_token` | bắt buộc | bắt buộc | không | frozen mặc định |
| `same_space_heat_bank` | bắt buộc hoặc theo config | không bắt buộc | bắt buộc | phải frozen |
| `unfreeze_last_block` | bắt buộc | optional | không được direct same-space | final block trainable |
| `no_bank` ablation | neutralized | không | không | frozen |
| `wrong_stimulus_bank` | dùng permutation có kiểm soát | optional | optional | frozen |

## 27. Những lỗi implementation nghiêm trọng cần tránh

1. Dùng raw `q0 - mu_trial` dù hai tensor thuộc pre/post-fusion spaces khác nhau.
2. Dùng validation HC để xây bank của fold đó.
3. Cho training HC subject dùng bank chứa chính subject đó khi crossfit đã được bật.
4. Chọn bank row sai `stimulus_index`.
5. Encode zero-filled missing trial như trial thật.
6. Softmax attention mà không mask missing slots.
7. Dùng trial-level HC/SZ BCE `[B,100]`.
8. Cho token Attention 2 nhận `R0` thay vì spatial bridge output.
9. Share parameter giữa LocalMHA1 và LocalMHA2.
10. Gọi attention weight là causal explanation.
11. Dùng direct heat-token cosine khi encoder đã unfreeze hoặc checkpoint khác bank.
12. Cho normative bank tensor nhận gradient.
13. Fit threshold/calibration bằng outer held-out fold trong strict evaluation.
14. Báo cáo 15,912 trials như independent diagnostic sample size.

## 28. Unit-test checklist theo module

| Module | Test bắt buộc |
|---|---|
| Bank builder | contributor/forbidden IDs, shape, finite, checksum |
| Dataset | length bằng subject count, missing mask đúng |
| Encoder | exact checkpoint keys, output `[N,192,128]`, frozen gradient |
| Query pooler | attention sum bằng 1 trên 192 patches |
| Bank gather | đúng row cho từng stimulus index |
| Relation | concatenated dimension đúng 770 |
| Token relation | dimension đúng 514 |
| Local attention | window không wrap; weights `[N,4,192,9]` |
| Serial topology | MHA2 input bằng bridge output |
| Category attention | mỗi present category có tổng mass `1/K_i` |
| Subject Transformer | input `[B,5,128]`, mask category đúng |
| Evidence head | contributions sum đúng auxiliary logit minus bias |
| Loss | diagnostic logits/labels chỉ `[B]` |
| Missingness | missing slots không vào attention/loss |
| Resume/artifacts | bank/checkpoint/config hashes phải khớp |

## 29. Ý nghĩa khoa học của từng tầng

| Tầng | Câu hỏi mà tầng đó học trả lời |
|---|---|
| Heatmap encoder | Subject phân bố gaze ở đâu và theo cấu trúc spatial nào? |
| Query pooling | Patch gaze nào hữu ích cho normative comparison? |
| Normative bank | HC thường phản ứng thế nào với stimulus này? |
| Bank adapters | Làm thế nào align heat-only query với post-fusion HC representation? |
| Relation encoder | Subject lệch khỏi HC norm theo hướng, magnitude và reliability nào? |
| Token semantic branch | Region nào có local gaze–semantic normative mismatch? |
| Category attention | Trong từng loại stimulus, ảnh nào cung cấp evidence quan trọng? |
| Subject Transformer | Pattern giữa bốn loại stimulus kết hợp như thế nào ở cấp subject? |
| Main classifier | Tổng hợp nonlinear representation để phân loại HC/SZ ra sao? |
| Evidence head | Stimulus nào đẩy prediction về HC hoặc SZ và bao nhiêu? |

## 30. Kết luận

Architecture này phân tách ba loại information:

1. **Heatmap representation** từ transferred Stage-1 heat encoder;
2. **Normative deviation** từ comparison với HC bank của đúng stimulus;
3. **Subject-level diagnostic pattern** từ category-balanced attention và Transformer.

Base trial-bank model là cấu hình nên triển khai và kiểm thử trước. Optional fused-token branch bổ sung local semantic information nhưng cần token-bank artifact, nhiều memory hơn và validation nghiêm ngặt hơn.

Hai output để trả lời mục tiêu nghiên cứu là:

- **Stimulus nào quan trọng:** ưu tiên signed contribution, kiểm tra cùng attention stability và leave-one-out effect;
- **Semantic information:** learned normative compatibility ở base level và token semantic map ở optional token level.

Mọi kết luận cuối cùng vẫn phải được đánh giá bằng subject-level cross-validation, fold-safe normative bank và stability qua nhiều seed.
