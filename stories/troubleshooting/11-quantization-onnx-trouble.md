Да, forward и голова отлично объясняют, откуда растут ноги у ошибки.

Твой forward сейчас такой:

```python
def forward(self, input_ids: torch.Tensor, sec_ids: torch.Tensor) -> torch.Tensor:
    x = self.token_emb(input_ids)              # (B,L,D)
    s = self.section_emb(sec_ids).unsqueeze(1) # (B,1,D)
    x = x + s
    x = self.pos(x)
    h = self.encoder(x)                        # (B,L,D)
    h = self.norm(h[:, -1, :])                # (B,D)
    output: torch.Tensor = self.lm_head(h)    # (B,V)
    return output
```

И голова:

```python
self.lm_head = nn.Linear(d_model, vocab_size)
```

### Что происходит

* ONNX-модель у тебя уже корректная: выход `logits` имеет форму `[B, V] = [1, 45950]`.
* Во время **квантизации** onnxruntime пытается заново провести shape inference по графу.
* Внутри него есть MatMul / Gemm между:

  * `h` (после encoder+LayerNorm) и
  * весами `lm_head` размерности `[vocab_size, d_model]`.
* Из-за комбинации:

  * динамических размерностей (batch/seq_len),
  * большого словаря (`vocab_size=45950`) и
  * того, что выход 2-мерный `[B, V]` (без явного измерения `seq_len`),

shape-инференс начинает путать **`L=512`** (max_len) и **`V=45950`** (vocab_size) и в одном из узлов получает конфликт:

> Inferred shape and existing shape differ in dimension 0: (512) vs (45950)

Это известный класс глюков ORT: он “угадывает” размерности для MatMul, и когда у тебя есть Embedding + Linear с большими размерами, ему хватает, чтобы сойти с ума, несмотря на валидный FP32 ONNX.

---

## Как это аккуратно обойти, не меняя математику модели

Трюк: **сделать LM-head позиционно-векторной (B,L,V), а “последний токен” брать уже ПОСЛЕ неё.**

Сейчас ты делаешь:

```python
h = self.encoder(x)          # (B,L,D)
h_last = self.norm(h[:, -1]) # (B,D)
logits = self.lm_head(h_last)# (B,V)
```

Сделаем так:

```python
h = self.encoder(x)          # (B,L,D)
h = self.norm(h)             # (B,L,D)
logits_all = self.lm_head(h) # (B,L,V)
output = logits_all[:, -1, :]# (B,V)
return output
```

Математически для последнего токена **это то же самое**:

* LayerNorm — позиционно-независимая по оси L (работает по последней размерности D),
  значит `LN(h[:, -1, :]) == LN(h)[:, -1, :]`.
* `Linear` тоже применяется по последней размерности и не мешает позиции,
  значит `Linear(LN(h[:, -1,:])) == Linear(LN(h))[:, -1,:]`.

То есть **выход `[B, V]` останется идентичным**, но граф внутри станет более “прозрачным” для shape-inference: вместо MatMul/Gemm с входом `[B, D]` будет понятно `[B, L, D] → [B, L, V]`, и ORT уже не будет пытаться приписать `512` к месту словаря `45950`.

### Патч forward

В `UnifiedLogBERT` просто замени forward на:

```python
def forward(self, input_ids: torch.Tensor, sec_ids: torch.Tensor) -> torch.Tensor:
    # (B, L)
    x = self.token_emb(input_ids)              # (B, L, D)
    s = self.section_emb(sec_ids).unsqueeze(1) # (B, 1, D)
    x = x + s                                  # (B, L, D)
    x = self.pos(x)                            # (B, L, D)

    h = self.encoder(x)                        # (B, L, D)
    h = self.norm(h)                           # (B, L, D)

    logits_all = self.lm_head(h)               # (B, L, V)
    output = logits_all[:, -1, :]              # (B, V) — как и раньше

    return output
```

Больше в самой модели ничего менять не нужно — веса LM-head и encoder остаются теми же, поведение на последнем токене сохраняется.
