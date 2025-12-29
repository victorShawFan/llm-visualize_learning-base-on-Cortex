document.addEventListener('DOMContentLoaded', () => {
    const visualContent = document.getElementById('visualContent');
    const infoBox = document.getElementById('infoBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const nextBtn = document.getElementById('nextStep');
    const prevBtn = document.getElementById('prevStep');
    const resetBtn = document.getElementById('reset');
    const currentStepSpan = document.getElementById('current-step');

    if (!visualContent || !infoBox || !nextBtn || !prevBtn || !resetBtn || !currentStepSpan) {
        console.error("Critical DOM elements missing");
        return;
    }

    let currentStep = 0;

    const batchData = [
        { text: "Hello world", ids: [101, 200, 300], labels: [200, 300, 1] }, // 1 is EOS
        { text: "AI is good", ids: [102, 400, 500], labels: [400, 500, 1] },
        { text: "Code is art", ids: [103, 400, 600], labels: [400, 600, 1] }
    ];

    const steps = [
        {
            title: "Input Data (Batching)",
            desc: "SFT 训练的第一步是准备数据。我们将 3 个简单的句子组成一个 Batch。输入是文本，目标是预测下一个 Token。",
            code: "batch = ['Hello world', 'AI is good', 'Code is art']",
            render: () => {
                let html = `<div class='layer-box active anim-slide-up'><h3>Input Batch (B=3)</h3>`;
                batchData.forEach((item, idx) => {
                    html += `<div class='data-stream anim-slide-left' style='animation-delay: ${idx * 0.2}s'>
                        <span style='width:30px;font-weight:bold'>#${idx}</span>
                        <div class='tensor-block'>${item.text}</div>
                    </div>`;
                });
                html += `</div>`;
                return html;
            }
        },
        {
            title: "Tokenizer (Text to IDs)",
            desc: "模型无法直接理解文本，需要通过 Tokenizer 将文本转换为整数 ID 序列。这里我们简化展示每个词对应一个 ID。",
            code: "input_ids = tokenizer(batch, padding=True, return_tensors='pt')",
            render: () => {
                let html = `<div class='layer-box anim-fade'><h3>Tokenizer Output</h3>`;
                batchData.forEach((item, idx) => {
                    html += `<div class='data-stream anim-scale' style='animation-delay: ${idx * 0.15}s'>
                        <span style='width:30px;font-weight:bold'>#${idx}</span>
                        <div class='tensor-block active'>IDs: [${item.ids.join(', ')}]</div>
                    </div>`;
                });
                html += `</div>`;
                return html;
            }
        },
        {
            title: "Embedding Layer",
            desc: "每个 Token ID 被查找（Lookup）为一个高维向量（Hidden State）。假设 Hidden Size = 4。",
            code: "x = self.tok_embeddings(input_ids) # [Batch, Seq, Hidden]",
            render: () => {
                let html = `<div class='layer-box anim-slide-up'><h3>Embedding Lookups</h3>`;
                html += `<div class='math-formula'>x.shape = [3, 3, 4]</div>`;
                batchData.forEach((item, idx) => {
                    html += `<div class='data-stream anim-slide-up' style='animation-delay: ${idx * 0.1}s'>
                        <span style='width:30px;font-weight:bold'>#${idx}</span>
                        <div class='tensor-block'>[0.1, -0.2, ...]</div>
                        <div class='tensor-block'>[0.9, 0.5, ...]</div>
                        <div class='tensor-block'>[-0.3, 0.1, ...]</div>
                    </div>`;
                });
                html += `</div>`;
                return html;
            }
        },
        {
            title: "Enter Transformer Blocks",
            desc: "嵌入向量进入 Transformer 层堆叠（Layers）。每一层都包含 Attention 和 FFN。我们以第一层为例。",
            code: "for layer in self.layers:\n    x = layer(x, freqs_cis, mask)",
            render: () => {
                return `<div class='layer-box active anim-scale'>
                    <h3>Transformer Layer 0</h3>
                    <p>Input x shape: [3, 3, 4]</p>
                    <div class='arrow-flow'>⬇️</div>
                </div>`;
            }
        },
        {
            title: "Layer Norm (Pre-Attention)",
            desc: "在进入 Attention 之前，先对输入进行 RMSNorm 归一化，保证训练稳定性。",
            code: "h = self.attention_norm(x)",
            render: () => {
                return `<div class='layer-box anim-fade'>
                    <h3>RMSNorm</h3>
                    <div class='math-formula'>h = x * w_norm</div>
                    <div class='tensor-block active anim-pulse'>Normalized Variance ≈ 1.0</div>
                </div>`;
            }
        },
        {
            title: "QKV Projection",
            desc: "输入 h 被投影为 Query, Key, Value 三个向量。这里我们展示多头注意力的拆分。",
            code: "xq, xk, xv = self.attention.wq(h), self.attention.wk(h), self.attention.wv(h)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Linear Projection</h3>
                    <div style='display:flex; gap:10px; justify-content:center'>
                        <div class='tensor-block active anim-slide-left' style='animation-delay:0s'>Q [3, 3, 4]</div>
                        <div class='tensor-block active anim-slide-up' style='animation-delay:0.2s'>K [3, 3, 4]</div>
                        <div class='tensor-block active anim-slide-left' style='animation-delay:0.4s; transform: scaleX(-1);'>V [3, 3, 4]</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "RoPE Position Embedding",
            desc: "对 Q 和 K 应用旋转位置编码，注入绝对位置信息。",
            code: "xq, xk = apply_rotary_emb(xq, xk, freqs_cis)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Rotary Positional Embeddings</h3>
                    <div class='math-formula'>Rotate(Q, m), Rotate(K, m)</div>
                    <div style='display:flex; justify-content:center; gap:20px; margin-top:15px;'>
                        <div style='animation: rotateSpin 4s linear infinite; width:40px; height:40px; border:2px dashed #3498db; border-radius:50%; display:flex; align-items:center; justify-content:center;'>Q</div>
                        <div style='animation: rotateSpin 4s linear infinite reverse; width:40px; height:40px; border:2px dashed #e74c3c; border-radius:50%; display:flex; align-items:center; justify-content:center;'>K</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Attention Scores (Q @ K.T)",
            desc: "计算 Q 和 K 的点积，得到注意力分数矩阵。形状为 [Batch, Seq, Seq]。",
            code: "scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(dim)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Dot Product Attention</h3>
                    <div class='mask-grid anim-scale' style='display:grid; grid-template-columns:repeat(3, 20px); gap:2px; margin:10px auto; width:fit-content'>
                        <div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0s'></div><div style='background:#eee;height:20px' class='anim-fade' style='animation-delay:0.1s'></div><div style='background:#eee;height:20px' class='anim-fade' style='animation-delay:0.2s'></div>
                        <div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0.3s'></div><div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0.4s'></div><div style='background:#eee;height:20px' class='anim-fade' style='animation-delay:0.5s'></div>
                        <div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0.6s'></div><div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0.7s'></div><div style='background:#f1c40f;height:20px' class='anim-fade' style='animation-delay:0.8s'></div>
                    </div>
                    <p>Raw Affinity Scores</p>
                </div>`;
            }
        },
        {
            title: "Causal Masking",
            desc: "应用因果掩码（Causal Mask），将未来位置（右上三角）的分数设为 -inf，防止偷看答案。",
            code: "scores = scores + mask  # mask is -inf at upper triangle",
            render: () => {
                return `<div class='layer-box active'>
                    <h3>Causal Mask Applied</h3>
                    <div class='mask-grid' style='display:grid; grid-template-columns:repeat(3, 20px); gap:2px; margin:10px auto; width:fit-content'>
                        <div style='background:#2ecc71;height:20px'></div><div style='background:#333;height:20px;opacity:0;animation:fadeIn 1s forwards'></div><div style='background:#333;height:20px;opacity:0;animation:fadeIn 1s forwards'></div>
                        <div style='background:#2ecc71;height:20px'></div><div style='background:#2ecc71;height:20px'></div><div style='background:#333;height:20px;opacity:0;animation:fadeIn 1s forwards'></div>
                        <div style='background:#2ecc71;height:20px'></div><div style='background:#2ecc71;height:20px'></div><div style='background:#2ecc71;height:20px'></div>
                    </div>
                    <p>Dark cells = -inf (Masked)</p>
                </div>`;
            }
        },
        {
            title: "Softmax",
            desc: "对分数进行 Softmax 归一化，得到概率分布（Attention Weights）。",
            code: "scores = F.softmax(scores.float(), dim=-1)",
            render: () => {
                return `<div class='layer-box anim-slide-up'>
                    <h3>Softmax Probability</h3>
                    <div style='font-family:monospace; text-align:left; display:inline-block'>
                        [1.0, 0.0, 0.0]<br>
                        [0.4, 0.6, 0.0]<br>
                        [0.2, 0.3, 0.5]
                    </div>
                </div>`;
            }
        },
        {
            title: "Weighted Sum (Scores @ V)",
            desc: "用权重对 Value 向量进行加权求和，得到 Context 向量。",
            code: "output = torch.matmul(scores, xv)",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Context Aggregation</h3>
                    <div class='math-formula'>Sum(Weight_i * V_i)</div>
                    <div class='tensor-block active anim-pulse'>Output [3, 3, 4]</div>
                </div>`;
            }
        },
        {
            title: "Output Projection (Wo)",
            desc: "将多头注意力的输出拼接并投影回模型维度。",
            code: "output = self.attention.wo(output)",
            render: () => {
                return `<div class='layer-box anim-fade'>
                    <h3>Output Linear</h3>
                    <div class='math-formula'>Project back to hidden_dim</div>
                </div>`;
            }
        },
        {
            title: "Residual Connection 1",
            desc: "将 Attention 的输出加回到原始输入 x 上（残差连接），缓解梯度消失。",
            code: "h = x + output",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Residual Add</h3>
                    <div class='math-formula'>x_new = x_old + attn_out</div>
                    <div class='arrow-flow'>⬇️</div>
                </div>`;
            }
        },
        {
            title: "Layer Norm (Pre-FFN)",
            desc: "在进入 FFN 之前，再次进行 RMSNorm。",
            code: "h_norm = self.ffn_norm(h)",
            render: () => {
                return `<div class='layer-box anim-fade'>
                    <h3>RMSNorm</h3>
                    <p>Prepare for FFN</p>
                </div>`;
            }
        },
        {
            title: "Feed Forward Network (FFN)",
            desc: "SwiGLU 结构：包含 Gate, Up, Down 三个投影。扩大维度再缩小，引入非线性。",
            code: "output = self.feed_forward(h_norm)",
            render: () => {
                return `<div class='layer-box active'>
                    <h3>FFN (SwiGLU)</h3>
                    <div class='math-formula'>down(act(gate(h)) * up(h))</div>
                    <div style='display:flex; justify-content:center; gap:5px; margin-top:10px;'>
                        <div class='tensor-block anim-expand' style='height:20px; animation-delay:0s'>4</div>
                        <div class='arrow-right'>→</div>
                        <div class='tensor-block anim-expand' style='height:40px; background:#e67e22; animation-delay:0.2s'>16</div>
                        <div class='arrow-right'>→</div>
                        <div class='tensor-block anim-expand' style='height:20px; animation-delay:0.4s'>4</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Residual Connection 2",
            desc: "将 FFN 的输出加回，完成这一层的计算。",
            code: "out = h + output",
            render: () => {
                return `<div class='layer-box'>
                    <h3>Residual Add</h3>
                    <div class='math-formula'>x_final = x_prev + ffn_out</div>
                    <p>End of Layer 0</p>
                </div>`;
            }
        },
        {
            title: "Final RMSNorm",
            desc: "穿过所有层后，进行最后一次归一化。",
            code: "out = self.norm(out)",
            render: () => {
                return `<div class='layer-box anim-fade'>
                    <h3>Final RMSNorm</h3>
                    <p>Stabilize final features</p>
                </div>`;
            }
        },
        {
            title: "Output Head (Unembed)",
            desc: "将隐向量投影到词表大小（Vocab Size），得到 Logits。",
            code: "logits = self.output(out) # [3, 3, VocabSize]",
            render: () => {
                return `<div class='layer-box active anim-scale'>
                    <h3>Output Head</h3>
                    <div class='math-formula'>Logits [3, 3, 32000]</div>
                    <p>Scores for every possible token</p>
                </div>`;
            }
        },
        {
            title: "Loss Calculation: Shift Labels",
            desc: "计算 Loss 时，我们将输入向左移一位作为 Label（预测下一个词）。",
            code: "shift_logits = logits[..., :-1, :]\nshift_labels = labels[..., 1:]",
            render: () => {
                return `<div class='layer-box anim-slide-left'>
                    <h3>Shift & Align</h3>
                    <div class='data-stream'>
                        <div>Pred: [Hello, world]</div>
                        <div class='anim-slide-left' style='color:red; margin-left:10px;'>← Shift</div>
                    </div>
                    <div class='data-stream'>
                        <div>Target: [world, EOS]</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Cross Entropy Loss",
            desc: "计算预测概率与真实标签之间的交叉熵损失。",
            code: "loss = F.cross_entropy(shift_logits, shift_labels)",
            render: () => {
                return `<div class='layer-box active anim-pulse' style='border-color:#e74c3c'>
                    <h3>🔥 Loss = <span id='lossValue'>2.45</span></h3>
                    <p>Measure of error</p>
                </div>`;
            }
        },
        {
            title: "Backward Pass",
            desc: "反向传播：计算 Loss 对所有可训练参数的梯度（Gradients）。",
            code: "loss.backward()",
            render: () => {
                return `<div class='layer-box active backward-flow' style='background:#fadbd8'>
                    <h3>Gradient Flow ⬆️</h3>
                    <p>Head → Layers → Embeddings</p>
                    <div style='font-size:24px; animation: slideInUp 0.5s infinite reverse;'>⬆️</div>
                </div>`;
            }
        },
        {
            title: "Optimizer Step",
            desc: "优化器根据梯度更新权重，使 Loss 下降。",
            code: "optimizer.step()\noptimizer.zero_grad()",
            render: () => {
                return `<div class='layer-box active anim-pulse' style='background:#d5f5e3'>
                    <h3>Weights Updated</h3>
                    <div class='math-formula'>W = W - lr * grad</div>
                    <p>Model learned a tiny bit!</p>
                </div>`;
            }
        }
    ];

    function renderStep(index) {
        if (index < 0) index = 0;
        if (index >= steps.length) index = steps.length - 1;
        
        currentStep = index;
        const stepData = steps[index];

        // Update UI
        currentStepSpan.textContent = currentStep + 1;
        infoBox.innerHTML = `<h3>${stepData.title}</h3><p>${stepData.desc}</p>`;
        codeSnippet.textContent = stepData.code || "";
        visualContent.innerHTML = stepData.render();

        // Specific JS animations if needed
        if (stepData.title.includes("Loss")) {
            // Animate loss value slightly
            const lossSpan = document.getElementById('lossValue');
            if(lossSpan) {
                let val = 2.45;
                const interval = setInterval(() => {
                    val = 2.40 + Math.random() * 0.1;
                    lossSpan.textContent = val.toFixed(2);
                }, 200);
                // Clear interval when leaving step? Simplify by just letting it run briefly or attached to element
                // Note: setInterval will persist if not cleared. But we replace innerHTML on next step.
                // However, the interval timer itself keeps running in memory. We should clear it.
                if (window.lossInterval) clearInterval(window.lossInterval);
                window.lossInterval = interval;
            }
        } else {
             if (window.lossInterval) {
                 clearInterval(window.lossInterval);
                 window.lossInterval = null;
             }
        }

        // Update buttons
        prevBtn.disabled = currentStep === 0;
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    // Event Listeners
    nextBtn.addEventListener('click', () => renderStep(currentStep + 1));
    prevBtn.addEventListener('click', () => renderStep(currentStep - 1));
    resetBtn.addEventListener('click', () => renderStep(0));

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') renderStep(currentStep - 1);
        if (e.key === 'ArrowRight') renderStep(currentStep + 1);
    });

    // Initial render
    renderStep(0);
});
