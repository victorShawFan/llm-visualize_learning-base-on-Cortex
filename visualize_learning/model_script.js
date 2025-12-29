document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in model_script");
        return;
    }

    let currentStep = 0;

    const steps = [
        {
            title: "Model Init (初始化)",
            description: "LlmModel 初始化：创建 Embeddings, DecoderLayers 和 Head，并根据配置决定是否 <code>tie_word_embeddings</code>。权重整体初始化为 N(0, 0.02)。<span class='step-badge'>llm_model.py:477-499</span>",
            code: "self.embed_tokens = nn.Embedding(vocab_size, hidden)\nself.layers = ModuleList([DecoderLayer(...)] * num_layers)",
            activeId: "init",
            shape: "N/A",
            details: []
        },
        {
            title: "Step 1: Embedding & RoPE 输入",
            description: "输入 Token ID 先通过 <code>get_input_embeddings</code> 查表得到词向量；同时构造 RoPE 模块所需的输入形状，为后续 position_ids 和 cos/sin 做准备。<span class='step-badge'>llm_model.py:555</span>",
            code: "inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, **kwargs)",
            activeId: "embed",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Lookup Table", "VLM 可重载", "Hidden States"]
        },
        {
            title: "Step 2: KVCache & Position Ids",
            description: "推理时，根据 <code>use_cache</code> 创建/复用 KVCache，查询已生成长度 <code>past_seen_tokens</code>，然后从该偏移开始构造新的 <code>position_ids</code>，实现自回归续写。<span class='step-badge'>llm_model.py:550-560, kv_cache.py:30-55</span>",
            code: "if use_cache and past_key_values is None:\n    past_key_values = KVCache()\npast_seen_tokens = past_key_values.get_seq_len() if past_key_values is not None else 0\nfull_seq_len = past_seen_tokens + seq_len\nposition_ids = torch.arange(past_seen_tokens, full_seq_len, device=inputs_embeds.device).unsqueeze(0)",
            activeId: "kv_cache",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Create KVCache", "past_seen_tokens", "full_seq_len", "position_ids"]
        },
        {
            title: "Step 3: Causal / Pad / Doc Mask",
            description: "使用 <code>prepare_decoder_attention_mask</code> 将三类掩码合并为 <code>causal_mask</code>：自回归下三角掩码、padding 掩码以及文档边界掩码 <code>doc_boundary_mask</code>，得到形状为 (B, 1, Lq, Lk)。<span class='step-badge'>llm_model.py:569-576, attention_masks.py:108-172</span>",
            code: "causal_mask = prepare_decoder_attention_mask(\n    attention_mask=attention_mask,\n    doc_boundary_mask=doc_boundary_mask,\n    input_shape=(batch_size, seq_len),\n    past_key_values_length=past_seen_tokens,\n    dtype=inputs_embeds.dtype,\n    device=inputs_embeds.device,\n)",
            activeId: "mask",
            shape: "[Batch, 1, SeqLen, TotalLen]",
            details: ["Causal Mask", "Padding Mask", "Doc Boundary", "(B,1,Lq,Lk)"]
        },
        {
            title: "Step 4: Decoder Layer 内部 Pre-Norm",
            description: "在每一层 <code>DecoderLayer</code> 内部，首先对输入做 RMSNorm (Pre-Norm)，再进入 Self-Attention。这一步并不是单独的模块级 Norm，而是每个 Block 内部的第一步。<span class='step-badge'>llm_model.py:417, 440-441</span>",
            code: "residual = hidden_states\nhidden_states = self.attn(\n    hidden_states=self.attn_norm(hidden_states),\n    position_embeddings=position_embeddings,\n    attention_mask=causal_mask,\n    past_key_values=past_key_values,\n)",
            activeId: "norm1",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["RMSNorm", "Pre-Norm", "Per Layer"]
        },
        {
            title: "Step 5: Attention (QKV + RoPE)",
            description: "在 Attention 内部，线性投影得到 Q/K/V，reshape 成多头后按 head 维度拆分，利用 RoPE 对 Q/K 做旋转编码，再结合 <code>causal_mask</code> 做 SDPA / 手写注意力计算。<span class='step-badge'>llm_model.py:260-269, 319-335, 361-375</span>",
            code: "q, k, v = q_proj(x), k_proj(x), v_proj(x)\nq, k = apply_rotary_pos_emb(q, k, cos, sin)\nattn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)",
            activeId: "attn",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Q/K/V Proj", "RoPE Rotate", "Masking", "SDPA"]
        },
        {
            title: "Step 6: Residual Add 1",
            description: "Self-Attention 的输出通过 <code>o_proj</code> 映射回隐藏维度后，与残差 <code>residual</code> 做逐元素相加，形成 Block 的第一条残差路径。<span class='step-badge'>llm_model.py:382-447</span>",
            code: "attn_out = self.o_proj(attn)\nhidden_states = attn_out + residual",
            activeId: "res1",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Element-wise Add", "Residual 1"]
        },
        {
            title: "Step 7: MLP / MoE 前的 RMSNorm",
            description: "第二个 RMSNorm 发生在 FFN / MoE 之前：如果当前层配置为 MoE，则会进入 MoE 路由；否则进入普通 SwiGLU MLP。<span class='step-badge'>llm_model.py:419-431, 448-451</span>",
            code: "if isinstance(self.mlp, MoE):\n    mlp_states, aux_loss = self.mlp(self.mlp_norm(hidden_states))\nelse:\n    mlp_states = self.mlp(self.mlp_norm(hidden_states))",
            activeId: "norm2",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["RMSNorm 2", "MoE or MLP"]
        },
        {
            title: "Step 8: SwiGLU MLP / 稀疏 MoE",
            description: "Dense 层使用门控 MLP：<code>Down(SiLU(Gate(x)) * Up(x))</code>；若启用稀疏 MoE，则由门控网络选择 Top-K 专家，对每个 token 做加权聚合并返回 <code>aux_loss</code> 参与负载均衡。<span class='step-badge'>llm_model.py:88-122, sparse_moe.py:13-94, 114-166</span>",
            code: "gate = self.gate_proj(x)\nup = self.up_proj(x)\nout = self.down_proj(F.silu(gate) * up)  # 或 MoE(experts)",
            activeId: "mlp",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["SwiGLU", "Top-K Experts", "Aux Loss"]
        },
        {
            title: "Step 9: Residual Add 2",
            description: "FFN / MoE 输出与 Block 输入做第二条残差连接，形成标准的 Pre-Norm Transformer Block。<span class='step-badge'>llm_model.py:452-453</span>",
            code: "hidden_states = mlp_states + hidden_states",
            activeId: "res2",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Element-wise Add", "Residual 2"]
        },
        {
            title: "Step 10: 叠加多层 DecoderLayer",
            description: "上述 DecoderLayer 在 <code>self.layers</code> 中被堆叠 <code>num_hidden_layers</code> 次，同时收集所有启用 MoE 层的 <code>aux_loss</code>。这部分形成深层解码网络。<span class='step-badge'>llm_model.py:483-484, 579-587</span>",
            code: "aux_losses = ()\nfor layer in self.layers:\n    hidden_states, aux_loss = layer(...)\n    if aux_loss is not None:\n        aux_losses += (aux_loss,)",
            activeId: "layer_block",
            shape: "[Batch, SeqLen, Hidden]",
            details: ["Layer Stack", "MoE Aux Loss"]
        },
        {
            title: "Step 11: Final Norm & LM Head",
            description: "所有层之后进行一次 <code>head_norm</code>，再通过 <code>lm_head</code> 映射到词表维度。如果配置了 <code>tie_word_embeddings</code>，则共享权重实现更稳定的训练。<span class='step-badge'>llm_model.py:485-488, 589-591</span>",
            code: "hidden_states = self.head_norm(hidden_states)\nlogits = self.lm_head(hidden_states)",
            activeId: "head",
            shape: "[Batch, SeqLen, VocabSize]",
            details: ["Final RMSNorm", "Linear (Vocab)", "Tie Embeddings"]
        },
        {
            title: "Step 12: 输出给 Trainer / Loss",
            description: "前向返回包含 <code>logits</code>、<code>hidden_states</code>、<code>past_key_values</code> 与聚合后的 <code>aux_loss</code> 的字典。不同任务的 Trainer（SFT / DPO / PPO / GRPO）在各自的 Python 文件中消费这些字段并计算对应的损失。模型本身保持“纯前向”职责。<span class='step-badge'>llm_model.py:592-596, llm_trainer/loss.py</span>",
            code: "return {\n  'logits': logits,\n  'hidden_states': hidden_states,\n  'past_key_values': past_key_values,\n  'aux_loss': None if len(aux_losses)==0 else sum(aux_losses),\n}\n# 在各 Trainer 中：loss = loss_fn(outputs, labels)",
            activeId: "head",
            shape: "Dict",
            details: ["Logits/States", "KVCache", "Aux Loss", "Hook to Trainer"]
        }
    ];

    function updateUI() {
        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `
            <div class="step-badge">Step ${currentStep}</div> 
            <strong>${step.title}</strong>
            <div style="margin-top:5px; font-family:monospace; color:#666; font-size:0.9em;">Shape: ${step.shape}</div>
            <div style="margin-top:10px;">${step.description}</div>
        `;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if(window.hljs) hljs.highlightElement(codeSnippet);
        }
        try {
            renderVisualization();
        } catch(e) {
            console.error("Render failed", e);
        }
        updateButtons();
    }

    function renderVisualization() {
        visualContent.innerHTML = '';
        const container = document.createElement('div');
        container.className = 'model-flow';
        
        // Define the full structure
        const structure = [
            { id: "init", label: "Model Init", type: "block" },
            { id: "embed", label: "Embedding Layer", type: "block" },
            { id: "kv_cache", label: "KV Cache & Position Ids", type: "block" },
            { id: "mask", label: "Causal / Pad / Doc Mask", type: "block" },
            { id: "layer_block", label: "Decoder Layer (Stack)", type: "container", children: [
                { id: "norm1", label: "RMSNorm 1", type: "op" },
                { id: "attn", label: "Self Attention", type: "op" },
                { id: "res1", label: "Residual (+)", type: "add" },
                { id: "norm2", label: "RMSNorm 2", type: "op" },
                { id: "mlp", label: "SwiGLU / MoE", type: "op" },
                { id: "res2", label: "Residual (+)", type: "add" }
            ]},
            { id: "head", label: "Head (Norm + Linear)", type: "block" }
        ];

        const createNode = (def) => {
            const node = document.createElement('div');
            node.className = 'node ' + def.type;
            if (def.id === steps[currentStep].activeId) node.classList.add('active');
            node.innerText = def.label;
            
            // Show details if active
            if (def.id === steps[currentStep].activeId && steps[currentStep].details.length > 0) {
                const detailsDiv = document.createElement('div');
                detailsDiv.className = 'node-details';
                steps[currentStep].details.forEach(d => {
                    const badge = document.createElement('span');
                    badge.className = 'detail-badge';
                    badge.innerText = d;
                    detailsDiv.appendChild(badge);
                });
                node.appendChild(detailsDiv);
            }

            if (def.children) {
                def.children.forEach(child => {
                    node.appendChild(createNode(child));
                });
            }
            return node;
        };

        structure.forEach(item => {
            container.appendChild(createNode(item));
            const arrow = document.createElement('div');
            arrow.className = 'arrow';
            arrow.innerHTML = '↓';
            container.appendChild(arrow);
        });
        
        // Remove last arrow
        if (container.lastChild) container.removeChild(container.lastChild);

        visualContent.appendChild(container);
    }

    function updateButtons() {
        if(prevBtn) prevBtn.disabled = currentStep === 0;
        if(nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
    }

    function goNext() {
        if (currentStep < steps.length - 1) {
            currentStep++;
            updateUI();
        }
    }

    function goPrev() {
        if (currentStep > 0) {
            currentStep--;
            updateUI();
        }
    }

    if(nextBtn) nextBtn.addEventListener('click', goNext);
    if(prevBtn) prevBtn.addEventListener('click', goPrev);
    if(resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; updateUI(); });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    updateUI();
});
