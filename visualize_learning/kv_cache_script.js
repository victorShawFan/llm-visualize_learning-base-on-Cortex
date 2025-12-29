document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    const numHeadsInput = document.getElementById('numHeads');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet || !numHeadsInput) {
        console.error("Required elements not found in KV Cache script");
        return;
    }

    let currentStep = 0;
    let numHeads = 2;
    let headDim = 4;

    const steps = [
        {
            title: "步骤 0: 初始化 (Cache Initialization)",
            description: "<code>KVCache</code> 被初始化为包含 <code>key_cache</code> 和 <code>value_cache</code> 的空容器。在自回归生成开始前，缓存没有任何数据，模型处于 Prefill 准备状态。<span class='step-badge'>kv_cache.py:18-24</span>",
            code: `class KVCache:\n    def __init__(self):\n        self.key_cache = []  # [[B, H, T, D], ...]\n        self.value_cache = []`,
            state: "init"
        },
        {
            title: "步骤 1: Prefill 预填充 (Context Storage)",
            description: "<b>Prefill 阶段</b>：处理输入的 Prompt。计算所有 token 的 KV 状态并一次性存入对应层的缓存。如果该层尚未建立索引，则动态补齐列表并赋值首个张量。<span class='step-badge'>kv_cache.py:69-77</span>",
            code: `if len(self.key_cache) <= layer_idx:\n    self.key_cache.append(key_states)\n    self.value_cache.append(value_states)`,
            state: "prefill"
        },
        {
            title: "步骤 2: 增量生成 - Token 1 (Decode Step)",
            description: "<b>Decode 阶段</b>：生成新 token。由于历史信息已缓存，模型只需计算当前 token 的 KV 值，并使用 <code>torch.cat</code> 沿 <code>seq_len</code> 维度 (-2) 拼接到旧张量之后。<span class='step-badge'>kv_cache.py:83-87</span>",
            code: `self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], k_new), dim=-2)`,
            state: "decode_1"
        },
        {
            title: "步骤 3: 循环增长 (Continuous Growth)",
            description: "随着生成持续，缓存 Tensor 在时间轴上平滑增长。<code>get_seq_len</code> 随时返回当前已计算的长度，确保 Attention Mask 能正确对齐所有历史位置。<span class='step-badge'>kv_cache.py:30-55</span>",
            code: `return self.key_cache[layer_idx].shape[-2] # 返回 T`,
            state: "decode_2"
        },
        {
            title: "步骤 4: 张量维度与内存估算 (Memory Layout)",
            description: "KV Cache 物理形状为 <code>[Batch, Heads, Seq, Dim]</code>。由于 K 和 V 是两套独立缓存，总参数量翻倍。随着 <code>Seq</code> 线性增加，显存开销是长文本推理的主要瓶颈。",
            code: `total_params = B * H * Seq * Dim * 2 # K and V`,
            state: "layout"
        },
        {
            title: "步骤 5: 多层协同 (Multi-layer Coordination)",
            description: "Transformer 每一层都维护独立的 KV 缓存。前向传播时，每一层都会读取自己的旧缓存并存入新 KV，确保跨层信息流的完整性与推理加速。",
            code: `for i in range(num_layers):\n    k_out, v_out = cache.update(k_i, v_i, i)`,
            state: "layers"
        }
    ];

    function updateUI() {
        const step = steps[currentStep];
        infoBox.innerHTML = `<div class="step-badge">Inference Optimization</div><strong>${step.title}</strong><p style="margin-top:10px;">${step.description}</p>`;
        codeSnippet.textContent = step.code;
        
        if (window.hljs) {
            hljs.highlightElement(codeSnippet);
        }

        try {
            render(step.state);
        } catch(e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    function render(state) {
        visualContent.innerHTML = '';
        const createTokenBlock = (txt, color='#e6fffa', border='#38b2ac') => {
            const d = document.createElement('div');
            d.style.cssText = `padding:6px; background:${color}; border:1px solid ${border}; border-radius:4px; font-size:11px; text-align:center; min-width:40px;`;
            d.innerText = txt;
            return d;
        };

        if (state === "init") {
            visualContent.innerHTML = `<div class="tensor-container" style="color:#aaa; border-style:dashed;">[ Empty Cache ]</div>`;
        } 
        else if (state === "prefill" || state === "decode_1" || state === "decode_2") {
            const h_cont = document.createElement('div');
            h_cont.style.display='flex'; h_cont.style.gap='20px';
            const tokens = state === "prefill" ? ['P1', 'P2', 'P3'] : 
                        (state === "decode_1" ? ['P1', 'P2', 'P3', 'D1'] : ['P1', 'P2', 'P3', 'D1', 'D2']);
            
            for(let h=0; h<numHeads; h++) {
                const card = document.createElement('div');
                card.className = "tensor-container";
                card.innerHTML = `<div style="font-size:10px; margin-bottom:10px;">Head ${h} KV</div>`;
                const list = document.createElement('div');
                list.style.display='flex'; list.style.flexDirection='column'; list.style.gap='2px';
                tokens.forEach((t, i) => {
                    const isNew = i === tokens.length - 1 && state.startsWith('decode');
                    list.appendChild(createTokenBlock(t, isNew?'#fff5f5':'#e6fffa', isNew?'#e53e3e':'#38b2ac'));
                });
                card.appendChild(list);
                h_cont.appendChild(card);
            }
            visualContent.appendChild(h_cont);
        }
        else if (state === "layout") {
            const s = 10;
            const total = 1 * numHeads * s * headDim * 2 * 2; // bytes
            visualContent.innerHTML = `
                <div class="tensor-container" style="width:100%; text-align:left;">
                    <div style="font-family:monospace; font-size:12px;">
                        Shape: [Batch:1, Heads:${numHeads}, Seq:${s}, Dim:${headDim}]<br>
                        Data Type: FP16 (2 bytes/param)<br>
                        Est. Memory: ${total} Bytes per Layer
                    </div>
                    <div style="display:flex; gap:10px; margin-top:20px;">
                        <div style="flex:1; border:2px solid #3182ce; padding:10px; border-radius:8px;"><strong>Key Tensor</strong></div>
                        <div style="flex:1; border:2px solid #e53e3e; padding:10px; border-radius:8px;"><strong>Value Tensor</strong></div>
                    </div>
                </div>
            `;
        }
        else if (state === "layers") {
            const stack = document.createElement('div');
            stack.style.display='flex'; stack.style.flexDirection='column-reverse'; stack.style.gap='5px';
            for(let l=0; l<4; l++) {
                const layer = document.createElement('div');
                layer.className = 'tensor-container';
                layer.style.padding='10px'; layer.style.background = l===0 ? '#ebf8ff' : '#f7fafc';
                layer.innerHTML = `Layer ${l} KV Store`;
                stack.appendChild(layer);
            }
            visualContent.appendChild(stack);
        }
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

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
    }

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentStep = 0;
            updateUI();
        });
    }

    if (numHeadsInput) {
        numHeadsInput.addEventListener('change', (e) => { 
            numHeads = parseInt(e.target.value) || 2; 
            updateUI(); 
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
