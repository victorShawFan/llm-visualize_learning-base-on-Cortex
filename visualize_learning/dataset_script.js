document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const stepIndicator = document.getElementById('stepIndicator');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!prevBtn || !nextBtn || !resetBtn || !stepIndicator || !infoBox || !visualContent || !codeSnippet) {
        console.error("Required elements not found in Dataset script");
        return;
    }

    let currentStep = 0;
    let currentInterval = null;

    const steps = [
        {
            title: "Step 1: 数据源识别 (Data Loading Strategy)",
            description: "<code>dataset.py</code> 支持三种核心数据格式。根据后缀名自动分发处理逻辑：<br>1. <b>.npy</b>: 内存映射 (mmap)，零拷贝加载，适合超大规模预训练。<br>2. <b>.jsonl</b>: 流式读取或一次性加载，适合 SFT/RLHF 对话数据。<br>3. <b>.pkl</b>: Python 序列化对象。<span class='step-badge'>dataset.py:15-35</span>",
            code: `def _get_file_type(file_path):\n    if file_path.endswith('.npy'): return 'npy'\n    elif file_path.endswith('.jsonl'): return 'jsonl'`,
            render: () => renderFileDispatch()
        },
        {
            title: "Step 2: SFT 数据流 - 原始对话 (SFT Raw Input)",
            description: "对于 SFT，输入通常是 JSONL 格式的多轮对话。每一行包含 system, user, assistant 的角色和内容，对应 <code>SFTDataset</code> 注释中的结构示例。<span class='step-badge'>dataset.py:123-144</span>",
            code: `{"role": "user", "content": "Hello"}\n{"role": "assistant", "content": "Hi"}`,
            render: () => renderJsonlInput()
        },
        {
            title: "Step 3: 模版应用 (Chat Template)",
            description: "使用 <code>Tokenizer.apply_chat_template</code> 将结构化对话列表转换为带有角色标记的纯文本串。每一轮对话都会被包上 <code>&lt;system&gt;</code>/<code>&lt;user&gt;</code>/<code>&lt;assistant&gt;</code> 以及结尾的 <code>&lt;/s&gt;</code>，带思考的回复还会插入 <code>&lt;think&gt;</code> 与 <code>&lt;answer&gt;</code> 标记。该步骤与 <code>SFTDataset.__getitem__</code> 中的调用完全一致。<span class='step-badge'>dataset.py:187-190</span><span class='step-badge'>tokenizer.py:300-349</span>",
            code: `text = tokenizer.apply_chat_template(conversations)\n# "<system>你是一个助手。</s><user>你好？</s><assistant><think>先自我介绍</think><answer>我会聊天。</answer></s>"`,
            render: () => renderTemplate()
        },
        {
            title: "Step 4: Tokenization & Tensor",
            description: "将文本或数组编码为整数 Token ID：预训练场景通过 <code>TrainerTools().tokenizer.encode</code> 累积到 <code>input_ids</code>；SFT/RLHF 场景则在 <code>__getitem__</code> 中将列表/np.ndarray 转成 <code>torch.long</code>。<span class='step-badge'>dataset.py:69-82,193-197</span>",
            code: `input_ids = tokenizer.encode(text)\ninputs = torch.tensor(input_ids)`,
            render: () => renderTokenization()
        },
        {
            title: "Step 5: 标签掩码 (Label Masking)",
            description: "<b>关键步骤：</b> 在 SFT/DPO 中，我们只希望模型学习 Assistant 的可监督部分。因此：<code>&lt;system&gt;/&lt;user&gt;</code> 到 <code>&lt;/s&gt;</code> 之间全部被 Mask 掉；紧跟其后的 <code>&lt;assistant&gt;</code> 头部和 <code>&lt;answer&gt;</code> 起始标签也会被 Mask (-100)，而 <code>&lt;think&gt;</code> 块以及答案正文则保留正常 ID。上述规则由 <code>_mask_prompt</code> 实现。<span class='step-badge'>utils.py:183-221,474-533</span>",
            code: `labels = inputs.clone()\nlabels = _mask_prompt(labels)  # Mask 掉 prompt / 模版，只保留 &lt;think&gt; + 答案`,
            render: () => renderMasking()
        },
        {
            title: "Step 6: Pretrain 数据流 (Sliding Window)",
            description: "对于预训练 (PretrainDataset)，数据通常是连续的 token 流。<code>__len__</code> 与 <code>__getitem__</code> 使用滑动窗口 <code>[start_idx:end_idx]</code> 切分样本。步长 (stride) 决定了相邻样本的重叠程度。<span class='step-badge'>dataset.py:40-52,86-120</span>",
            code: `start_idx = item * stride\nend_idx = start_idx + block_size\ndata = self.input_ids[start_idx:end_idx]`,
            render: () => renderSlidingWindow()
        },
        {
            title: "Step 7: 多模态 Token 扩展 (VLM Token Expansion)",
            description: "<b>VLM 关键逻辑：</b> 当 SFT 样本包含图像时，<code>SFTDataset</code> 会调用 <code>repeat_image_tok</code>。它会找到序列中第一个 <code>&lt;image&gt;</code> 占位符，并将其替换为 <code>tokens_per_image</code> 个相同的 token（通常是 576），以匹配视觉特征提取后的序列长度。<span class='step-badge'>dataset.py:200-203</span><span class='step-badge'>utils.py:138-155</span>",
            code: `mask = (tokens == image_tok)\nidxs = torch.nonzero(mask)\nimage_tok_idx = idxs[0, 0].item()\nnew_tokens = torch.cat([tokens[:idx], repeat_toks, tokens[idx+1:]])`,
            render: () => renderSFTImageFlow()
        },
        {
            title: "Step 8: 序列打包与文档边界 (Sequence Packing)",
            description: "为了提高效率，预训练通常将多个短文档拼接在一起。为了防止『跨文档注意力』，<code>create_doc_boundary_mask</code> 会根据 <code>eot</code> (End of Text) 标记生成一个特殊的 mask，确保当前文档的 token 只能看到自己文档的内容。<span class='step-badge'>utils.py:32-91</span>",
            code: `is_eot = (input_ids == tokenizer.end)\ndoc_ids = torch.cumsum(is_eot, dim=-1)\nboundary_mask = query_doc_ids > key_doc_ids\nfinal_mask.masked_fill_(boundary_mask, -inf)`,
            render: () => renderDocBoundary()
        },
        {
            title: "Step 9: Position ID 重置 (Position ID Reset)",
            description: "在打包序列中，每个文档的 <code>position_ids</code> 应该从 0 开始重新计数。<code>generate_position_ids</code> 检测 <code>t-1</code> 是否为 EOT，若是则将当前位置重置为 0，否则累加。这保证了模型对每个独立文档的位置感知是一致的。<span class='step-badge'>utils.py:94-125</span>",
            code: `is_reset = (input_ids[:, t-1] == tokenizer.end)\nposition_ids[:, t] = torch.where(is_reset, 0, prev_pos + 1)`,
            render: () => renderPositionReset()
        },
        {
            title: "Step 10: 最终 Batch 堆叠 (DataLoader)",
            description: "DataLoader 将处理好的样本堆叠为 Batch Tensor <code>[B, Seq_Len]</code>。预训练阶段使用 <code>pretrain_collate_fn</code> 自动构造 <code>inputs</code> 和 <code>labels</code> (labels 默认是 inputs 的副本，padding 部分填 -100)。<span class='step-badge'>utils.py:170-180, trainer.py:436-444</span>",
            code: `loader = DataLoader(dataset, batch_size=4)\n# Yields: Tensor[B, Seq_Len]`,
            render: () => renderBatchStack()
        },
        {
            title: "实战模拟: 序列打包 (Sequence Packing Sim)",
            description: "<b>动手试一试：</b> 观察两个不同长度的文档 Doc A 和 Doc B 如何被“打包”进一个固定长度的 Sequence 中，并自动生成 block-diagonal Mask 和重置 Position ID。",
            code: `Doc A: "Hello world" (2 tokens)\nDoc B: "Cortex is fast" (3 tokens)\nSeq Len: 8`,
            render: () => renderPackingSim()
        }
    ];

    function updateUI() {
        if (currentInterval) {
            clearInterval(currentInterval);
            currentInterval = null;
        }

        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        stepIndicator.textContent = `Step ${currentStep + 1}/${steps.length}`;
        infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        codeSnippet.textContent = step.code;
        
        if (window.hljs) {
            hljs.highlightElement(codeSnippet);
        }

        visualContent.innerHTML = '';
        visualContent.className = 'fade-in';
        
        try {
            step.render();
        } catch(e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    // --- Renderers ---

    function renderPackingSim() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:20px; align-items:center; width:100%;">
                <div style="display:flex; gap:20px; width:100%; justify-content:center;">
                    <div class="doc-box" style="padding:10px; border:2px solid #2ecc71; border-radius:5px; background:#d5f5e3;">
                        <strong>Doc A</strong><br>
                        [101, 204] <br>
                        <span style="font-size:10px; color:#555;">(Len=2)</span>
                    </div>
                    <div class="doc-box" style="padding:10px; border:2px solid #3498db; border-radius:5px; background:#d6eaf8;">
                        <strong>Doc B</strong><br>
                        [305, 406, 507] <br>
                        <span style="font-size:10px; color:#555;">(Len=3)</span>
                    </div>
                </div>
                
                <div class="arrow">⬇ Pack (Seq Len = 8) ⬇</div>
                
                <div class="seq-container" style="display:grid; grid-template-columns:repeat(8, 1fr); gap:5px; width:100%; max-width:600px;">
                    <!-- Headers -->
                    <div class="h-cell">Index</div>
                    <div class="h-cell">0</div><div class="h-cell">1</div><div class="h-cell">2</div>
                    <div class="h-cell">3</div><div class="h-cell">4</div><div class="h-cell">5</div><div class="h-cell">6</div>
                    
                    <!-- Token IDs -->
                    <div class="row-label">Token</div>
                    <div class="cell" style="background:#d5f5e3;">101</div><div class="cell" style="background:#d5f5e3;">204</div><div class="cell eot">EOT</div>
                    <div class="cell" style="background:#d6eaf8;">305</div><div class="cell" style="background:#d6eaf8;">406</div><div class="cell" style="background:#d6eaf8;">507</div><div class="cell eot">EOT</div>
                    
                    <!-- Pos IDs -->
                    <div class="row-label">Pos ID</div>
                    <div class="cell">0</div><div class="cell">1</div><div class="cell">2</div>
                    <div class="cell reset">0</div><div class="cell">1</div><div class="cell">2</div><div class="cell">3</div>
                </div>
                
                <div style="margin-top:10px; font-weight:bold;">Attention Mask (Block Diagonal)</div>
                <canvas id="mask-canvas" width="200" height="200" style="border:1px solid #333;"></canvas>
            </div>
            
            <style>
                .h-cell { font-weight:bold; font-size:12px; text-align:center; color:#555; }
                .cell { border:1px solid #ccc; text-align:center; padding:5px; border-radius:4px; font-family:monospace; }
                .eot { background:#f5b7b1; border-color:#c0392b; color:#c0392b; font-weight:bold; }
                .reset { border:2px solid #e74c3c; color:#e74c3c; font-weight:bold; }
                .row-label { font-weight:bold; text-align:right; padding-right:10px; font-size:12px; align-self:center;}
            </style>
        `;
        
        // Draw Mask
        const canvas = document.getElementById('mask-canvas');
        if(!canvas) return;
        const ctx = canvas.getContext('2d');
        const s = 200 / 8; // 8 tokens
        
        // Logical groups: 
        // 0-2 (Doc A + EOT): indices 0,1,2
        // 3-6 (Doc B + EOT): indices 3,4,5,6
        // 7 (Pad/Empty): index 7 - Wait, 2+1 + 3+1 = 7 tokens used. 
        // Total len 8. Index 7 is Pad.
        
        // Let's correct grid above: indices 0..6 used. Index 7 is PAD.
        // Update the HTML grid to have 8 columns (0-7)
        // I need to update the grid HTML above slightly to include Pad at index 7.
        // Actually I rendered 0-6 in HTML logic implicitly? 
        // Let's re-check the HTML string... 
        // 0,1,2 -> A
        // 3,4,5,6 -> B
        // 7 -> ? Not rendered in Token row above properly.
        
        // Let me re-inject the HTML with correct 8 columns.
        const gridHTML = `
            <!-- Headers -->
            <div class="h-cell">Idx</div>
            <div class="h-cell">0</div><div class="h-cell">1</div><div class="h-cell">2</div>
            <div class="h-cell">3</div><div class="h-cell">4</div><div class="h-cell">5</div><div class="h-cell">6</div><div class="h-cell">7</div>
            
            <!-- Token IDs -->
            <div class="row-label">Tok</div>
            <div class="cell" style="background:#d5f5e3;">A1</div><div class="cell" style="background:#d5f5e3;">A2</div><div class="cell eot">EOT</div>
            <div class="cell" style="background:#d6eaf8;">B1</div><div class="cell" style="background:#d6eaf8;">B2</div><div class="cell" style="background:#d6eaf8;">B3</div><div class="cell eot">EOT</div><div class="cell pad">PAD</div>
            
            <!-- Pos IDs -->
            <div class="row-label">Pos</div>
            <div class="cell">0</div><div class="cell">1</div><div class="cell">2</div>
            <div class="cell reset">0</div><div class="cell">1</div><div class="cell">2</div><div class="cell">3</div><div class="cell pad">0</div>
        `;
        
        const gridContainer = document.querySelector('.seq-container');
        if(gridContainer) {
            gridContainer.style.gridTemplateColumns = 'repeat(9, 1fr)'; // Label + 8 cols
            gridContainer.innerHTML = gridHTML;
        }

        // Now Draw
        const groups = [0,0,0, 1,1,1,1, 2]; // 0=A, 1=B, 2=Pad
        
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1;
                
                if (groups[r] === 2 || groups[c] === 2) {
                    // Pad area -> Masked
                    ctx.fillStyle = "#ccc"; // Masked
                } else if (groups[r] === groups[c]) {
                    // Same group -> Visible
                    if (c <= r) { // Causal
                         ctx.fillStyle = groups[r] === 0 ? "#2ecc71" : "#3498db";
                    } else {
                         ctx.fillStyle = "#eee"; // Future masked
                    }
                } else {
                    // Cross group -> Masked
                    ctx.fillStyle = "#e74c3c"; // Blocked
                }
                
                ctx.fillRect(c*s, r*s, s, s);
                ctx.strokeRect(c*s, r*s, s, s);
            }
        }
    }

    // --- Renderers ---

    function renderFileDispatch() {
        visualContent.innerHTML = `
            <div style="display:flex; justify-content:space-around; width:100%; margin-top:20px;">
                <div class="file-icon" style="border-color:#3498db; color:#3498db;">
                    .npy
                    <div style="font-size:10px; margin-top:5px;">MMAP Load</div>
                </div>
                <div class="file-icon" style="border-color:#e67e22; color:#e67e22;">
                    .jsonl
                    <div style="font-size:10px; margin-top:5px;">Line Parsing</div>
                </div>
                <div class="file-icon" style="border-color:#9b59b6; color:#9b59b6;">
                    .pkl
                    <div style="font-size:10px; margin-top:5px;">Pickle Load</div>
                </div>
            </div>
            <div style="text-align:center; margin-top:30px; font-weight:bold; color:#555;">dataset.py: _get_file_type()</div>
        `;
    }

    function renderJsonlInput() {
        visualContent.innerHTML = `
            <div class="code-block-dark">
                <span style="color:#e67e22">"messages"</span>: [<br>
                &nbsp;&nbsp;{<span style="color:#f1c40f">"role"</span>: "user", <span style="color:#f1c40f">"content"</span>: "Hi"},<br>
                &nbsp;&nbsp;{<span style="color:#f1c40f">"role"</span>: "assistant", <span style="color:#f1c40f">"content"</span>: "Hello!"}<br>
                ]
            </div>
            <div class="arrow">↓</div>
            <div style="text-align:center;">SFTDataset.__getitem__</div>
        `;
    }

    function renderTemplate() {
        visualContent.innerHTML = `
            <div style="font-family:monospace; background:#fff; padding:20px; border:1px solid #ddd; border-radius:8px; width:80%;">
                &lt;system&gt;你是一个助手。&lt;/s&gt;<br>
                &lt;user&gt;你好？&lt;/s&gt;<br>
                &lt;assistant&gt;&lt;answer&gt;你好，我是助手。&lt;/answer&gt;&lt;/s&gt;<br>
                &lt;user&gt;你会什么？&lt;/s&gt;<br>
                &lt;assistant&gt;&lt;think&gt;先自我介绍&lt;/think&gt;&lt;answer&gt;我会聊天。&lt;/answer&gt;&lt;/s&gt;
            </div>
            <div style="margin-top:10px; font-size:12px; color:#666;">Tokenizer.apply_chat_template 拼接后的 chat_template 文本</div>
        `;
    }

    function renderTokenization() {
        const toks = [101, 582, 3922, 102, 101, 774, 8821, 102];
        let html = `<div style="display:flex; gap:5px; flex-wrap:wrap;">`;
        toks.forEach(t => {
            html += `<div class="tok-box">${t}</div>`;
        });
        html += `</div>`;
        visualContent.innerHTML = html;
    }

    function renderMasking() {
        const toks = [
            {t:"<user>", masked:true},
            {t:"question", masked:true},
            {t:"</s>", masked:true},
            {t:"<assistant>", masked:true},
            {t:"<think>", masked:false},
            {t:"reasoning", masked:false},
            {t:"</think>", masked:false},
            {t:"<answer>", masked:true},
            {t:"result", masked:false},
            {t:"</s>", masked:false}
        ];
        
        let html = `<div style="display:flex; gap:5px; flex-wrap:wrap; justify-content:center;">`;
        toks.forEach(t => {
            const style = t.masked ? "border:2px dashed #e74c3c; opacity:0.6;" : "border:2px solid #2ecc71;";
            html += `<div class="tok-box" style="${style}">${t.t}</div>`;
        });
        html += `</div><div style="margin-top:20px; text-align:center; font-size:12px; color:#666;">红色虚线 = 被 Mask (-100)，不计算 Loss</div>`;
        visualContent.innerHTML = html;
    }

    function renderSlidingWindow() {
        visualContent.innerHTML = `
            <div id="window-container" style="position:relative; width:300px; height:60px; display:flex; gap:5px; border:1px solid #ccc; padding:10px;">
                ${Array.from({length:10}).map((_,i) => `<div class="tok-box" style="width:20px; height:20px;">T${i}</div>`).join('')}
                <div id="slider" style="position:absolute; top:5px; left:5px; width:100px; height:45px; border:3px solid #3498db; background:rgba(52,152,219,0.2); transition: left 1s ease-in-out;"></div>
            </div>
            <div style="margin-top:20px; text-align:center;" id="slider-info">Stride: 2, Block: 4</div>
        `;
        let pos = 0;
        currentInterval = setInterval(() => {
            pos = (pos + 1) % 4;
            const slider = document.getElementById('slider');
            if(slider) slider.style.left = `${5 + pos * 50}px`;
            const info = document.getElementById('slider-info');
            if(info) info.innerText = `Step: ${pos}, Start Index: ${pos * 2}`;
        }, 2000);
    }

    function renderSFTImageFlow() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:20px; width:100%;">
                <div id="vlm-seq" style="display:flex; gap:5px;">
                    <div class="tok-box">User</div>
                    <div class="tok-box" id="img-tok" style="background:#fadbd8; transition: all 0.5s;">&lt;image&gt;</div>
                    <div class="tok-box">Hello</div>
                </div>
                <div id="expansion-hint" style="font-size:14px; color:#e74c3c; font-weight:bold; opacity:0;">Expanding...</div>
                <div id="vlm-expanded" style="display:flex; gap:2px; flex-wrap:wrap; width:200px; justify-content:center; opacity:0; transition: opacity 0.5s;">
                    ${Array.from({length:16}).map(() => `<div style="width:15px; height:15px; background:#fadbd8; border:1px solid #e74c3c;"></div>`).join('')}
                    <div style="font-size:10px;">... x576</div>
                </div>
            </div>
        `;
        setTimeout(() => {
            const imgTok = document.getElementById('img-tok');
            const hint = document.getElementById('expansion-hint');
            const expanded = document.getElementById('vlm-expanded');
            if(imgTok) {
                imgTok.style.transform = "scale(1.2)";
                hint.style.opacity = "1";
                setTimeout(() => {
                    expanded.style.opacity = "1";
                }, 500);
            }
        }, 1000);
    }

    function renderDocBoundary() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:10px; align-items:center;">
                <div style="display:flex; gap:2px;">
                    <div class="tok-box" style="background:#d5f5e3;">Doc A</div>
                    <div class="tok-box" style="background:#d5f5e3;">EOT</div>
                    <div class="tok-box" style="background:#d6eaf8;">Doc B</div>
                    <div class="tok-box" style="background:#d6eaf8;">EOT</div>
                </div>
                <div style="margin-top:10px; font-weight:bold;">Attention Mask Matrix</div>
                <div style="display:grid; grid-template-columns:repeat(4, 30px); gap:2px; border:1px solid #333; padding:2px;">
                    <div style="width:30px; height:30px; background:#2ecc71;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div>
                    <div style="width:30px; height:30px; background:#2ecc71;"></div><div style="width:30px; height:30px; background:#2ecc71;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div>
                    <div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#3498db;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div>
                    <div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#e74c3c;"></div><div style="width:30px; height:30px; background:#3498db;"></div><div style="width:30px; height:30px; background:#3498db;"></div>
                </div>
                <div style="font-size:10px; color:#666;">Green: Doc A, Blue: Doc B, Red: Masked (-inf)</div>
            </div>
        `;
    }

    function renderPositionReset() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:15px; align-items:center;">
                <div style="display:flex; gap:5px;">
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <div class="tok-box" style="background:#d5f5e3;">Hello</div>
                        <div style="font-size:10px; font-weight:bold;">Pos: 0</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <div class="tok-box" style="background:#d5f5e3;">EOT</div>
                        <div style="font-size:10px; font-weight:bold;">Pos: 1</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <div class="tok-box" style="background:#d6eaf8; border:2px solid #e74c3c;">World</div>
                        <div style="font-size:10px; font-weight:bold; color:#e74c3c;">Pos: 0</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <div class="tok-box" style="background:#d6eaf8;">!</div>
                        <div style="font-size:10px; font-weight:bold;">Pos: 1</div>
                    </div>
                </div>
                <div style="font-size:12px; color:#e74c3c; font-weight:bold;">检测到 EOT，Position ID 重置为 0</div>
            </div>
        `;
    }

    function renderBatchStack() {
        visualContent.innerHTML = `
            <div style="display:grid; grid-template-rows:repeat(4, 1fr); gap:10px; width:60%;">
                <div class="tensor-row">Batch 0: [ 10, 20, 30 ... ]</div>
                <div class="tensor-row">Batch 1: [ 44, 55, 66 ... ]</div>
                <div class="tensor-row">Batch 2: [ 12, 11, 10 ... ]</div>
                <div class="tensor-row">Batch 3: [ 99, 88, 77 ... ]</div>
            </div>
            <div style="margin-left:20px; font-weight:bold; color:#2c3e50; align-self:center;">
                Tensor Shape:<br>[4, Seq_Len]
            </div>
        `;
        visualContent.style.display = "flex";
        visualContent.style.justifyContent = "center";
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

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
