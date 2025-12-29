document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    
    // Guard: Ensure essential elements exist
    if (!visualContent) {
        console.error("Required elements not found in attn_script");
        return;
    }

    let currentStep = 0;
    
    // 按照副本中的细腻风格，细化为 11 个阶段
    const steps = [
        {
            title: '步骤 0: 输入隐藏状态 (Hidden States)',
            description:
              'Attention 的输入是上一层输出的隐藏状态 <code>hidden_states</code>，形状为 <code>[B, L, D]</code>。每一个 Token 向量都包含了该位置在经过之前层处理后的丰富语义。<span class="step-badge">llm_model.py:309-311</span>',
            code: 'batch, seq_len, _ = hidden_states.shape',
            state: 'init',
        },
        {
            title: '步骤 1: 线性投影得到 Q/K/V',
            description:
              '使用三组线性层将输入分别映射到 Query / Key / Value 空间。这一步是特征提取的过程：Q 代表“我要找什么”，K 代表“我有什么特征”，V 代表“我可以提供什么信息”。<span class="step-badge">llm_model.py:311-321</span>',
            code: 'query_states  = self.q_proj(hidden_states)\nkey_states    = self.k_proj(hidden_states)\nvalue_states  = self.v_proj(hidden_states)',
            state: 'proj',
        },
        {
            title: '步骤 2: QK Norm (可选归一化)',
            description:
              '在某些先进模型中（如 DeepSeek），会在此处对每个 Head 的 Q 和 K 独立应用 RMSNorm。这能增强大规模训练的数值稳定性，防止 Attention 分数出现极端值。<span class="step-badge">llm_model.py:322-324</span>',
            code: 'if self.use_qk_norm:\n    query_states = self.q_norm(query_states)\n    key_states = self.k_norm(key_states)',
            state: 'qknorm',
        },
        {
            title: '步骤 3: 重塑并分头 (Split & Permute)',
            description:
              '将长向量 <code>D</code> 拆分为 <code>n_heads × d_head</code>。通过 <code>permute</code> 将 Head 维度提前，使得后续计算可以对所有 Head 并行处理：<code>[B, H, L, D_h]</code>。<span class="step-badge">llm_model.py:328-330</span>',
            code: 'query_states = query_states.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)',
            state: 'reshape',
        },
        {
            title: '步骤 4: 应用 RoPE 位置编码',
            description:
              '对 Q 和 K 的每一对分量进行旋转。RoPE 通过复数乘法在不增加额外参数的情况下，将绝对位置信息转化为相对位置信息，使模型能感知距离。<span class="step-badge">llm_model.py:334</span>',
            code: 'query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)',
            state: 'rope',
        },
        {
            title: '步骤 5: GQA 头部扩展 (Grouped Query Attention)',
            description:
              '当 K/V 头数少于 Q 头数时，通过在 group 维度上 <code>expand</code> 复制 K/V。例如 8 个 Q 头共享 1 个 K 头，极大地节省了推理时的 KV Cache 内存开销。<span class="step-badge">llm_model.py:337-357</span>',
            code: 'key_states = key_states[:, :, None, :, :].expand(B, n_kv, n_groups, L, D_h).reshape(B, n_heads, L, D_h)',
            state: 'gqa',
        },
        {
            title: '步骤 5.5: 小白速算 (Score Calculation Example)',
            description:
              '暂停一下！我们用简单的数字来看看 Score 是怎么算出来的。<br>假设 Q=[1, 0] (找狗), K1=[1, 0] (是狗), K2=[0, 1] (是猫)。<br>Score1 = 1*1 + 0*0 = 1 (高分)<br>Score2 = 1*0 + 0*1 = 0 (低分)<br><b>结论：向量越相似，点积越大。</b>',
            code: 'Score = q · k = q[0]*k[0] + q[1]*k[1] + ...',
            state: 'score_example',
        },
        {
            title: '步骤 6: 计算注意力分数 (Matrix Multiplication)',
            description:
              '计算 <code>Q @ K<sup>T</sup></code>。这一步衡量了 Sequence 中每两个 Token 之间的关联程度。分数越高，表示 Query 所在的 Token 越关注 Key 所在的 Token。<span class="step-badge">llm_model.py:371</span>',
            code: 'attn_scores = (self.scale * query_states) @ key_states.transpose(-1, -2)',
            state: 'score',
        },
        {
            title: '步骤 7: Causal Masking (因果掩码)',
            description:
              '在推理或 SFT 训练时，为了防止“偷看未来”，会将上三角区域的分数加上 <code>-1e9</code>（即 -inf）。这样在 Softmax 后，未来的权重将严格为 0。<span class="step-badge">llm_model.py:373</span>',
            code: 'attn_scores = attn_scores + attention_mask',
            state: 'mask',
        },
        {
            title: '步骤 8: Softmax 归一化',
            description:
              '对分数进行 Softmax，使每一行之和为 1。这得到了最终的注意力权重分配，决定了每个 Token 在聚合信息时各个位置的贡献占比。<span class="step-badge">llm_model.py:374</span>',
            code: 'attn_weights = attn_scores.softmax(dim=-1)',
            state: 'softmax',
        },
        {
            title: '步骤 9: 加权聚合 Value (Context Vector)',
            description:
              '使用注意力权重对 Value 矩阵进行加权求和：<code>context = Weights @ V</code>。每个 Token 此时都融合了它所关注的其他位置的信息。<span class="step-badge">llm_model.py:378</span>',
            code: 'context = attn_weights @ value_states  // [B, H, L, D_h]',
            state: 'agg',
        },
        {
            title: '步骤 10: 拼接并输出投影 (Output Projection)',
            description:
              '将所有 Head 的结果拼回 <code>[B, L, D]</code>，然后通过 <code>o_proj</code> 线性层。这一步允许模型整合不同注意力头捕获的异构信息。<span class="step-badge">llm_model.py:382-384</span>',
            code: 'attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)\nout = self.o_proj(attn)',
            state: 'out',
        },
    ];

    function updateUI() {
        if (!visualContent) return;
        const step = steps[currentStep];
        
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><p>${step.description}</p>`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if(window.hljs) hljs.highlightElement(codeSnippet);
        }
        
        render();
        updateButtons();
    }

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
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

    function render() {
        visualContent.innerHTML = '';
        const viz = document.createElement('div');
        viz.className = 'attn-viz';
        viz.style.display = 'flex';
        viz.style.flexDirection = 'column';
        viz.style.alignItems = 'center';
        viz.style.gap = '20px';

        const state = steps[currentStep].state;

        try {
            if (state === 'init') {
                viz.appendChild(createMatrix(4, 8, 'H', 'hidden-states [B, L, D]'));
            } else if (state === 'proj') {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '20px';
                row.appendChild(createMatrix(4, 4, 'Q', 'Query', '#d6eaf8'));
                row.appendChild(createMatrix(4, 2, 'K', 'Key', '#d1f2eb'));
                row.appendChild(createMatrix(4, 2, 'V', 'Value', '#f9ebea'));
                viz.appendChild(row);
                viz.appendChild(createLabel('投影到各自的空间 (GQA 示例: Q 头数=2, KV 头数=1)'));
            } else if (state === 'qknorm') {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '20px';
                row.appendChild(createMatrix(4, 4, 'Q̂', 'Q-Normed', '#a9cce3'));
                row.appendChild(createMatrix(4, 2, 'K̂', 'K-Normed', '#a2d9ce'));
                viz.appendChild(row);
                viz.appendChild(createLabel('可选的 QK-Normalization (RMSNorm per head)'));
            } else if (state === 'reshape') {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '30px';
                
                const qGroup = document.createElement('div');
                qGroup.innerHTML = '<div style="margin-bottom:5px; font-weight:bold;">Q Heads [H, L, D_h]</div>';
                const qContainer = document.createElement('div');
                qContainer.style.display = 'flex'; qContainer.style.gap='10px';
                qContainer.appendChild(createMatrix(3, 3, 'q0', 'Head 0', '#d6eaf8'));
                qContainer.appendChild(createMatrix(3, 3, 'q1', 'Head 1', '#d6eaf8'));
                qGroup.appendChild(qContainer);
                
                const kGroup = document.createElement('div');
                kGroup.innerHTML = '<div style="margin-bottom:5px; font-weight:bold;">K Heads [H_kv, L, D_h]</div>';
                kGroup.appendChild(createMatrix(3, 3, 'kv0', 'Head 0 (Shared)', '#d1f2eb'));
                
                row.appendChild(qGroup);
                row.appendChild(kGroup);
                viz.appendChild(row);
            } else if (state === 'rope') {
                const container = document.createElement('div');
                container.style.display = 'flex'; container.style.gap = '40px';
                
                // Custom RoPE Visual with animation
                const createRoPEViz = (label, color) => {
                    const wrap = document.createElement('div');
                    wrap.style.textAlign = 'center';
                    const circle = document.createElement('div');
                    circle.style.width = '80px'; circle.style.height = '80px';
                    circle.style.borderRadius = '50%';
                    circle.style.border = `2px solid ${color}`;
                    circle.style.position = 'relative';
                    circle.style.margin = '0 auto 10px auto';
                    circle.style.background = '#fff';
                    
                    // Vector
                    const vec = document.createElement('div');
                    vec.style.position = 'absolute';
                    vec.style.top = '50%'; vec.style.left = '50%';
                    vec.style.width = '35px'; vec.style.height = '2px';
                    vec.style.background = color;
                    vec.style.transformOrigin = '0 0';
                    vec.style.transform = 'rotate(-30deg)'; // Initial
                    vec.className = 'rotating'; // Add CSS animation
                    circle.appendChild(vec);
                    
                    // Rotation Arrow
                    const arrow = document.createElement('div');
                    arrow.style.position = 'absolute';
                    arrow.style.top = '10px'; arrow.style.right = '10px';
                    arrow.style.fontSize = '12px';
                    arrow.innerText = '↺ θ';
                    circle.appendChild(arrow);
                    
                    wrap.appendChild(circle);
                    wrap.appendChild(document.createTextNode(label));
                    return wrap;
                };
                
                container.appendChild(createRoPEViz('Query Vector (Rotated)', '#3498db'));
                container.appendChild(createRoPEViz('Key Vector (Rotated)', '#1abc9c'));
                viz.appendChild(container);
                viz.appendChild(createLabel('RoPE: 将绝对位置 m 编码为旋转角度 mθ'));
            } else if (state === 'gqa') {
                const container = document.createElement('div');
                container.style.display = 'flex';
                container.style.gap = '40px';
                container.style.alignItems = 'center';
                
                const kHeads = document.createElement('div');
                kHeads.innerHTML = '<div style="margin-bottom:5px;">Original K Head</div>';
                kHeads.appendChild(createMatrix(2, 2, 'K0', '', '#d1f2eb'));
                
                const arrow = document.createElement('div');
                arrow.style.fontSize = '24px'; arrow.innerText = '➡ Expand ➡';
                arrow.className = 'anim-scale';
                arrow.style.animationDelay = '0.5s';
                
                const expanded = document.createElement('div');
                expanded.innerHTML = '<div style="margin-bottom:5px;">Expanded for Q Heads</div>';
                const grp = document.createElement('div');
                grp.style.display = 'flex'; grp.style.gap='10px';
                grp.appendChild(createMatrix(2, 2, 'K0', 'For Q0', '#d1f2eb'));
                grp.appendChild(createMatrix(2, 2, 'K0', 'For Q1', '#d1f2eb'));
                expanded.appendChild(grp);
                
                container.appendChild(kHeads);
                container.appendChild(arrow);
                container.appendChild(expanded);
                viz.appendChild(container);
            } else if (state === 'score_example') {
                viz.innerHTML = `
                    <div style="display:flex; flex-direction:column; gap:20px; align-items:center; font-family:monospace; font-size:16px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div style="background:#d6eaf8; padding:5px; border:1px solid #3498db; border-radius:4px;">Q = [1, 0]</div>
                            <div>·</div>
                            <div style="background:#d1f2eb; padding:5px; border:1px solid #2ecc71; border-radius:4px;">K1 = [1, 0]</div>
                            <div>= <span style="color:green; font-weight:bold;">1.0</span> (Match!)</div>
                        </div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div style="background:#d6eaf8; padding:5px; border:1px solid #3498db; border-radius:4px;">Q = [1, 0]</div>
                            <div>·</div>
                            <div style="background:#f9ebea; padding:5px; border:1px solid #e74c3c; border-radius:4px;">K2 = [0, 1]</div>
                            <div>= <span style="color:red; font-weight:bold;">0.0</span> (No Match)</div>
                        </div>
                        <div style="margin-top:10px; font-size:14px; color:#666;">这就是 Attention 的本质：计算向量间的匹配度。</div>
                    </div>
                `;
            } else if (state === 'score') {
                viz.innerHTML = `
                    <div style="display:flex; align-items:center; gap:10px;">
                        ${createMatrix(3, 4, 'Q', '', '#d6eaf8').outerHTML}
                        <div style="font-size:1.5em;">@</div>
                        ${createMatrix(4, 3, 'Kᵀ', '', '#d1f2eb').outerHTML}
                        <div style="font-size:1.5em;">=</div>
                        ${createMatrix(3, 3, 'S', 'Scores', '#fef5e7').outerHTML}
                    </div>
                `;
                viz.appendChild(createLabel('Q(Row i) • K(Col j) = Score(i,j)'));
            } else if (state === 'mask') {
                const m = createMatrix(4, 4, '', 'Causal Mask', '#fef5e7');
                const cells = m.querySelectorAll('.cell');
                cells.forEach((c, i) => {
                    const row = Math.floor(i / 4);
                    const col = i % 4;
                    if (col > row) {
                        c.innerText = '-∞';
                        c.style.background = '#fadbd8';
                        c.style.color = '#c0392b';
                        c.title = 'Future Token Masked';
                        c.style.animationDelay = `${i*0.05}s`; // Staggered fade in
                    } else {
                        c.innerText = 'v';
                        c.title = 'Visible';
                    }
                });
                viz.appendChild(m);
                viz.appendChild(createLabel('掩盖未来 Token 的可见性 (Upper Triangle)'));
            } else if (state === 'softmax') {
                const m = createMatrix(4, 4, 'w', 'Attn Weights', '#d5f5e3');
                const cells = m.querySelectorAll('.cell');
                cells.forEach((c, i) => {
                    const row = Math.floor(i / 4);
                    const col = i % 4;
                    if (col > row) {
                        c.innerText = '0';
                        c.style.background = '#f2f2f2';
                        c.style.color = '#ccc';
                    } else {
                        // Simulate some weights
                        const opacity = (1 / (row + 1)).toFixed(2);
                        c.style.background = `rgba(46, 204, 113, ${opacity})`;
                        c.innerText = opacity;
                    }
                });
                viz.appendChild(m);
                viz.appendChild(createLabel('Softmax 归一化，每行和为 1.0'));
            } else if (state === 'agg') {
                viz.innerHTML = `
                    <div style="display:flex; align-items:center; gap:10px;">
                        ${createMatrix(3, 3, 'W', '', '#d5f5e3').outerHTML}
                        <div style="font-size:1.5em;">@</div>
                        ${createMatrix(3, 4, 'V', '', '#f9ebea').outerHTML}
                        <div style="font-size:1.5em;">=</div>
                        ${createMatrix(3, 4, 'C', 'Context', '#fdf2e9').outerHTML}
                    </div>
                `;
            } else if (state === 'out') {
                viz.appendChild(createMatrix(4, 8, 'Z', 'Concatenated Heads', '#fdf2e9'));
                viz.innerHTML += '<div style="font-size:1.5em;" class="anim-scale" style="animation-delay:0.5s">↓ o_proj ↓</div>';
                viz.appendChild(createMatrix(4, 8, 'O', 'Output [B, L, D]', '#fcf3cf'));
            }
            visualContent.appendChild(viz);
        } catch (e) {
            console.error("Render error:", e);
        }
    }

    function createMatrix(rows, cols, labelPrefix, title = '', color = '#fff') {
        const container = document.createElement('div');
        container.style.display = 'inline-block';
        container.style.textAlign = 'center';
        
        if (title) {
            const t = document.createElement('div');
            t.style.fontSize = '0.8em';
            t.style.marginBottom = '4px';
            t.innerText = title;
            container.appendChild(t);
        }
        
        const m = document.createElement('div');
        m.style.display = 'grid';
        m.style.gridTemplateColumns = `repeat(${cols}, 24px)`;
        m.style.gridTemplateRows = `repeat(${rows}, 24px)`;
        m.style.gap = '2px';
        m.style.border = '1px solid #34495e';
        m.style.padding = '3px';
        m.style.background = '#f4f6f7';
        m.style.borderRadius = '4px';

        for (let i = 0; i < rows * cols; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell anim-scale'; // Added animation class
            cell.style.animationDelay = `${i * 0.02}s`; // Staggered delay
            
            cell.style.width = '24px';
            cell.style.height = '24px';
            cell.style.background = color;
            cell.style.border = '1px solid #bdc3c7';
            cell.style.fontSize = '9px';
            cell.style.display = 'flex';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'center';
            if (labelPrefix) {
                cell.innerText = labelPrefix + (labelPrefix.length > 1 ? '' : i);
            }
            m.appendChild(cell);
        }
        container.appendChild(m);
        return container;
    }

    function createLabel(text) {
        const d = document.createElement('div');
        d.className = 'op-label';
        d.style.marginTop = '10px';
        d.style.fontWeight = 'bold';
        d.style.color = '#2c3e50';
        d.innerText = text;
        return d;
    }

    // Bind Events
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; updateUI(); });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    updateUI();
});
