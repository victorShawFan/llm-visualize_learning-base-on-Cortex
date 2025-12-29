document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const formulaBox = document.getElementById('formulaBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const chartArea = document.getElementById('chartArea');

    // Guard: Ensure essential elements exist
    if (!chartArea || !infoBox || !formulaBox || !codeSnippet) {
        console.error("Required elements not found in loss_math_script");
        return;
    }

    let currentStep = 0;

    // 覆盖 llm_trainer/loss.py 中所有核心损失：LM CE、加权 CE、KD、DPO/IPO、PPO、GRPO
    const steps = [
        {
            title: "Step 0: 从 Logits 到 Loss 的总览",
            description: "所有损失都从 <code>logits</code> 出发：先做序列错位 (shift)，再根据 Mask 过滤有效位置，最后用交叉熵或偏好优化公式聚合成标量。<span class='step-badge'>loss.py:40-89</span>",
            formula: "(logits, labels) → shift → mask(ignore=-100) → loss",
            code: "loss = LMLoss(...)(logits, labels)",
            type: "tensor_overview",
            data: { logits: ["t0", "t1", "t2", "t3"], labels: ["t0", "t1", "t2", "t3"] }
        },
        {
            title: "Step 1: Input Logits & Labels (输入)",
            description: "模型输出的 Logits 形状为 <code>[B, Seq, V]</code>，标签为 <code>[B, Seq]</code>。假设 Seq=4，本步展示原始对齐关系。<span class='step-badge'>loss.py:45-47</span>",
            formula: "logits[b, t] → 预测 label[b, t+1]",
            code: "logits = model(input_ids)  # [B, S, V]\nlabels = input_ids          # [B, S]",
            type: "tensor_input",
            data: { logits: ["A", "B", "C", "D"], labels: ["A", "B", "C", "D"] }
        },
        {
            title: "Step 2: Shift Logits (错位预测)",
            description: "移除 Logits 的最后一个时间步，只保留下标 0～S-2；因为它们各自负责预测下一步标签。<span class='step-badge'>loss.py:52</span>",
            formula: "shift_logits = logits[..., :-1, :]",
            code: "shift_logits = logits[..., :-1, :].contiguous()",
            type: "tensor_shift_logits",
            data: { logits: ["A", "B", "C", "D"], labels: ["A", "B", "C", "D"] }
        },
        {
            title: "Step 3: Shift Labels (对齐标签)",
            description: "对应地，把 Labels 的第一个 token 删掉，让 <code>shift_logits[t]</code> 与 <code>shift_labels[t]</code> 精确对齐。<span class='step-badge'>loss.py:53</span>",
            formula: "shift_labels = labels[..., 1:]",
            code: "shift_labels = labels[..., 1:].contiguous()",
            type: "tensor_shift_labels",
            data: { logits: ["A", "B", "C"], labels: ["A", "B", "C", "D"] }
        },
        {
            title: "Step 4: 展平为 (N, V) 与 (N)",
            description: "交叉熵实现期望输入二维 logits 和一维 targets。源码中用 <code>reshape(-1, vocab)</code> 和 <code>reshape(-1)</code> 将所有时间步摊平。<span class='step-badge'>loss.py:56-57</span>",
            formula: "logits = shift_logits.view(N, V)\ntargets = shift_labels.view(N)",
            code: "logits = shift_logits.reshape(-1, logits.size(-1))\ntargets = shift_labels.reshape(-1)",
            type: "tensor_flatten",
            data: { logits: ["A", "B", "C"], labels: ["B", "C", "D"] }
        },
        {
            title: "Step 5: Masking (忽略 Padding / Prompt)",
            description: "交叉熵通过 <code>ignore_index=-100</code> 自动忽略 Prompt 和 Padding 的梯度，配合前面的 <code>_mask_prompt</code> 使用。<span class='step-badge'>loss.py:60-65, utils.py:474-533</span>",
            formula: "CE(logits, targets, ignore_index = -100)",
            code: "ce_loss = F.cross_entropy(\n    logits, targets, ignore_index=self.ignore_index,\n    weight=self.weights if use_crit else None\n)",
            type: "tensor_mask",
            data: { logits: ["A", "B", "C"], labels: ["-100", "C", "D"] }
        },
        {
            title: "Step 6: Critical Tokens 加权 (重要Token放大权重)",
            description: "当配置了 <code>critical_tokens</code> 时，LMLoss 会注册一个权重向量，对例如 <code>&lt;/s&gt;</code> 等关键 Token 放大损失权重。<span class='step-badge'>loss.py:33-37</span>",
            formula: "weights[i] = α (i ∈ critical_tokens), 1 其他",
            code: "self.register_buffer('weights', torch.ones(vocab_size))\nself.weights[self.critical_tokens] = critical_alpha",
            type: "tensor_weights",
            data: { vocab: ["其它", "EOS", "特殊"], weights: ["1.0", "α", "1.0"] }
        },
        {
            title: "Step 7: KDLoss – 软标签的交叉熵",
            description: "KDLoss 用教师分布 <code>teacher_probs</code> 作为软标签，对学生的 <code>logprobs</code> 做期望，得到 KL 项的一部分。<span class='step-badge'>loss.py:120-136</span>",
            formula: "L_KD = - E_{y∼P_T} [ log P_S(y) ]",
            code: "teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)\nlogprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)\nprod = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)\nx = prod.sum(-1).view(-1)\nmask = (labels != ignore_index).int()\ndistil_loss = - (x * mask).sum() / mask.sum()",
            type: "curve_kd"
        },
        {
            title: "Step 8: DPO / IPO – 直接偏好优化",
            description: "DPO 使用策略/参考模型在 chosen/rejected 上的 log 比差值作为 logit，最小化 logistic 损失；IPO 用二次项代替 logistic。<span class='step-badge'>loss.py:193-210</span>",
            formula: "logits = (π_c-π_r) - (π_ref_c-π_ref_r)\nL_DPO = -log σ(β·logits)",
            code: "pi_logratios  = policy_chosen_logps - policy_reject_logps\nref_logratios = ref_chosen_logps - ref_reject_logps\nlogits = pi_logratios - ref_logratios\nif self.ipo:\n    losses = (logits - 1 / (2 * self.beta)) ** 2\nelse:\n    losses = -F.logsigmoid(self.beta * logits)",
            type: "curve_dpo"
        },
        {
            title: "Step 9: PPO – 裁剪后的策略损失与 Value 损失",
            description: "PPO 同时优化 Actor 和 Critic：对 <code>ratio</code> × advantage 做裁剪，同时对 Value 使用 clip+MSE，并加权求和。<span class='step-badge'>loss.py:275-283,287-297</span>",
            formula: "L_actor = - E[min(r·A, clip(r,1±ε)·A)]\nL_v = 0.5·max(MSE(v, R), MSE(v_clip, R))",
            code: "ratio = torch.exp(log_probs - old_log_probs)\nsurr1 = ratio * advantages\nsurr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages\nactor_loss = - (torch.min(surr1, surr2) * mask).sum() / mask.sum()",
            type: "curve_ppo"
        },
        {
            title: "Step 10: GRPO / GSPO – 重要性采样 + 双边裁剪",
            description: "GRPOLoss 在 token / 序列级别计算重要性权重 <code>exp(log_importance)</code>，对优势做双边裁剪，并可选叠加 KL 正则和 BNPO / DR-GRPO 归一化。<span class='step-badge'>loss.py:357-393</span>",
            formula: "w = exp(log_w), w_clip = clip(w, 1±ε)\nL = -min(w·A, w_clip·A) + β·KL",
            code: "log_ratio = log_probs - old_log_probs\nlog_importance_weights = ...  # token 或 seq 级别\ncoef_1 = torch.exp(log_importance_weights)\ncoef_2 = torch.clamp(coef_1, 1-clip_low, 1+clip_high)\nper_token_loss = -torch.min(coef_1*A, coef_2*A)\nif beta != 0: per_token_loss += beta * per_token_kl",
            type: "curve_grpo"
        }
    ];

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
    }

    function renderTensorView(step) {
        const container = document.createElement('div');
        container.className = 'tensor-grid-container';

        // Helper to create row
        const createRow = (label, data, highlightIndex = -1, fadeIndex = -1, isLabel = false) => {
            const row = document.createElement('div');
            row.className = 'tensor-row';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'tensor-label';
            labelDiv.textContent = label;
            row.appendChild(labelDiv);

            data.forEach((val, idx) => {
                const box = document.createElement('div');
                box.className = 'tensor-box';
                box.textContent = val;
                
                if (idx === highlightIndex) box.classList.add('active');
                if (idx === fadeIndex) box.classList.add('shifted-out');
                
                // Special case for mask
                if (step.type === 'tensor_mask' && isLabel) {
                     // 将被忽略的位置（标签为 -100）用灰色虚线高亮，直观对应 ignore_index 语义
                     if (val === '-100') {
                         box.classList.add('ignore');
                     }
                }
                
                row.appendChild(box);
            });
            return row;
        };

        if (step.type === 'tensor_overview') {
            container.appendChild(createRow("logits", step.data.logits));
            container.appendChild(createRow("labels", step.data.labels));
            const arrow = document.createElement('div');
            arrow.className = 'arrow-down';
            arrow.textContent = 'shift → flatten → mask(ignore=-100) → loss';
            container.appendChild(arrow);
        } else if (step.type === 'tensor_input') {
            container.appendChild(createRow("Logits (t)", step.data.logits));
            container.appendChild(createRow("Labels (t)", step.data.labels));
        } else if (step.type === 'tensor_shift_logits') {
            container.appendChild(createRow("Logits (t)", step.data.logits, -1, 3)); // 3 is 'D', fade it
            const arrow = document.createElement('div');
            arrow.className = 'arrow-down';
            arrow.textContent = '↓ Drop Last';
            container.appendChild(arrow);
            container.appendChild(createRow("Shifted", ["A", "B", "C"]));
        } else if (step.type === 'tensor_shift_labels') {
             container.appendChild(createRow("Labels (t)", step.data.labels, -1, 0)); // 0 is 'A', fade it
             const arrow = document.createElement('div');
             arrow.className = 'arrow-down';
             arrow.textContent = '↓ Drop First';
             container.appendChild(arrow);
             container.appendChild(createRow("Shifted", ["B", "C", "D"]));
        } else if (step.type === 'tensor_mask') {
             container.appendChild(createRow("S. Logits", step.data.logits));
             container.appendChild(createRow("S. Labels", step.data.labels, -1, -1, true));
             const arrow = document.createElement('div');
             arrow.className = 'arrow-down';
             arrow.textContent = '↓ Ignore -100';
             container.appendChild(arrow);
             // Visual representation of loss calc only on valid tokens
             const validBox = document.createElement('div');
             validBox.innerHTML = "Calculate Loss on: <b>C, D</b> only";
             validBox.style.padding = "10px";
             validBox.style.background = "#e8f8f5";
             validBox.style.borderRadius = "8px";
             container.appendChild(validBox);
        } else if (step.type === 'tensor_flatten') {
             container.appendChild(createRow("Shift logits", step.data.logits));
             const arrow = document.createElement('div');
             arrow.className = 'arrow-down';
             arrow.textContent = 'view(-1, V) / view(-1)';
             container.appendChild(arrow);
             container.appendChild(createRow("Flatten logits", ["A", "B", "C"]));
             container.appendChild(createRow("Flatten targets", step.data.labels));
        } else if (step.type === 'tensor_weights') {
             const row = document.createElement('div');
             row.className = 'tensor-row';
             const labelDiv = document.createElement('div');
             labelDiv.className = 'tensor-label';
             labelDiv.textContent = 'weights';
             row.appendChild(labelDiv);
             step.data.vocab.forEach((tok, idx) => {
                 const box = document.createElement('div');
                 box.className = 'tensor-box';
                 if (tok === 'EOS') {
                     box.classList.add('active');
                 }
                 box.innerHTML = `<div>${tok}</div><div style="font-size:11px; color:#555;">w=${step.data.weights[idx]}</div>`;
                 row.appendChild(box);
             });
             container.appendChild(row);
        }

        chartArea.innerHTML = '';
        chartArea.appendChild(container);
    }

    function renderCurveView(step) {
        chartArea.innerHTML = `<svg id="svgRoot" viewBox="0 0 600 400">
            <line x1="50" y1="350" x2="550" y2="350" class="axis" />
            <text x="560" y="355">Input</text>
            <line x1="50" y1="350" x2="50" y2="50" class="axis" />
            <text x="45" y="40">Loss</text>
            <path id="curvePath" class="curve" d="" />
        </svg>`;
        
        const curvePath = document.getElementById('curvePath');
        if (!curvePath) return;

        let points = [];
        
        if (step.type === "curve_ce") {
            for (let x = 0.05; x <= 1; x += 0.05) {
                const y = -Math.log(x);
                points.push({x: 50 + x * 450, y: 350 - y * 50});
            }
            curvePath.classList.add('ce-curve');
        } else if (step.type === "curve_kd") {
            for (let x = 0.05; x <= 1; x += 0.05) {
                const y = -0.5 * Math.log(x); 
                points.push({x: 50 + x * 450, y: 350 - y * 50});
            }
            curvePath.classList.add('ce-curve');
            curvePath.style.stroke = "#9b59b6";
        } else if (step.type === "curve_dpo") {
            for (let x = -3; x <= 3; x += 0.2) {
                const sigmoid = 1 / (1 + Math.exp(-x));
                const y = -Math.log(sigmoid); // DPO
                // IPO would be (x - margin)^2
                points.push({x: 300 + x * 75, y: 350 - y * 50});
            }
            curvePath.classList.add('dpo-curve');
        } else if (step.type === "curve_grpo") {
            const adv = 1;
            const eps = 0.2;
            for (let x = 0.5; x <= 1.5; x += 0.05) {
                const ratio = x;
                const sur1 = ratio * adv;
                const sur2 = Math.max(Math.min(ratio, 1+eps), 1-eps) * adv;
                const val = Math.min(sur1, sur2);
                points.push({x: 50 + (x-0.5) * 450, y: 350 - val * 100});
            }
            curvePath.classList.add('ppo-curve');
        }

        if (points.length > 0) {
            let d = `M ${points[0].x} ${points[0].y}`;
            for (let i = 1; i < points.length; i++) {
                d += ` L ${points[i].x} ${points[i].y}`;
            }
            curvePath.setAttribute('d', d);
        }
    }

    function render() {
        if (!infoBox || !formulaBox || !codeSnippet || !chartArea) return;
        
        const step = steps[currentStep];
        infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        formulaBox.innerHTML = `<code style="font-size:1.1em; color:#e67e22">${step.formula}</code>`;
        codeSnippet.textContent = step.code;

        try {
            if (step.type.startsWith("tensor")) {
                renderTensorView(step);
            } else {
                renderCurveView(step);
            }
        } catch (e) {
            console.error("Render failed", e);
        }
        
        updateButtons();
    }

    function goNext() {
        if (currentStep < steps.length - 1) {
            currentStep++;
            render();
        }
    }

    function goPrev() {
        if (currentStep > 0) {
            currentStep--;
            render();
        }
    }

    // Event Listeners
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; render(); });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Initial render
    render();
});
