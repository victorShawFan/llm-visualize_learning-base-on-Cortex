document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    const container = document.getElementById('math-container'); // This script seems to use 'container' variable in render functions but 'visualContent' is passed?
    // Wait, let me check the original code again.
    // Original code used `container.innerHTML = ...` in render functions, but defined `container` variable inside render functions?
    // No, `renderRepl` used `container.innerHTML`. Where is `container` defined?
    // It wasn't defined in the top scope in previous `Read` output! 
    // Ah, wait. The previous Read output for `grpo_math_script.js` showed:
    // 73-> function renderRepl() {
    // 74->    container.innerHTML = ...
    // But `container` was NOT defined at the top. It was implicitly global or missing?
    // If I look at the very top:
    // 1-> const prevBtn ...
    // 5-> const visualContent = document.getElementById('visualContent');
    // It seems `visualContent` is what should be used. The original code likely had a bug reference to `container` if it wasn't defined.
    // OR `container` might be `visualContent`.
    // Let's assume `container` refers to `visualContent`. I will alias it.

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
        console.error("Required elements not found in GRPO Math script");
        return;
    }

    const container = visualContent; // Fix potential variable name mismatch

    let currentStep = 0;

    const steps = [
        {
            title: "Prompt 复制 (Replication)",
            desc: "设 Batch=1, Group Size=4。将原始 Prompt 复制 4 份，送入模型生成 4 个不同的回答 (Completion)。",
            badge: "grpo_trainer.py:274",
            state: "repl"
        },
        {
            title: "奖励评分 (Scoring)",
            desc: "对这 4 个生成结果分别计算 Reward（包括 KL 惩罚和环境奖励）。假设得到得分：[0.8, 0.9, 0.4, 0.7]。",
            badge: "grpo_trainer.py:384",
            state: "score"
        },
        {
            title: "组内统计 (Group Stats)",
            desc: "计算该组得分的均值 (Mean) 和标准差 (Std)。<br>Mean = (0.8+0.9+0.4+0.7)/4 = 0.7<br>Std ≈ 0.216",
            badge: "grpo_trainer.py:234",
            state: "stats"
        },
        {
            title: "优势归一化 (Normalization)",
            desc: "计算相对优势：Adv_i = (Reward_i - Mean) / Std。表现好的结果获得正优势，差的获得负优势。",
            badge: "grpo_trainer.py:244",
            state: "norm"
        },
        {
            title: "零均值检验 (Zero-Mean Check)",
            desc: "归一化后的组内 Advantage 之和应接近 0，表示只是重新分配了权重，而不会整体放大奖励。",
            badge: "grpo_trainer.py:244",
            state: "zero"
        },
        {
            title: "梯度分配 (Gradient Sharing)",
            desc: "优势越高的样本，其梯度更新越大；劣质样本的梯度被压制，从而实现组内“好样本带坏样本”的效果。",
            badge: "loss.py:359",
            state: "grad"
        },
        {
            title: "归一化前后的分布对比",
            desc: "在归一化之前，Reward 分布可能非常偏斜；归一化之后，优势向量在 0 附近分布，更适合作为梯度缩放因子，避免数值爆炸。",
            badge: "grpo_trainer.py:234-244",
            state: "stats"
        },
        {
            title: "不同 Group 之间的独立性",
            desc: "每个 Prompt 的 4 个回答只在组内比较，互不影响其他 Prompt 的优势分配。这使得 GRPO 可以在大 Batch 上并行计算，而不会产生跨组的相互干扰。",
            badge: "grpo_trainer.py:230",
            state: "repl"
        },
        {
            title: "带 KL 的有效奖励 (R_eff)",
            desc: "在本仓库实现中，KL 惩罚通过 <code>GRPOLoss</code> 中的 <code>beta * per_token_kl</code> 加到 loss 上；也可以把 Reward 预先写成 <code>R_eff = R_env - β·KL(π||π_ref)</code> 的形式，此处的 0.8, 0.9 等分数就可以理解为已经“扣掉”KL 后的净收益。",
            badge: "loss.py:359, 386-393",
            state: "score"
        },
        {
            title: "从数学图解到代码实现",
            desc: "本页面展示的是单个 Prompt 的数学过程；在源码中，这一流程会在 Batch 维度上展开，对所有 Prompt 同时计算 Advantage 并回传梯度，与 <code>GRPOTrainer</code> 的训练循环一一对应。",
            badge: "grpo_trainer.py:270-310",
            state: "grad"
        },
        {
            title: "实战模拟 (Simulation)",
            desc: "让我们通过一个动态例子来巩固理解。点击“刷新数据”生成一组新的随机 Reward，观察 Advantage 如何随 Reward 分布变化而变化。注意：当所有 Reward 都很高时，Advantage 依然会有正有负（相对优劣）。",
            badge: "Interactive Simulation",
            state: "sim"
        }
    ];

    function updateUI() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        if(infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.desc}`;
        if(codeSnippet) codeSnippet.textContent = step.badge;
        
        container.innerHTML = ''; // Clear container
        
        try {
            if (step.state === 'repl') renderRepl();
            else if (step.state === 'score') renderScore();
            else if (step.state === 'stats') renderStats();
            else if (step.state === 'norm') renderNorm();
            else if (step.state === 'zero') renderZero();
            else if (step.state === 'grad') renderGrad();
            else if (step.state === 'sim') renderSim();
        } catch(e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    function renderSim() {
        container.innerHTML = `
            <div class="sim-box" style="display:flex; flex-direction:column; align-items:center; gap:20px; width:100%;">
                <button id="regen-btn" style="padding:10px 20px; background:#3498db; color:#fff; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">🎲 刷新数据 (Generate New Rewards)</button>
                
                <div id="sim-viz" style="display:flex; gap:20px; width:100%; justify-content:center; flex-wrap:wrap;">
                    <!-- Content will be injected by JS -->
                </div>
            </div>
        `;
        
        const btn = document.getElementById('regen-btn');
        if(btn) btn.onclick = runSimulation;
        
        runSimulation();
    }

    function runSimulation() {
        const simViz = document.getElementById('sim-viz');
        if(!simViz) return;
        
        // 1. Generate 4 random rewards (e.g., between 0.0 and 1.0)
        const rewards = Array.from({length: 4}, () => parseFloat((Math.random()).toFixed(2)));
        
        // 2. Calc Mean & Std
        const mean = rewards.reduce((a,b)=>a+b, 0) / 4;
        const variance = rewards.reduce((a,b)=>a + Math.pow(b - mean, 2), 0) / 4;
        const std = Math.sqrt(variance) + 1e-4; // avoid div by zero
        
        // 3. Calc Adv
        const advs = rewards.map(r => (r - mean) / std);
        
        let html = '';
        
        // Render Group Stats
        html += `
            <div style="width:100%; text-align:center; background:#f4f6f7; padding:10px; border-radius:8px; margin-bottom:10px;">
                <span style="margin-right:20px;">Mean: <b>${mean.toFixed(3)}</b></span>
                <span>Std: <b>${std.toFixed(3)}</b></span>
            </div>
        `;
        
        rewards.forEach((r, i) => {
            const adv = advs[i];
            const isPos = adv >= 0;
            const color = isPos ? '#27ae60' : '#c0392b';
            const bg = isPos ? '#d5f5e3' : '#fadbd8';
            const icon = isPos ? '↑' : '↓';
            
            html += `
                <div class="sim-card" style="background:${bg}; border:2px solid ${color}; padding:15px; border-radius:8px; width:120px; text-align:center; transition:all 0.3s;">
                    <div style="font-size:12px; color:#555;">Sample ${i+1}</div>
                    <div style="font-size:18px; font-weight:bold; margin:5px 0;">R: ${r.toFixed(2)}</div>
                    <div style="height:1px; background:${color}; margin:5px 0; opacity:0.5;"></div>
                    <div style="font-size:14px; color:${color}; font-weight:bold;">Adv: ${adv.toFixed(2)}</div>
                    <div style="font-size:20px; color:${color};">${icon}</div>
                </div>
            `;
        });
        
        simViz.innerHTML = html;
    }

    function renderRepl() {
        container.innerHTML = `
            <div class="grpo-flow" style="display:flex; flex-direction:column; align-items:center; width:100%;">
                <div class="prompt-box" style="background:#e8f8f5; border:2px solid #27ae60; padding:10px 20px; border-radius:8px; margin-bottom:20px;">Prompt: "Write a poem"</div>
                <div class="arrow-down" style="font-size:24px; color:#2c3e50;">⬇ × 4 (Group)</div>
                <div class="completion-row" style="display:flex; gap:10px; margin-top:20px;">
                    <div class="comp-card" style="background:#fff; border:1px solid #ddd; padding:10px; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">C1: "Moon..."</div>
                    <div class="comp-card" style="background:#fff; border:1px solid #ddd; padding:10px; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">C2: "Sun..."</div>
                    <div class="comp-card" style="background:#fff; border:1px solid #ddd; padding:10px; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">C3: "Star..."</div>
                    <div class="comp-card" style="background:#fff; border:1px solid #ddd; padding:10px; border-radius:6px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">C4: "Sky..."</div>
                </div>
            </div>
        `;
    }

    function renderScore() {
        container.innerHTML = `
            <div class="score-viz" style="display:flex; justify-content:center; gap:20px; margin-top:30px;">
                <div class="score-item" style="text-align:center;">
                    <div class="comp-mini" style="font-size:12px; color:#666;">C1</div>
                    <div class="score-val" style="font-size:24px; color:#27ae60; font-weight:bold;">0.8</div>
                </div>
                <div class="score-item" style="text-align:center;">
                    <div class="comp-mini" style="font-size:12px; color:#666;">C2</div>
                    <div class="score-val" style="font-size:24px; color:#27ae60; font-weight:bold;">0.9</div>
                </div>
                <div class="score-item" style="text-align:center;">
                    <div class="comp-mini" style="font-size:12px; color:#666;">C3</div>
                    <div class="score-val bad" style="font-size:24px; color:#c0392b; font-weight:bold;">0.4</div>
                </div>
                <div class="score-item" style="text-align:center;">
                    <div class="comp-mini" style="font-size:12px; color:#666;">C4</div>
                    <div class="score-val" style="font-size:24px; color:#f39c12; font-weight:bold;">0.7</div>
                </div>
            </div>
        `;
    }

    function renderStats() {
        container.innerHTML = `
            <div class="stats-box" style="width:100%; max-width:600px; margin:0 auto; padding:20px; background:#f8f9fa; border-radius:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
                    <div class="stat-line">Sum = 2.8</div>
                    <div class="stat-line highlight" style="color:#2980b9; font-weight:bold;">Mean (μ) = 0.7</div>
                    <div class="stat-line">Std (σ) ≈ 0.216</div>
                </div>
                <div class="baseline-viz" style="position:relative; height:60px; background:#e5e7e9; border-radius:30px;">
                    <div class="line-base"></div>
                    <!-- Points scaled: 0.4=0%, 0.9=100% (range 0.5) -->
                    <div class="point p1" style="left: 80%; background:#27ae60; position:absolute; width:12px; height:12px; border-radius:50%; top:50%; transform:translateY(-50%);"></div>
                    <div class="point p2" style="left: 100%; background:#27ae60; position:absolute; width:12px; height:12px; border-radius:50%; top:50%; transform:translateY(-50%);"></div>
                    <div class="point p3" style="left: 0%; background:#c0392b; position:absolute; width:12px; height:12px; border-radius:50%; top:50%; transform:translateY(-50%);"></div>
                    <div class="point p4" style="left: 60%; background:#f39c12; position:absolute; width:12px; height:12px; border-radius:50%; top:50%; transform:translateY(-50%);"></div>
                    
                    <div class="mean-marker" style="left: 60%; position:absolute; height:100%; width:2px; background:#2980b9; top:0;"></div>
                    <div style="position:absolute; left:60%; top:-20px; color:#2980b9; font-size:12px; transform:translateX(-50%);">μ=0.7</div>
                </div>
            </div>
        `;
    }

    function renderNorm() {
        container.innerHTML = `
            <div class="adv-results" style="display:flex; justify-content:center; gap:15px; margin-top:20px;">
                <div class="adv-card positive" style="padding:15px; border-radius:8px; background:#d5f5e3; border:1px solid #27ae60; text-align:center;">
                    <div class="id" style="font-weight:bold;">C1</div>
                    <div class="calc" style="font-size:10px; color:#666; margin:5px 0;">(0.8 - 0.7)/0.216</div>
                    <div class="res" style="font-size:18px; color:#27ae60; font-weight:bold;">+0.46</div>
                </div>
                <div class="adv-card positive" style="padding:15px; border-radius:8px; background:#d5f5e3; border:1px solid #27ae60; text-align:center;">
                    <div class="id" style="font-weight:bold;">C2</div>
                    <div class="calc" style="font-size:10px; color:#666; margin:5px 0;">(0.9 - 0.7)/0.216</div>
                    <div class="res" style="font-size:18px; color:#27ae60; font-weight:bold;">+0.92</div>
                </div>
                <div class="adv-card negative" style="padding:15px; border-radius:8px; background:#fadbd8; border:1px solid #c0392b; text-align:center;">
                    <div class="id" style="font-weight:bold;">C3</div>
                    <div class="calc" style="font-size:10px; color:#666; margin:5px 0;">(0.4 - 0.7)/0.216</div>
                    <div class="res" style="font-size:18px; color:#c0392b; font-weight:bold;">-1.38</div>
                </div>
                <div class="adv-card neutral" style="padding:15px; border-radius:8px; background:#f4f6f7; border:1px solid #bdc3c7; text-align:center;">
                    <div class="id" style="font-weight:bold;">C4</div>
                    <div class="calc" style="font-size:10px; color:#666; margin:5px 0;">(0.7 - 0.7)/0.216</div>
                    <div class="res" style="font-size:18px; color:#7f8c8d; font-weight:bold;">0.00</div>
                </div>
            </div>
        `;
    }

    function renderZero() {
        container.innerHTML = `
            <div class="stats-box" style="text-align:center; padding:30px; background:#f0f3f4; border-radius:10px;">
                <div class="stat-line" style="font-family:monospace; margin-bottom:15px;">
                    0.46 + 0.92 + (-1.38) + 0.00
                </div>
                <div class="stat-line highlight" style="font-size:24px; color:#2c3e50; font-weight:bold;">
                    Sum(A) ≈ 0.0
                </div>
                <div style="margin-top:10px; font-size:12px; color:#7f8c8d;">(Due to Mean Subtraction)</div>
            </div>
        `;
    }

    function renderGrad() {
        container.innerHTML = `
            <div class="adv-results" style="display:flex; justify-content:center; gap:20px; margin-top:20px;">
                <div class="adv-card positive" style="text-align:center;">
                    <div class="id" style="font-weight:bold;">C1 (+0.46)</div>
                    <div class="res" style="color:#27ae60; font-size:20px;">梯度 ↑</div>
                </div>
                <div class="adv-card positive" style="text-align:center;">
                    <div class="id" style="font-weight:bold;">C2 (+0.92)</div>
                    <div class="res" style="color:#27ae60; font-size:28px;">梯度 ↑↑</div>
                </div>
                <div class="adv-card negative" style="text-align:center;">
                    <div class="id" style="font-weight:bold;">C3 (-1.38)</div>
                    <div class="res" style="color:#c0392b; font-size:28px;">梯度 ↓↓</div>
                </div>
                <div class="adv-card neutral" style="text-align:center;">
                    <div class="id" style="font-weight:bold;">C4 (0.00)</div>
                    <div class="res" style="color:#7f8c8d; font-size:20px;">不变</div>
                </div>
            </div>
        `;
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
