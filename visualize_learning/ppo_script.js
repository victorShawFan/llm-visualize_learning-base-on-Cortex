document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
        console.error("Required elements not found in PPO script");
        return;
    }

    let currentStep = 0;
    let currentInterval = null; // Track active intervals for cleanup

    const steps = [
        {
            title: "Phase 1: 模型架构 (Actor-Critic)",
            description: "PPO 训练涉及四个角色：<br>1. <b>Policy (Actor)</b>: 正在训练的生成模型。<br>2. <b>Reference Model</b>: 冻结的基准模型 (KL 锚点)。<br>3. <b>Value Model (Critic)</b>: 预测状态价值 $V(s)$。<br>4. <b>Reward Model</b>: 外部打分模型 (Frozen)。<br><span class='step-badge'>ppo_trainer.py:31-72</span>",
            code: "class PolicyAndValueModelWrapper(nn.Module):\n    # 共享基础权重，分别输出动作概率和状态价值\n    return policy_out, value_out",
            render: () => renderSetup()
        },
        {
            title: "Phase 2: 采样与生成 (Rollout)",
            description: "使用 Policy Model 生成 Completion。记录每一步的概率分布（Old LogProbs）。<br><span class='step-badge'>ppo_trainer.py:299</span>",
            code: "full_ids, logitss = batch_generate(policy_model, ...)\nold_log_probs = log_softmax(logitss, completion_ids)",
            render: () => renderGeneration()
        },
        {
            title: "Phase 3: 奖励信号融合 (Reward Engineering)",
            description: "将 <b>KL 惩罚</b>（防止跑偏）与 <b>外部奖励</b>（如 RM 分数）结合。注意：环境奖励通常只加在序列的<b>最后一个 Token</b> 上。<br><span class='step-badge'>ppo_trainer.py:338-368</span>",
            code: "rewards = -beta * kl_penalty\nrewards[:, -1] += env_reward  # 仅在末尾注入主奖励",
            render: () => renderRewards()
        },
        {
            title: "Phase 4: GAE 优势估计 (Advantage Calculation)",
            description: "使用 GAE 算法从后往前递归计算优势 $A_t$。它衡量了：在该状态下采取该动作，比平均水平好多少？<br><span class='step-badge'>ppo_trainer.py:253-261</span>",
            code: "delta = r_t + gamma * V_{t+1} - V_t\nlast_gae = delta + gamma * lam * last_gae",
            render: () => renderGAE()
        },
        {
            title: "Phase 5: 截断优化 (Clipped Update)",
            description: "计算 $r_t = \\frac{\\pi_{new}}{\\pi_{old}}$。如果 $r_t$ 偏离 [0.8, 1.2] 太远且 Advantage 为正，则强行截断梯度，确保训练稳健。<br><span class='step-badge'>loss.py:270</span>",
            code: "surr1 = ratio * advantages\nsurr2 = clamp(ratio, 1-eps, 1+eps) * advantages\nloss = -min(surr1, surr2).mean()",
            render: () => renderClip()
        },
        {
            title: "Phase 6: 价值损失与总 Loss",
            description: "同步更新 Value Model，使其预测的 $V(s)$ 尽可能接近实际观测到的累积回报（Returns）。<br><span class='step-badge'>loss.py:275-283</span>",
            code: "value_loss = (current_values - returns) ** 2\ntotal_loss = policy_loss + vf_coef * value_loss",
            render: () => renderTotalLoss()
        },
        {
            title: "Phase 7: 多 Epoch 迭代 (Replay Buffer)",
            description: "采样一次 Rollout 后，模型会在这些数据上反复学习多轮（<code>ppo_epochs</code>），通过 Mini-batch SGD 挤干数据的价值。<br><span class='step-badge'>ppo_trainer.py:430</span>",
            code: "for ppo_epoch in range(ppo_config.ppo_epochs):\n    indices = torch.randperm(batch_size)\n    # 随机打乱并分批更新",
            render: () => renderBatching()
        }
    ];

    function updateUI() {
        // Clear any active interval from previous step
        if (currentInterval) {
            clearInterval(currentInterval);
            currentInterval = null;
        }

        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        codeSnippet.textContent = step.code;
        
        if (window.hljs) {
            hljs.highlightElement(codeSnippet);
        }

        visualContent.innerHTML = '';
        visualContent.className = 'ppo-viz fade-in';
        
        try {
            step.render();
        } catch (e) {
            console.error("Render error:", e);
        }

        updateButtons();
    }

    // --- Renderers ---

    function renderSetup() {
        visualContent.innerHTML = `
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:30px; width:90%;">
                <div class="group-box" style="border:2px dashed #e74c3c; padding:15px; border-radius:10px;">
                    <div style="text-align:center; font-weight:bold; margin-bottom:10px; color:#e74c3c;">Trainable (PPO Updates)</div>
                    <div style="display:flex; justify-content:center; gap:20px;">
                        <div class="model-box policy">
                            <div class="model-title">Policy (Actor)</div>
                            <div class="model-desc">Generate Text (π)</div>
                        </div>
                        <div class="model-box value">
                            <div class="model-title">Value (Critic)</div>
                            <div class="model-desc">Predict Score (V)</div>
                        </div>
                    </div>
                </div>
                
                <div class="group-box" style="border:2px dashed #95a5a6; padding:15px; border-radius:10px;">
                    <div style="text-align:center; font-weight:bold; margin-bottom:10px; color:#7f8c8d;">Frozen (Baselines)</div>
                    <div style="display:flex; justify-content:center; gap:20px;">
                        <div class="model-box ref" style="opacity:0.8;">
                            <div class="model-title">Ref Model</div>
                            <div class="model-desc">KL Anchor (π_ref)</div>
                        </div>
                        <div class="model-box ref" style="opacity:0.8; border-color:#f1c40f;">
                            <div class="model-title">Reward Model</div>
                            <div class="model-desc">External Score</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function renderGeneration() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:20px;">
                <div class="model-box policy">Policy Model (Sampling)</div>
                <div class="arrow">↓ Generate Token by Token</div>
                <div class="seq-viz">
                    <div class="token prompt">Prompt</div>
                    <div id="gen-area" style="display:flex; gap:5px;"></div>
                </div>
                <div id="logp-viz" style="display:flex; gap:5px; margin-top:5px; height:20px; opacity:0;"></div>
            </div>
        `;
        
        const tokens = ["Sure", ",", " quantum", " physics", " is", "..."];
        const logps = ["-0.1", "-0.01", "-1.2", "-0.5", "-0.2", "-0.8"];
        const area = document.getElementById('gen-area');
        const logpArea = document.getElementById('logp-viz');
        
        if (!area || !logpArea) return;

        let idx = 0;
        
        // Reset animation
        logpArea.style.opacity = '1';
        
        currentInterval = setInterval(() => {
            if(idx >= tokens.length) {
                if (currentInterval) clearInterval(currentInterval);
                currentInterval = null;
                return;
            }
            // Guard against navigating away while interval runs
            if (!document.getElementById('gen-area')) return;

            const t = document.createElement('div');
            t.className = 'token gen';
            t.innerText = tokens[idx];
            t.style.animation = "popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
            area.appendChild(t);
            
            // Visualize logprob bar
            const bar = document.createElement('div');
            const val = Math.abs(parseFloat(logps[idx]));
            bar.style.width = '40px'; 
            bar.style.height = '4px';
            bar.style.marginTop = '10px';
            bar.style.background = `rgba(52, 152, 219, ${1/(val+0.5)})`; // Higher prob = darker blue
            bar.title = `LogP: ${logps[idx]}`;
            logpArea.appendChild(bar);
            
            idx++;
        }, 300);
    }

    function renderRewards() {
        visualContent.innerHTML = `
            <div class="reward-calc" style="width: 100%;">
                <div class="reward-row">
                    <div class="r-label">Step 1</div>
                    <div class="r-bar negative" style="width:40px">-0.1 KL</div>
                </div>
                <div class="reward-row">
                    <div class="r-label">Step 2</div>
                    <div class="r-bar negative" style="width:30px">-0.05 KL</div>
                </div>
                <div class="reward-row">
                    <div class="r-label">Step 3</div>
                    <div class="r-bar negative" style="width:60px">-0.2 KL</div>
                </div>
                <div class="reward-row">
                    <div class="r-label">Step 4 (End)</div>
                    <div class="r-bar positive" style="width:150px; position:relative;">
                        +1.0 Env Reward
                        <div style="position:absolute; right:-120px; top:0; color:#f39c12; font-size:0.8em;">← From Reward Model</div>
                    </div>
                </div>
                <div class="sum-line"></div>
                <div style="font-weight:bold; color:#2c3e50; text-align:center;">Total Reward R(t) = r_env + r_kl</div>
            </div>
        `;
    }

    function renderGAE() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; width:100%;">
                <div class="gae-chain" style="display:flex; gap:10px; align-items:center;">
                    <div class="gae-step" id="gs-1">T1</div>
                    <div class="arrow">←</div>
                    <div class="gae-step" id="gs-2">T2</div>
                    <div class="arrow">←</div>
                    <div class="gae-step" id="gs-3">T3</div>
                    <div class="arrow">←</div>
                    <div class="gae-step active" id="gs-4">T4 (End)</div>
                </div>
                
                <div style="display:flex; gap:20px; margin-top:20px;">
                    <div style="text-align:center;">
                        <div style="font-size:0.8em; color:#666;">Current R</div>
                        <div class="tensor-cell" style="background:#e8f8f5;">+1.0</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.8em; color:#666;">Value V(s)</div>
                        <div class="tensor-cell" style="background:#ebf5fb;">0.8</div>
                    </div>
                    <div style="font-size:24px; align-self:center;">➜</div>
                    <div style="text-align:center;">
                        <div style="font-size:0.8em; color:#666;">Advantage</div>
                        <div class="tensor-cell" style="background:#fff3cd; font-weight:bold;">+0.2</div>
                    </div>
                </div>

                <div id="gae-formula" style="margin-top:20px; font-family:monospace; background:#f0f4f8; padding:15px; border-radius:10px; border-left:4px solid #3498db; width:80%;">
                    Adv_4 = Reward_4 - Value_4
                </div>
                <div class="label-text" style="margin-top:10px;">GAE 从序列末尾向开头递归计算，利用 V(s) 减小方差</div>
            </div>
        `;
        
        let t = 4;
        const formulas = [
            "Adv_4 = R_4 - V_4",
            "Adv_3 = δ_3 + γλ * Adv_4",
            "Adv_2 = δ_2 + γλ * Adv_3",
            "Adv_1 = δ_1 + γλ * Adv_2"
        ];
        
        currentInterval = setInterval(() => {
            if(t < 1) { 
                if (currentInterval) clearInterval(currentInterval); 
                currentInterval = null;
                return; 
            }
            
            const stepEl = document.getElementById(`gs-${t}`);
            const formulaEl = document.getElementById('gae-formula');
            
            // Safety check
            if(!stepEl || !formulaEl) return;
            
            document.querySelectorAll('.gae-step').forEach(s => s.classList.remove('active'));
            stepEl.classList.add('active');
            formulaEl.innerText = formulas[4-t];
            t--;
        }, 1500);
    }

    function renderClip() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:20px; width:100%;">
                <div style="position:relative; width:320px; height:200px; border-left:2px solid #333; border-bottom:2px solid #333; background:white;">
                    <div style="position:absolute; bottom:-25px; left:50%; transform:translateX(-50%); font-size:12px;">Probability Ratio r(θ) = π_new / π_old</div>
                    <div style="position:absolute; left:-40px; top:50%; transform:rotate(-90deg) translateX(50%); font-size:12px;">Surrogate Objective</div>
                    
                    <!-- Clip Range -->
                    <div style="position:absolute; left:110px; top:0; width:100px; height:200px; background:rgba(46, 204, 113, 0.1); border-left:1px dashed #27ae60; border-right:1px dashed #27ae60;"></div>
                    <div style="position:absolute; bottom:-18px; left:110px; font-size:10px; color:#27ae60;">1-ε (0.8)</div>
                    <div style="position:absolute; bottom:-18px; left:210px; font-size:10px; color:#27ae60;">1+ε (1.2)</div>
                    
                    <!-- Curves -->
                    <svg width="320" height="200" style="position:absolute; top:0; left:0;">
                        <!-- Unclipped -->
                        <path d="M 0 180 L 160 100 L 320 20" stroke="#e74c3c" stroke-width="1" fill="none" stroke-dasharray="4,2"/>
                        <!-- Clipped (for positive advantage) -->
                        <path d="M 0 180 L 110 125 L 210 100 L 320 100" stroke="#2ecc71" stroke-width="3" fill="none"/>
                    </svg>
                    
                    <div style="position:absolute; top:10px; right:10px; font-size:10px; background:rgba(255,255,255,0.8); padding:5px; border:1px solid #ddd;">
                        <span style="color:#2ecc71">━</span> PPO Objective<br>
                        <span style="color:#e74c3c; opacity:0.5;">---</span> Raw Policy Gradient
                    </div>
                </div>
                <div style="font-size:0.85em; color:#e67e22; font-weight:bold;">Example: Adv > 0 (Good action). We clamp update if prob ratio > 1.2</div>
            </div>
        `;
    }

    function renderTotalLoss() {
        visualContent.innerHTML = `
            <div class="math-flow">
                <div class="row" style="display:flex; gap:10px; align-items:center;">
                    <div class="box" style="background:#e74c3c; color:white; border:none;">Policy Loss<br><span style="font-size:0.7em;">(Maximize Advantage)</span></div>
                    <div class="op">+</div>
                    <div class="box" style="background:#8e44ad; color:white; border:none;">Value Loss<br><span style="font-size:0.7em;">(MSE: Predict Returns)</span></div>
                    <div class="op">-</div>
                    <div class="box" style="background:#34495e; color:white; border:none;">Entropy<br><span style="font-size:0.7em;">(Exploration Bonus)</span></div>
                </div>
                <div class="arrow">↓ Backward</div>
                <div class="box total" style="font-weight:bold; border:2px solid #2c3e50;">Update Policy & Value Nets</div>
            </div>
        `;
    }

    function renderBatching() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:15px; align-items:center;">
                <div style="text-align:center; font-size:0.9em; color:#666; margin-bottom:10px;">Rollout Buffer (N samples)</div>
                <div class="tensor-viz" style="width:300px; flex-wrap:wrap;">
                    ${Array(12).fill(0).map((_,i) => `<div class="tensor-cell" style="width:20px; height:20px; font-size:8px; background:${i%2==0?'#a9dfbf':'#aed6f1'};">${i}</div>`).join('')}
                </div>
                <div class="arrow">Shuffle & Mini-Batching (Epochs)</div>
                <div style="display:flex; gap:20px;">
                    <div class="dict-entry" style="width:80px; background:#f9e79f;">Batch 1<br><small>Update</small></div>
                    <div class="dict-entry" style="width:80px; background:#f9e79f;">Batch 2<br><small>Update</small></div>
                    <div class="dict-entry" style="width:80px; background:#f9e79f;">Batch 3<br><small>Update</small></div>
                </div>
            </div>
        `;
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

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentStep = 0;
            updateUI();
        });
    }

    // Decoupled Keyboard Navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    updateUI();
});
