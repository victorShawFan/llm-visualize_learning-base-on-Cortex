document.addEventListener('DOMContentLoaded', () => {
    const visualContent = document.getElementById('visualContent');
    const infoBox = document.getElementById('infoBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const nextBtn = document.getElementById('nextStep');
    const prevBtn = document.getElementById('prevStep');
    const resetBtn = document.getElementById('reset');
    const currentStepSpan = document.getElementById('current-step');

    if (!visualContent || !infoBox) return;

    let currentStep = 0;

    // Helper to render the 4-model grid
    function renderModels(activeModel) {
        // activeModel: 'actor', 'critic', 'ref', 'reward', or 'all', or 'none'
        const models = [
            { id: 'actor', name: 'Actor (Policy)', type: 'Trainable', color: '#3498db' },
            { id: 'critic', name: 'Critic (Value)', type: 'Trainable', color: '#e67e22' },
            { id: 'ref', name: 'Ref Model', type: 'Frozen ❄️', color: '#95a5a6' },
            { id: 'reward', name: 'Reward Model', type: 'Frozen ❄️', color: '#f1c40f' }
        ];

        let html = `<div class="models-grid">`;
        models.forEach((m, idx) => {
            const isActive = activeModel === m.id || activeModel === 'all';
            const isFrozen = m.type.includes('Frozen');
            const activeClass = isActive ? 'active' : '';
            const frozenClass = isFrozen ? 'frozen' : '';
            const animClass = isActive ? 'anim-fade-scale' : '';
            
            html += `
            <div class="model-box ${activeClass} ${frozenClass} ${animClass}" style="${isActive ? 'border-color:'+m.color : ''}; animation-delay:${idx*0.1}s">
                <h4 style="${isActive ? 'color:'+m.color : ''}">${m.name}</h4>
                <p>${m.type}</p>
                ${isActive ? '<div class="data-packet" style="top:40px; left:50%; animation: flowData 1s infinite;"></div>' : ''}
            </div>`;
        });
        html += `</div>`;
        return html;
    }

    const steps = [
        {
            title: "The 4-Model Architecture",
            desc: "PPO 训练通常涉及 4 个模型：<br>1. <b>Actor</b>: 我们要训练的主模型。<br>2. <b>Critic</b>: 估计状态价值 (Value)，用于减少方差。<br>3. <b>Ref Model</b>: 原始模型的冻结副本，用于计算 KL 散度防止模型跑偏。<br>4. <b>Reward Model</b>: 也是冻结的，用于给出客观评分。",
            code: "self.actor = LlmModel(...)\nself.critic = ValueHead(...)\nself.ref_model = load_pretrained(...)\nself.reward_model = load_pretrained(...)",
            render: () => {
                let html = renderModels('none');
                html += `<div class="stage-container anim-fade-scale">
                    <p>Standard PPO Setup in Cortex</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 1: Rollout (Experience Collection)",
            desc: "首先，使用当前的 <b>Actor</b> 根据 Prompt 生成回复（Trajectory）。这是一个采样过程。",
            code: "sequence, attention_mask = self.actor.generate(prompts, ...)",
            render: () => {
                let html = renderModels('actor');
                html += `<div class="stage-container">
                    <h3>Generating Trajectory...</h3>
                    <div class="math-block anim-fade-scale">Prompt: "How to fix a bug?"</div>
                    <div class="arrow-down anim-flow-down">⬇️</div>
                    <div class="math-block highlight-blue anim-fade-scale" style="animation-delay:0.5s">Response: "Debug step by step..."</div>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 1: Reference LogProbs (KL Penalty)",
            desc: "为了防止 Actor 为了刷分而生成怪异内容（Reward Hacking），我们需要计算 Ref Model 的生成概率。这一步是为了计算 <b>KL 散度</b>，并将其作为<b>惩罚项</b>从 Reward 中扣除：如果 Actor 偏离原模型太远，KL 变大，总奖励就会大幅降低。",
            code: "with torch.no_grad():\n    ref_logprobs = self.ref_model(sequence)\n    # Later: reward = reward - beta * KL(actor, ref)",
            render: () => {
                let html = renderModels('ref');
                html += `<div class="stage-container">
                    <h3>Compute KL Penalty</h3>
                    <div class="math-block anim-fade-scale">KL = log(P_actor) - log(P_ref)</div>
                    <p class="anim-fade-scale" style="animation-delay:0.3s">If KL is high, we penalize the reward.</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 1: Reward Evaluation",
            desc: "生成的完整回复被送入 <b>Reward Model</b>，得到一个标量评分（Scalar Score）。",
            code: "rewards = self.reward_model(sequence)",
            render: () => {
                let html = renderModels('reward');
                html += `<div class="stage-container">
                    <h3>Scoring</h3>
                    <div class="math-block anim-fade-scale">Response -> [Reward Model] -> <span class="highlight-green" style="display:inline-block; animation: highlightPulse 0.5s 0.5s both;">0.85</span></div>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 1: Value Estimation (Critic)",
            desc: "Critic 模型读取每个 Token 的 Hidden State，经过线性层输出一个标量 V(s)。<br>V(s) 代表<b>“预期未来总回报”</b>。例如，如果生成了“Debug”，Critic 觉得这很靠谱，可能会预测 V=0.3；接着生成“step by step”，Critic 确信这是高质量回复，V 值一路飙升到 0.9。<br>这为计算 Advantage 提供了基准。",
            code: "# hidden_states: [Batch, Seq, Dim]\nvalues = self.critic.value_head(hidden_states)\n# Output: Scalar value per token\n# Example: [0.3, 0.5, 0.7, 0.9]",
            render: () => {
                let html = renderModels('critic');
                
                const tokens = [
                    { t: "Debug", v: 0.3 },
                    { t: "step", v: 0.5 },
                    { t: "by", v: 0.7 },
                    { t: "step", v: 0.9 }
                ];
                
                let tokenHtml = tokens.map((item, i) => `
                    <div style="display:flex; flex-direction:column; align-items:center; margin:0 5px;">
                        <div class="token-box" style="margin-bottom:5px; padding:5px 10px; border:1px solid #ccc; border-radius:4px; background:white;">"${item.t}"</div>
                        <div style="font-size:12px; color:#aaa;">⬇️</div>
                        <div class="value-box anim-fade-scale" style="animation-delay:${i*0.2}s; font-weight:bold; color:#e67e22; margin-top:5px;">V=${item.v}</div>
                    </div>
                `).join('');

                html += `<div class="stage-container">
                    <h3>Critic Scoring per Token</h3>
                    <div style="display:flex; justify-content:center; margin-top:15px; flex-wrap:wrap;">
                        ${tokenHtml}
                    </div>
                    <p style="margin-top:15px; font-size:0.9em; color:#666;">Note: Critic predicts future reward at each step.</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 2: GAE Calculation (Advantage)",
            desc: "我们如何判断一步操作好不好？通过<b>Advantage（优势函数）</b>。<br>核心逻辑是：<b>实际发生的价值 - 预期的价值</b>。<br>接着上一步的例子：Critic 预期 'Debug' 只有 0.3 分，但它引出了 value=0.5 的 'step'。这说明 'Debug' 这步走得比预期好（惊喜！），Advantage > 0，模型应该受到鼓励。",
            code: "# TD Error (Simplified GAE)\ndelta = reward + gamma * next_value - current_value\n\n# Example for token 'Debug':\n# r=0, V_next=0.5, V_curr=0.3, gamma=0.99\n# Adv ≈ 0 + 0.99*0.5 - 0.3 = +0.195 (Positive!)",
            render: () => {
                let html = renderModels('none');
                html += `<div class="stage-container">
                    <h3>Computing Advantage (Step: "Debug")</h3>
                    
                    <div style="display:flex; justify-content:center; align-items:center; gap:20px; margin: 20px 0;">
                        <div class="math-item" style="text-align:center;">
                            <div style="font-size:12px; color:#aaa;">Next State Value</div>
                            <div style="font-size:1.5em; color:#2ecc71;">0.5</div>
                            <div style="font-size:12px;">(Future is bright)</div>
                        </div>
                        <div style="font-size:2em;">-</div>
                        <div class="math-item" style="text-align:center;">
                            <div style="font-size:12px; color:#aaa;">Current Value</div>
                            <div style="font-size:1.5em; color:#e74c3c;">0.3</div>
                            <div style="font-size:12px;">(Expectation)</div>
                        </div>
                        <div style="font-size:2em;">=</div>
                        <div class="math-item anim-bounce" style="text-align:center; border: 2px solid #3498db; padding: 10px; border-radius: 8px;">
                            <div style="font-size:12px; color:#3498db; font-weight:bold;">Advantage</div>
                            <div style="font-size:1.5em; color:#3498db;">+0.2</div>
                            <div style="font-size:12px;">(Better than expected!)</div>
                        </div>
                    </div>

                    <div class="math-block anim-fade-scale">δ = r + γV(s') - V(s)</div>
                    <p style="font-size:0.9em; color:#666;">Adv > 0: Action was good -> Increase Prob<br>Adv < 0: Action was bad -> Decrease Prob</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 3: PPO Loss (Actor Update)",
            desc: "既然 'Debug' 是好动作 (Adv=+0.2)，我们希望<b>提高</b>它的生成概率。但 PPO 的核心哲学是：<b>“别改太猛！”</b>。<br>我们计算 <b>Ratio = 新概率 / 旧概率</b>。如果 Ratio 超过了安全范围（比如 1.2 倍），我们就<b>截断 (Clip)</b> 它，只保留边界收益。这防止了模型因为一次错误的奖励而彻底跑偏。",
            code: "ratio = new_prob / old_prob # e.g. 0.26 / 0.20 = 1.3\nlimit = 1 + epsilon     # e.g. 1 + 0.2 = 1.2\n\n# Ratio(1.3) > Limit(1.2) -> CLIP triggered!\n# We only credit the update up to 1.2x",
            render: () => {
                let html = renderModels('actor');
                html += `<div class="stage-container">
                    <h3>PPO Clipping in Action</h3>
                    
                    <div style="display:flex; justify-content:space-around; align-items:flex-end; height:120px; margin-bottom:10px; border-bottom:1px solid #ddd;">
                        <div style="text-align:center">
                            <div style="margin-bottom:5px; font-size:12px;">Old Prob</div>
                            <div style="height:60px; width:40px; background:#bdc3c7; margin:0 auto; position:relative;">
                                <span style="position:absolute; top:-20px; left:0; right:0;">0.20</span>
                            </div>
                        </div>
                        <div style="text-align:center; position:relative;">
                             <div style="margin-bottom:5px; font-size:12px; font-weight:bold; color:#3498db;">New Prob (Updated)</div>
                             <div class="anim-grow-height" style="width:40px; background:#3498db; margin:0 auto; height:78px;">
                                <span style="position:absolute; top:-20px; left:0; right:0;">0.26</span>
                             </div>
                             <div style="font-size:10px; margin-top:5px; color:#3498db;">Ratio = 1.3x</div>
                        </div>
                        <div style="text-align:center; position:relative; width: 80px;">
                            <div style="position:absolute; bottom:72px; width:100%; border-top: 2px dashed #e74c3c; z-index:10;">
                                <span style="background:white; color:#e74c3c; font-size:10px; padding:0 2px;">Max Limit (1.2x)</span>
                            </div>
                            <div class="anim-fade-scale" style="height:78px; width:40px; border:2px solid #3498db; background:rgba(52, 152, 219, 0.1); margin:0 auto; position:relative; animation-delay:0.5s;">
                                <div style="position:absolute; bottom:0; left:0; right:0; height:72px; background:repeating-linear-gradient(45deg,#2ecc71,#2ecc71 5px,#27ae60 5px,#27ae60 10px); opacity:0.7;"></div>
                            </div>
                            <div style="font-size:10px; margin-top:5px; color:#e74c3c; font-weight:bold;">CLIPPED to 1.2!</div>
                        </div>
                    </div>
                    
                    <p style="font-size:0.9em; text-align:center;">Safety First: Even though the model wants to change by 30%,<br>we only allow a 20% update to remain stable.</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 3: Critic Loss (Value Correction)",
            desc: "Critic 之前预测 'Debug' 的价值只有 <b>0.3</b>。但实际上，这条回复最后得了高分，真实的 <b>Return (回报)</b> 算下来高达 <b>0.9</b>。<br>Critic 犯了错（严重低估），产生了误差。我们需要最小化这个 <b>MSE Loss</b>，强迫 Critic 下次大胆一点，把预测值往 0.9 拉近。",
            code: "# Target (Actual Return) = 0.9\n# Prediction (V_old) = 0.3\n\nloss = (prediction - target) ** 2\nloss = (0.3 - 0.9) ** 2 = 0.36\n# Gradient Update: Critic weights shift to predict higher.",
            render: () => {
                let html = renderModels('critic');
                html += `<div class="stage-container">
                    <h3>Correcting the Critic</h3>
                    
                    <div style="display:flex; justify-content:center; align-items:flex-end; height:150px; gap:40px; margin-bottom:10px;">
                        
                        <!-- Prediction -->
                        <div style="text-align:center; width:60px;">
                            <div style="margin-bottom:5px; font-size:12px; color:#e67e22;">Predicted</div>
                            <div style="height:45px; width:100%; background:#e67e22; border-radius:4px 4px 0 0; position:relative;">
                                <span style="position:absolute; top:-20px; left:0; right:0; font-weight:bold;">0.3</span>
                            </div>
                            <div style="margin-top:5px; font-size:12px;">Too Low!</div>
                        </div>

                        <!-- Error Arrow -->
                        <div style="text-align:center; padding-bottom:30px;">
                            <div style="color:#e74c3c; font-weight:bold; margin-bottom:5px;">Error</div>
                            <div style="font-size:2em;">⬅️</div>
                            <div style="font-size:10px; color:#aaa;">Pull Up</div>
                        </div>

                        <!-- Target -->
                        <div style="text-align:center; width:60px;">
                            <div style="margin-bottom:5px; font-size:12px; color:#27ae60;">Actual Return</div>
                            <div class="anim-fade-scale" style="height:135px; width:100%; background:#27ae60; border-radius:4px 4px 0 0; position:relative; opacity:0.8;">
                                <span style="position:absolute; top:-20px; left:0; right:0; font-weight:bold;">0.9</span>
                            </div>
                            <div style="margin-top:5px; font-size:12px;">Truth</div>
                        </div>
                    </div>

                    <div class="math-block">Loss = (0.3 - 0.9)² = 0.36</div>
                    <p style="font-size:0.9em; color:#666;">Critic Update: "Next time I see 'Debug', I'll guess 0.35!"</p>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 3: Backpropagation (Global Update)",
            desc: "最后，我们将 Actor 的改进需求 (Loss=-0.24) 和 Critic 的修正需求 (Loss=0.36) 合并。<br>注意：Critic 的 Loss 通常是个无界的 MSE 值，比较大，所以我们通常乘以系数 (0.1) 把它<b>缩小</b>，防止它喧宾夺主，干扰了 Actor 的训练。",
            code: "# Actor Loss (Objective) ≈ -0.24\n# Critic Loss (MSE) = 0.36\n\ntotal_loss = actor_loss + 0.1 * critic_loss\n           = -0.24 + 0.036 = -0.204\n\ntotal_loss.backward() # Update all weights",
            render: () => {
                let html = renderModels('all');
                html += `<div class="stage-container">
                    <h3>Weighted Loss Combination</h3>
                    
                    <div style="display:flex; justify-content:center; align-items:center; gap:15px; margin:20px 0;">
                        <!-- Actor Component -->
                        <div class="math-item" style="text-align:center; opacity:0; animation: fadeInScale 0.5s forwards;">
                            <div style="font-size:12px; color:#3498db;">Actor Loss</div>
                            <div style="font-size:1.2em; font-weight:bold;">-0.24</div>
                        </div>

                        <div style="font-size:1.5em; color:#aaa;">+</div>

                        <!-- Critic Component -->
                        <div class="math-item" style="text-align:center; opacity:0; animation: fadeInScale 0.5s 0.3s forwards;">
                            <div style="font-size:12px; color:#e67e22;">0.1 × Critic Loss</div>
                            <div style="font-size:1.2em; font-weight:bold;">0.036</div>
                            <div style="font-size:10px; color:#aaa;">(0.1 × 0.36)</div>
                        </div>

                        <div style="font-size:1.5em; color:#aaa;">=</div>

                        <!-- Total -->
                        <div class="math-item" style="text-align:center; border:2px solid #8e44ad; padding:10px; border-radius:8px; background:rgba(142, 68, 173, 0.1); opacity:0; animation: fadeInScale 0.5s 0.6s forwards;">
                            <div style="font-size:12px; color:#8e44ad; font-weight:bold;">Total Loss</div>
                            <div style="font-size:1.5em; color:#8e44ad;">-0.204</div>
                        </div>
                    </div>

                    <div class="anim-flow-up" style="text-align:center; margin-top:10px; color:#8e44ad; font-weight:bold;">
                        ⬆️ Gradients Flowing Back ⬆️
                    </div>
                </div>`;
                return html;
            }
        }
    ];

    function renderStep(index) {
        if (index < 0) index = 0;
        if (index >= steps.length) index = steps.length - 1;
        
        currentStep = index;
        const stepData = steps[index];

        currentStepSpan.textContent = currentStep + 1;
        infoBox.innerHTML = `<h3>${stepData.title}</h3><p>${stepData.desc}</p>`;
        codeSnippet.textContent = stepData.code || "";
        visualContent.innerHTML = stepData.render();

        prevBtn.disabled = currentStep === 0;
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    nextBtn.addEventListener('click', () => renderStep(currentStep + 1));
    prevBtn.addEventListener('click', () => renderStep(currentStep - 1));
    resetBtn.addEventListener('click', () => renderStep(0));

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') renderStep(currentStep - 1);
        if (e.key === 'ArrowRight') renderStep(currentStep + 1);
    });

    renderStep(0);
});
