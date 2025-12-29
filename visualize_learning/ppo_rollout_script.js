document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const resetBtn = document.getElementById('reset-btn');
    const container = document.getElementById('rollout-container');
    const currentStepSpan = document.getElementById('current-step');
    const stepTitle = document.getElementById('step-title');
    const stepDesc = document.getElementById('step-desc');
    const stepBadge = document.querySelector('.step-badge');

    // Guard
    if (!container) {
        console.error("Required elements not found in ppo_rollout_script");
        return;
    }

    const steps = [
        {
            title: "步骤 0: 批量生成 (Policy Generation)",
            desc: "对一批 Prompts 执行 batch_generate。通过 <code>unwrap_model_for_generation</code> 解包并行包装，同时在构建模型时已经关闭 Dropout。输出 <code>full_ids</code> (Prompt + Completion) 和生成过程的 <code>logitss</code>。",
            badge: "ppo_trainer.py:299",
            state: "gen"
        },
        {
            title: "步骤 1: LogProbs 计算 (Action Probabilities)",
            desc: "基于生成时的 Logits，计算每个 token 在当前策略下的对数概率 <code>old_log_probs</code>。这是 PPO 计算 Ratio (r_t) 的分母。",
            badge: "ppo_trainer.py:321",
            state: "logprobs"
        },
        {
            title: "步骤 2: 价值估计 & 参考模型 (Values & Ref)",
            desc: "并行执行两项前向传播：<br>1. Value Model 估计状态价值 <code>values</code> (用于 GAE)。<br>2. Reference Model 计算 <code>ref_log_probs</code> (用于 KL 惩罚)。",
            badge: "ppo_trainer.py:315",
            state: "val_ref"
        },
        {
            title: "步骤 3: 奖励合成 (Reward Computation)",
            desc: "计算 KL 散度惩罚项，并调用外部 <code>reward_func</code> 获取环境奖励。最终 Reward = KL 奖励 (dense) + Env Reward (sparse, 通常只在 EOS 上非零)。如果某条序列完全没有生成 <code>EOS</code>，源码会通过 <code>missing_eos_penalty</code> 在该条样本的环境奖励上施加额外惩罚。",
            badge: "ppo_trainer.py:335",
            state: "reward"
        },
        {
            title: "步骤 4: 数据打包 (Rollout Dictionary)",
            desc: "将核心张量 (prompt_ids, completion_ids, old_log_probs, values, rewards, dones 等) 打包成字典，准备传入 <code>_ppo_learning_phase</code> 进行 GAE 计算和多 epoch 更新。",
            badge: "ppo_trainer.py:370",
            state: "pack"
        },
        {
            title: "步骤 5: 序列长度与 Padding 记录",
            desc: "在构造 rollout 时通过 <code>completion_mask</code> 等掩码记录哪些位置是有效生成 token，后续 GAE 与 loss 计算会根据这些 mask 裁剪掉纯 PAD 区域，避免无效梯度。",
            badge: "ppo_trainer.py:304",
            state: "reward"
        },
        {
            title: "步骤 6: Prompt / Completion 切分",
            desc: "<code>full_ids</code> 会按照 <code>prompt_len</code> 被切分为 <code>prompt_ids</code> 与 <code>completion_ids</code> 两部分，只对 completion 段计算 KL 与环境奖励；Prompt 部分只参与条件建模。",
            badge: "ppo_trainer.py:308",
            state: "gen"
        },
        {
            title: "步骤 7: 对齐 values / log_probs / rewards",
            desc: "Value 轨迹、old_log_probs、ref_log_probs 与 rewards 都会被对齐到相同的时间维度，保证每个时间步的 (s_t, a_t, r_t, v_t) 一一对应，方便后续 GAE 递推。",
            badge: "ppo_trainer.py:315-341",
            state: "val_ref"
        },
        {
            title: "步骤 8: 多样本拼接成大 Rollout Batch",
            desc: "同一个 RL batch 内多条样本的 rollout 会在 batch 维度上组成一个 <code>rollout_batch</code>，随后在 <code>_ppo_learning_phase</code> 中按 mini-batch 切分做多 epoch 更新。",
            badge: "ppo_trainer.py:364",
            state: "gen"
        },
        {
            title: "步骤 9: 将 Rollout 喂入 PPO 主循环",
            desc: "最终构造出的 <code>rollout_dict</code> 被传入 <code>_ppo_learning_phase</code>，在其中完成 GAE、ratio 计算和多 epoch 小批量更新。这一阶段不再重新生成 Token，只消费本步骤打包好的轨迹数据。",
            badge: "ppo_trainer.py:372",
            state: "pack"
        }
    ];

    let currentStep = 0;

    function render() {
        if (!container) return;
        const step = steps[currentStep];
        if (currentStepSpan) currentStepSpan.innerText = currentStep + 1;
        if (stepTitle) stepTitle.innerText = step.title;
        if (stepDesc) stepDesc.innerHTML = step.desc;
        if (stepBadge) stepBadge.innerText = step.badge;

        container.innerHTML = '';
        
        try {
            if (step.state === 'gen') {
                renderGen();
            } else if (step.state === 'logprobs') {
                renderLogProbs();
            } else if (step.state === 'val_ref') {
                renderValRef();
            } else if (step.state === 'reward') {
                renderReward();
            } else if (step.state === 'pack') {
                renderPack();
            }
        } catch (e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    function updateButtons() {
        if (prevBtn) prevBtn.disabled = currentStep === 0;
        if (nextBtn) nextBtn.innerText = currentStep === steps.length - 1 ? "完成" : "下一步";
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

    function renderGen() {
        container.innerHTML = `
            <div class="rollout-flow">
                <div class="model-box policy">Policy Model (Actor)</div>
                <div class="arrow-down">⬇️ Generate (Sampling)</div>
                <div class="seq-viz">
                    <span class="token prompt" style="background:#e8eaf6; color:#3f51b5;">Prompt</span>
                    <span class="token gen" style="animation: popIn 0.5s forwards;">Token 1</span>
                    <span class="token gen" style="animation: popIn 0.5s 0.2s forwards;">Token 2</span>
                    <span class="token gen" style="animation: popIn 0.5s 0.4s forwards; background:#fbe9e7; color:#c0392b;">EOS</span>
                </div>
                <div class="data-tag">Outputs: full_ids [B, L], logitss [B, L, V]</div>
            </div>
        `;
    }

    function renderLogProbs() {
        container.innerHTML = `
            <div class="calc-box" style="text-align:center;">
                <div class="math-row" style="font-family:monospace; margin-bottom:10px;">Logits [B, L, V]</div>
                <div class="arrow-down">⬇️ log_softmax(gather index=completion_ids)</div>
                <div class="tensor-viz" style="display:flex; justify-content:center; gap:5px; margin-top:10px;">
                    <div class="tensor-cell" style="background:#d4e6f1;">-0.10</div>
                    <div class="tensor-cell" style="background:#d4e6f1;">-0.52</div>
                    <div class="tensor-cell" style="background:#a9cce3;">-0.01</div>
                </div>
                <div class="data-tag" style="margin-top:10px; font-weight:bold; color:#2980b9;">old_log_probs (π_old)</div>
            </div>
        `;
    }

    function renderValRef() {
        container.innerHTML = `
            <div class="parallel-ops" style="display:flex; justify-content:center; gap:40px;">
                <div class="branch" style="text-align:center;">
                    <div class="model-box value" style="border:2px solid #8e44ad; padding:10px; border-radius:8px;">Value Model</div>
                    <div class="arrow-down" style="margin:10px 0;">⬇️</div>
                    <div class="tensor-viz value-color" style="display:flex; flex-direction:column; gap:5px;">
                        <div class="tensor-cell" style="background:#f5eef8;">0.50</div>
                        <div class="tensor-cell" style="background:#ebdef0;">0.85</div>
                        <div class="tensor-cell" style="background:#d7bde2;">1.20</div>
                    </div>
                    <div class="data-tag" style="color:#8e44ad; font-weight:bold;">values V(s)</div>
                </div>
                <div class="branch" style="text-align:center;">
                    <div class="model-box ref" style="border:2px solid #7f8c8d; padding:10px; border-radius:8px; opacity:0.8;">Ref Model</div>
                    <div class="arrow-down" style="margin:10px 0;">⬇️</div>
                    <div class="tensor-viz" style="display:flex; flex-direction:column; gap:5px;">
                        <div class="tensor-cell" style="background:#f2f3f4;">-0.12</div>
                        <div class="tensor-cell" style="background:#e5e7e9;">-0.45</div>
                        <div class="tensor-cell" style="background:#d7dbdd;">-0.08</div>
                    </div>
                    <div class="data-tag" style="color:#7f8c8d; font-weight:bold;">ref_log_probs</div>
                </div>
            </div>
        `;
    }

    function renderReward() {
        container.innerHTML = `
            <div class="reward-calc" style="width:100%; max-width:500px; margin:0 auto;">
                <div class="reward-row" style="display:flex; align-items:center; margin-bottom:10px;">
                    <div class="r-label" style="width:120px; font-weight:bold;">KL Penalty</div>
                    <div class="r-bar negative" style="background:#fadbd8; color:#c0392b; padding:5px 10px; border-radius:4px; flex-grow:1;">Dense: π vs Ref divergence</div>
                </div>
                <div class="reward-row" style="display:flex; align-items:center; margin-bottom:10px;">
                    <div class="r-label" style="width:120px; font-weight:bold;">Env Reward</div>
                    <div class="r-bar positive" style="background:#d5f5e3; color:#27ae60; padding:5px 10px; border-radius:4px; flex-grow:1;">Sparse: Score at EOS (+1.5)</div>
                </div>
                <hr style="border:0; border-top:1px dashed #ccc; margin:15px 0;">
                <div class="tensor-viz reward-color" style="display:flex; justify-content:center; gap:5px;">
                    <div class="tensor-cell" style="background:#fadbd8;">-0.01</div>
                    <div class="tensor-cell" style="background:#fadbd8;">-0.02</div>
                    <div class="tensor-cell" style="background:#d5f5e3; font-weight:bold;">+1.48</div>
                </div>
                <div class="data-tag" style="text-align:center; margin-top:5px; color:#2c3e50;">Final Rewards Tensor</div>
            </div>
        `;
    }

    function renderPack() {
        container.innerHTML = `
            <div class="dict-viz" style="background:#f8f9fa; padding:20px; border-radius:10px; font-family:monospace;">
                <div style="margin-bottom:10px; font-weight:bold; color:#2c3e50;">rollout_batch = {</div>
                <div class="dict-entry" style="padding-left:20px;">'prompt_ids': Tensor[B, Lp],</div>
                <div class="dict-entry" style="padding-left:20px;">'completion_ids': Tensor[B, Lc],</div>
                <div class="dict-entry" style="padding-left:20px;">'old_log_probs': Tensor[B, Lc],</div>
                <div class="dict-entry" style="padding-left:20px;">'values': Tensor[B, Lc],</div>
                <div class="dict-entry" style="padding-left:20px;">'rewards': Tensor[B, Lc],</div>
                <div class="dict-entry" style="padding-left:20px;">'dones': Tensor[B, Lc]</div>
                <div style="margin-top:10px; font-weight:bold; color:#2c3e50;">}</div>
            </div>
            <div class="next-phase" style="text-align:center; margin-top:20px; color:#2980b9; font-weight:bold;">
                Ready for PPO Learning Phase (GAE & Update) ➜
            </div>
        `;
    }

    if(nextBtn) nextBtn.addEventListener('click', goNext);
    if(prevBtn) prevBtn.addEventListener('click', goPrev);
    if(resetBtn) resetBtn.addEventListener('click', () => {
        currentStep = 0;
        render();
    });

    render();

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });
});
