document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('loss-container');
    const currentStepSpan = document.getElementById('current-step');
    const stepTitle = document.getElementById('step-title');
    const stepDesc = document.getElementById('step-desc');
    const stepBadge = document.querySelector('.step-badge');
    const nextBtn = document.getElementById('next-btn');
    const prevBtn = document.getElementById('prev-btn');
    const resetBtn = document.getElementById('reset-btn');

    if (!container || !currentStepSpan || !stepTitle || !stepDesc || !stepBadge || !nextBtn || !prevBtn) {
        console.error("Required elements not found in Loss Impl script");
        return;
    }

    const steps = [
        {
            title: "Step 0: Logits 与 Labels 错位 (Shift & Flatten)",
            desc: "<b>关键预处理：</b>LLM 的任务是预测下一个 Token。因此，Logits 的第 `t` 个时间步对应的是 Labels 的第 `t+1` 个时间步。代码通过切片 `logits[..., :-1, :].contiguous()` 和 `labels[..., 1:].contiguous()` 实现了这种错位对齐。<span class='step-badge'>loss.py:52-53</span>",
            badge: "loss.py:52-53",
            state: "shift"
        },
        {
            title: "Step 1: 关键 Token 加权 (Weighted Loss)",
            desc: "对于某些关键 Token（如 `<EOS>` 或特定领域词汇），我们希望模型给予更多关注。<code>LMLoss</code> 初始化时通过 `register_buffer` 创建权重向量，并将 `critical_tokens` 的权重设为 `critical_alpha`（>1.0）。<span class='step-badge'>loss.py:37</span>",
            badge: "loss.py:37",
            state: "weighting"
        },
        {
            title: "Step 2: 交叉熵计算 (Cross Entropy)",
            desc: "使用 `F.cross_entropy` 计算加权损失。Ignore Index（通常为 -100）处的 Loss 会被自动忽略。展平操作 `reshape(-1, vocab_size)` 确保了 batch 和 sequence 维度被统一处理。<span class='step-badge'>loss.py:60</span>",
            badge: "loss.py:60",
            state: "ce"
        },
        {
            title: "Step 3: 知识蒸馏 (Knowledge Distillation)",
            desc: "<code>KDLoss</code> 引入教师模型。计算公式为 $- \\sum P_{teacher} \\cdot \\log P_{student}$。通过 `inf_mask` 处理数值不稳定性，且只在有效标签位置（非 -100）计算 KL 散度。<span class='step-badge'>loss.py:107</span>",
            badge: "loss.py:107",
            state: "kd"
        },
        {
            title: "Step 4: DPO - 隐式奖励建模 (Implicit Reward)",
            desc: "DPO 不需要显式 Reward Model。它通过计算策略模型与参考模型在 `Chosen` 和 `Rejected` 回答上的 Log 概率差（Log Ratios），构造出隐式的 Logits 差值。<span class='step-badge'>loss.py:193</span>",
            badge: "loss.py:193",
            state: "dpo-logits"
        },
        {
            title: "Step 5: DPO - 偏好优化 (Preference Optimization)",
            desc: "最终 DPO 损失为 `-log(sigmoid(beta * logits))`。这意味着如果模型越偏好 `Chosen`（logits > 0），Loss 越小；反之 Loss 越大。Label Smoothing 可引入软标签以防止过拟合。<span class='step-badge'>loss.py:205</span>",
            badge: "loss.py:205",
            state: "dpo-loss"
        },
        {
            title: "Step 6: PPO - 价值函数裁剪 (Value Clipping)",
            desc: "为了防止 Value Network 更新过快，PPOLoss 计算两个 MSE：一个是直接预测值，一个是相对于旧 Value 裁剪后的预测值。取两者的最大值作为 Loss（悲观估计）。<span class='step-badge'>loss.py:275</span>",
            badge: "loss.py:275",
            state: "ppo-vf"
        },
        {
            title: "Step 7: PPO - 策略裁剪 (Actor Clipping)",
            desc: "核心机制：计算概率比率 $r_t(\\theta)$。如果 $r_t$ 超出 $[1-\\epsilon, 1+\\epsilon]$ 范围且 Advantage 为正（或负），则截断梯度。取 `min(surr1, surr2)` 形成著名的“信任区域”下界。<span class='step-badge'>loss.py:287</span>",
            badge: "loss.py:287",
            state: "ppo-ratio"
        },
        {
            title: "Step 8: GRPO - 群体相对优势 (Group Relative Policy)",
            desc: "GRPO 去掉了 Critic 模型，直接使用一组采样的平均 Log Ratio 作为基线。通过双边裁剪 `clip_eps_low/high` 和重要性采样权重，实现稳定的策略迭代。<span class='step-badge'>loss.py:363</span>",
            badge: "loss.py:363",
            state: "grpo-is"
        }
    ];

    let currentStep = 0;

    function render() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        if (currentStepSpan) currentStepSpan.innerText = currentStep + 1;
        if (stepTitle) stepTitle.innerText = step.title;
        if (stepDesc) stepDesc.innerHTML = step.desc;
        if (stepBadge) stepBadge.innerText = step.badge;

        container.innerHTML = '';
        
        try {
            if (step.state === 'shift') renderShift();
            else if (step.state === 'weighting') renderWeighting();
            else if (step.state === 'ce') renderCE();
            else if (step.state === 'kd') renderKD();
            else if (step.state === 'dpo-logits') renderDPONLogits();
            else if (step.state === 'dpo-loss') renderDPOLoss();
            else if (step.state === 'ppo-vf') renderPPOValue();
            else if (step.state === 'ppo-ratio') renderPPORatio();
            else if (step.state === 'grpo-is') renderGRPOIS();
        } catch(e) {
            console.error("Render failed", e);
        }

        prevBtn.disabled = currentStep === 0;
        nextBtn.innerText = currentStep === steps.length - 1 ? "完成" : "下一步";
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    function renderShift() {
        container.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:20px;">
                <div style="display:flex; gap:40px;">
                    <div class="tensor-block" style="border-color:#3182ce;">
                        <div class="tensor-label">Logits [B, L, V]</div>
                        <div style="display:flex; gap:5px; padding:10px;">
                            <div class="mini-box" style="background:#bee3f8;">P1</div>
                            <div class="mini-box" style="background:#bee3f8;">P2</div>
                            <div class="mini-box" style="background:#bee3f8;">P3</div>
                            <div class="mini-box" style="background:#cbd5e0; opacity:0.5;">_</div>
                        </div>
                        <div style="text-align:center; font-size:10px; color:#3182ce;">[:-1]</div>
                    </div>
                    <div class="tensor-block" style="border-color:#e53e3e;">
                        <div class="tensor-label">Labels [B, L]</div>
                        <div style="display:flex; gap:5px; padding:10px;">
                            <div class="mini-box" style="background:#cbd5e0; opacity:0.5;">_</div>
                            <div class="mini-box" style="background:#fed7d7;">L2</div>
                            <div class="mini-box" style="background:#fed7d7;">L3</div>
                            <div class="mini-box" style="background:#fed7d7;">L4</div>
                        </div>
                        <div style="text-align:center; font-size:10px; color:#e53e3e;">[1:]</div>
                    </div>
                </div>
                <div style="font-weight:bold; font-size:20px;">⬇️ Align ⬇️</div>
                <div style="display:flex; gap:10px; padding:15px; border:1px dashed #666; border-radius:10px; background:#f7fafc;">
                    <div style="display:flex; flex-direction:column; gap:5px;">
                        <div class="tensor-row" style="background:#bee3f8;">P1 (Predicts L2)</div>
                        <div class="tensor-row" style="background:#bee3f8;">P2 (Predicts L3)</div>
                        <div class="tensor-row" style="background:#bee3f8;">P3 (Predicts L4)</div>
                    </div>
                    <div style="font-size:20px; align-self:center;">≈</div>
                    <div style="display:flex; flex-direction:column; gap:5px;">
                        <div class="tensor-row" style="background:#fed7d7;">L2</div>
                        <div class="tensor-row" style="background:#fed7d7;">L3</div>
                        <div class="tensor-row" style="background:#fed7d7;">L4</div>
                    </div>
                </div>
            </div>
        `;
    }

    function renderWeighting() {
        container.innerHTML = `
            <div style="display:flex; justify-content:center; gap:10px;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="mini-box" style="width:50px; height:50px;">ID:0</div>
                    <div>w=1.0</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="mini-box" style="width:50px; height:50px;">ID:1</div>
                    <div>w=1.0</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="mini-box" style="width:50px; height:50px; background:#fbd38d; border:2px solid #ed8936;">EOS</div>
                    <div style="color:#ed8936; font-weight:bold;">w=2.0</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="mini-box" style="width:50px; height:50px;">ID:3</div>
                    <div>w=1.0</div>
                </div>
            </div>
            <div style="margin-top:20px; text-align:center; color:#666;">High weight on EOS forces model to learn termination condition.</div>
        `;
    }

    function renderCE() {
        container.innerHTML = `
            <div class="ce-calc">
                <div>Target Class: <strong>2 (EOS)</strong></div>
                <div style="font-size:24px;">⬇️</div>
                <div>Log Softmax: [ -2.3, -4.5, <strong style="color:#2ecc71">-0.1</strong>, -3.2 ]</div>
                <div style="font-size:24px;">⬇️</div>
                <div>Negative Log Likelihood: <strong>0.1</strong></div>
                <div style="font-size:24px;">⬇️</div>
                <div style="background:#fffaf0; padding:10px; border:1px solid #ed8936; border-radius:5px;">
                    Weighted Loss = 0.1 * <span style="color:#ed8936; font-weight:bold;">2.0</span> = 0.2
                </div>
            </div>
        `;
    }

    function renderKD() {
        container.innerHTML = `
            <div class="kd-viz">
                <div style="display:flex; align-items:center; gap:10px; width:100%;">
                    <div style="width:80px; text-align:right;">Teacher</div>
                    <div style="flex:1; height:30px; background:linear-gradient(to right, #4299e1 80%, #eee 20%); border-radius:4px;"></div>
                </div>
                <div style="display:flex; align-items:center; gap:10px; width:100%;">
                    <div style="width:80px; text-align:right;">Student</div>
                    <div style="flex:1; height:30px; background:linear-gradient(to right, #ed8936 60%, #eee 40%); border-radius:4px;"></div>
                </div>
                <div style="margin-top:20px; font-weight:bold;">KL Divergence Minimized</div>
            </div>
        `;
    }

    function renderDPONLogits() {
        container.innerHTML = `
            <div style="display:grid; grid-template-columns: 1fr auto 1fr; gap:10px; align-items:center;">
                <div class="tensor-block">
                    <div>Policy Model</div>
                    <div style="color:#2ecc71;">log π(w) = -1.2</div>
                    <div style="color:#e53e3e;">log π(l) = -2.5</div>
                    <hr>
                    <div>Δ = 1.3</div>
                </div>
                <div style="font-size:30px;">-</div>
                <div class="tensor-block" style="opacity:0.7;">
                    <div>Ref Model</div>
                    <div style="color:#2ecc71;">log ref(w) = -1.5</div>
                    <div style="color:#e53e3e;">log ref(l) = -2.0</div>
                    <hr>
                    <div>Δ = 0.5</div>
                </div>
            </div>
            <div style="text-align:center; margin-top:20px; font-weight:bold; color:#2b6cb0;">
                Final DPO Logit = 1.3 - 0.5 = 0.8
            </div>
        `;
    }

    function renderDPOLoss() {
        container.innerHTML = `
            <div class="ce-calc" style="width:100%;">
                <div style="display:flex; justify-content:space-between; width:80%; margin:0 auto;">
                    <div style="color:#2ecc71; font-weight:bold;">Winner (Chosen)</div>
                    <div style="color:#e53e3e; font-weight:bold;">Loser (Rejected)</div>
                </div>
                <div style="position:relative; width:80%; height:40px; background:#eee; border-radius:20px; margin:10px auto;">
                    <div id="dpo-knob" style="position:absolute; left:50%; top:0; width:40px; height:40px; background:#2b6cb0; border-radius:50%; transition:all 1s;"></div>
                </div>
                <div style="text-align:center; margin-top:10px;">Pushing logits apart...</div>
            </div>
        `;
        setTimeout(() => {
            const knob = document.getElementById('dpo-knob');
            if(knob) knob.style.left = "20%";
        }, 500);
    }

    function renderPPOValue() {
        container.innerHTML = `
            <div class="tensor-flow">
                <div class="tensor-row" style="justify-content:space-between;">
                    <span>New Value</span>
                    <span>Old Value</span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <div class="mini-box" style="width:100px;">1.5</div>
                    <div style="font-size:20px;">-</div>
                    <div class="mini-box" style="width:100px;">1.0</div>
                </div>
                <div style="margin:10px 0;">Diff = 0.5</div>
                <div class="tensor-row" style="background:#fff5f5; border-color:#e53e3e;">
                    Clipped Diff = clamp(0.5, -0.2, 0.2) = 0.2
                </div>
            </div>
        `;
    }

    function renderPPORatio() {
        container.innerHTML = `
            <div class="ppo-viz" style="text-align:center;">
                <div style="font-size:18px; margin-bottom:10px;">Ratio $r_t(\\theta)$</div>
                <div style="width:100%; height:100px; background:#f7fafc; position:relative; border-bottom:2px solid #ccc;">
                    <div style="position:absolute; bottom:0; left:40%; width:20%; height:100%; background:rgba(46, 204, 113, 0.2); border-left:2px dashed #27ae60; border-right:2px dashed #27ae60;"></div>
                    <div style="position:absolute; top:40%; left:50%; transform:translate(-50%, -50%); color:#27ae60;">Trust Region</div>
                    <div id="ratio-dot" style="position:absolute; bottom:10px; left:50%; width:15px; height:15px; background:#e74c3c; border-radius:50%; transition:all 1s;"></div>
                </div>
                <div style="margin-top:10px; font-size:12px; color:#666;">If dot moves outside green zone, gradient is clipped.</div>
            </div>
        `;
        setTimeout(() => {
            const dot = document.getElementById('ratio-dot');
            if(dot) dot.style.left = "80%";
        }, 500);
    }

    function renderGRPOIS() {
        container.innerHTML = `
            <div class="kd-viz">
                <div style="display:flex; gap:20px; align-items:center;">
                    <div class="tensor-block" style="width:120px;">
                        <div>Old Policy</div>
                        <div style="font-size:20px;">📜</div>
                    </div>
                    <div style="font-size:20px;">vs</div>
                    <div class="tensor-block" style="width:120px;">
                        <div>New Policy</div>
                        <div style="font-size:20px;">📜</div>
                    </div>
                </div>
                <div style="margin-top:20px;">Importance Weight = New / Old</div>
                <div style="font-size:12px; color:#666;">Used to re-weight advantages without a critic model.</div>
            </div>
        `;
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

    nextBtn.addEventListener('click', goNext);
    prevBtn.addEventListener('click', goPrev);
    
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentStep = 0;
            render();
        });
    }

    render();

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });
});
