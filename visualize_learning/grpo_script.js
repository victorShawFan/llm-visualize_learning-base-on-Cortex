document.addEventListener('DOMContentLoaded', () => {
  const prevBtn = document.getElementById('prevStep');
  const nextBtn = document.getElementById('nextStep');
  const resetBtn = document.getElementById('reset');
  const infoBox = document.getElementById('infoBox');
  const visualContent = document.getElementById('visualContent');
  const codeSnippet = document.getElementById('codeSnippet');

  if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
    console.error("Required elements not found in GRPO script");
    return;
  }

  let currentStep = 0;
  let currentTimer = null; // Can hold either timeout or interval ID
  let timerType = null; // 'timeout' or 'interval'

  const steps = [
    {
      title: '步骤 0: GRPO 核心理念 (No Critic)',
      description:
        'GRPO (Group Relative Policy Optimization) 是 DeepSeek-R1 采用的算法。它最大的创新是<b>取消了独立的 Value (Critic) 网络</b>，转而通过一组回答的相对得分来估计优势（Advantage）。<br><span class="step-badge">grpo_trainer.py:88</span>',
      code: 'if self.train_config.grpo_config.loss_beta == 0.0:\n    return None # 不需要 ref_model (可选)',
      render: () => renderConfig(),
    },
    {
      title: '步骤 1: 群体采样 (Group Sampling)',
      description:
        '对每个 Prompt，利用 <code>repeat_interleave</code> 复制 <code>G</code> 份，并行生成 <code>G</code> 个不同的回答。这一步通过 <code>batch_generate</code> 高效完成。<br><span class="step-badge">grpo_trainer.py:274, 282</span>',
      code: 'prompt_ids = prompt_ids.repeat_interleave(group_size, 0)\n# 一次性生成 G 个 completion',
      render: () => renderGroupSampling(),
    },
    {
      title: '步骤 2: 多样化奖励评分 (Multi-Reward)',
      description:
        '将生成的回答送入 <code>reward_func</code>。奖励可以来自模型打分、代码执行结果（Compiler）或数学逻辑校验。GRPO 能整合多种异构奖励。<br><span class="step-badge">grpo_trainer.py:384</span>',
      code: 'rewards = torch.tensor(self.reward_func(prompts, completions, answers))',
      render: () => renderMultiRewards(),
    },
    {
      title: '步骤 3: 组内均值与标准差 (Group Stats)',
      description:
        '在 Batch 维度上将奖励 <code>view</code> 为 <code>(N, G)</code>，计算每组回答的均值 $μ$ 和标准差 $σ$。这是计算相对优势的基础。<br><span class="step-badge">grpo_trainer.py:230-235</span>',
      code: 'group_means = rewards.view(-1, G).mean(dim=1)\ngroup_stds = rewards.view(-1, G).std(dim=1)',
      render: () => renderStatsDetail(),
    },
    {
      title: '步骤 4: 相对优势归一化 (Relative Advantage)',
      description:
        '计算归一化优势：<code>advantages = (rewards - mean) / (std + 1e-4)</code>。表现优于同组平均水平的回答获得正向激励，反之则受罚。这实现了“组内自我进化”。<br><span class="step-badge">grpo_trainer.py:244</span>',
      code: 'advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)',
      render: () => renderAdvantageAnimation(),
    },
    {
      title: '步骤 5: 截断目标与 KL 惩罚',
      description:
        '损失函数结合了 PPO 的比率裁剪。计算重要性权重 <code>coef = exp(log_ratio)</code>，并对其裁剪到 <code>[1-eps, 1+eps]</code>。最终 Loss 取未裁剪与裁剪项的较小值（PPO-style conservative bound）。<br><span class="step-badge">loss.py:372-381</span>',
      code: 'coef_1 = torch.exp(log_ratio)\ncoef_2 = torch.clamp(coef_1, 1-eps, 1+eps)\nper_token_loss = -torch.min(coef_1 * A, coef_2 * A)',
      render: () => renderObjectiveFlow(),
    },
    {
      title: '步骤 6: 同批次多步更新 (Multi-step Update)',
      description:
        '与 SFT 不同，GRPO 会对同一批 Rollout 数据执行多次（<code>grpo_steps</code>）梯度更新。这能更充分地利用昂贵的采样数据。<br><span class="step-badge">grpo_trainer.py:508</span>',
      code: 'for grpo_step in range(config.grpo_steps):\n    loss = maximize_grpo_objective(rollout_data)\n    loss.backward(); optimizer.step()',
      render: () => renderUpdateLoop(),
    },
  ];

  function updateUI() {
    // Clear existing timer
    if (currentTimer) {
      if (timerType === 'interval') clearInterval(currentTimer);
      else clearTimeout(currentTimer);
      currentTimer = null;
      timerType = null;
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
    
    try {
        step.render();
    } catch(e) {
        console.error("Render failed", e);
    }

    updateButtons();
  }

  function renderConfig() {
    visualContent.innerHTML = `
      <div style="display:flex; justify-content:center; gap:20px;">
        <div class="model-box policy">Policy (π)</div>
        <div class="model-box ref">Ref (π_ref)</div>
      </div>
      <div style="margin-top:20px; display:flex; justify-content:center; gap:20px;">
        <div class="dict-entry" style="width:120px;">group_size = 4</div>
        <div class="dict-entry" style="width:120px;">gen_max_new_tokens = 64</div>
      </div>
    `;
  }

  function renderGroupSampling() {
      visualContent.innerHTML = `
          <div style="display:flex; flex-direction:column; align-items:center; gap:20px;">
              <div class="box" style="background:#fdf2e9; border:2px solid #e67e22; width:200px;">Prompt ID [1, L]</div>
              <div class="arrow-down"><code>.repeat_interleave(G=4)</code> ↓</div>
              <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;">
                  <div class="box" style="background:#fdf2e9; opacity:0.8;">P1</div>
                  <div class="box" style="background:#fdf2e9; opacity:0.8;">P1</div>
                  <div class="box" style="background:#fdf2e9; opacity:0.8;">P1</div>
                  <div class="box" style="background:#fdf2e9; opacity:0.8;">P1</div>
              </div>
              <div class="arrow-down">↓ <code>batch_generate</code> ↓</div>
              <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;">
                  <div class="comp-card" id="c1">C1 (Correct)</div>
                  <div class="comp-card" id="c2">C2 (Partial)</div>
                  <div class="comp-card" id="c3">C3 (Wrong)</div>
                  <div class="comp-card" id="c4">C4 (Slow)</div>
              </div>
          </div>
      `;
  }

  function renderMultiRewards() {
      visualContent.innerHTML = `
          <div style="display:flex; flex-direction:column; align-items:center; gap:15px; width:100%;">
              <div style="display:flex; gap:10px; width:100%; justify-content:center;">
                  <div style="text-align:center;">
                      <div class="comp-card" style="height:40px; font-size:10px;">C1</div>
                      <div class="arrow-down">↓</div>
                      <div class="box" style="border-color:#27ae60;">+1.0</div>
                  </div>
                  <div style="text-align:center;">
                      <div class="comp-card" style="height:40px; font-size:10px;">C2</div>
                      <div class="arrow-down">↓</div>
                      <div class="box" style="border-color:#f1c40f;">+0.5</div>
                  </div>
                  <div style="text-align:center;">
                      <div class="comp-card" style="height:40px; font-size:10px;">C3</div>
                      <div class="arrow-down">↓</div>
                      <div class="box" style="border-color:#e74c3c;">-1.0</div>
                  </div>
                  <div style="text-align:center;">
                      <div class="comp-card" style="height:40px; font-size:10px;">C4</div>
                      <div class="arrow-down">↓</div>
                      <div class="box" style="border-color:#95a5a6;">0.0</div>
                  </div>
              </div>
              <div style="background:#f8f9fa; padding:10px; border-radius:8px; font-size:0.8em; color:#666;">
                  奖励函数支持：准确性、格式（XML/JSON）、代码运行、反思令牌计数等。
              </div>
          </div>
      `;
  }

  function renderStatsDetail() {
      visualContent.innerHTML = `
          <div class="stats-box" style="width:100%;">
              <div style="display:flex; justify-content:around; margin-bottom:20px;">
                  <div class="stat-circle">μ = 0.125</div>
                  <div class="stat-circle">σ = 0.74</div>
              </div>
              <div class="baseline-viz" style="height:60px; position:relative; background:#eee; border-radius:30px;">
                  <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:#34495e;"></div>
                  <div class="point" style="left:90%; background:#27ae60;"></div>
                  <div class="point" style="left:70%; background:#f1c40f;"></div>
                  <div class="point" style="left:10%; background:#e74c3c;"></div>
                  <div class="point" style="left:45%; background:#95a5a6;"></div>
                  <div style="position:absolute; left:50%; top:-25px; font-size:10px;">Mean (μ)</div>
              </div>
          </div>
      `;
  }

  function renderAdvantageAnimation() {
      visualContent.innerHTML = `
          <div style="display:flex; gap:10px;">
              <div class="adv-card" id="a1">A1: +1.18</div>
              <div class="adv-card" id="a2">A2: +0.51</div>
              <div class="adv-card" id="a3">A3: -1.52</div>
              <div class="adv-card" id="a4">A4: -0.17</div>
          </div>
          <div style="margin-top:20px; font-size:0.9em; color:#2c3e50; font-weight:bold;">
              Sum(Advantages) ≈ 0.0
          </div>
      `;
      // Add colors safely with timeout
      currentTimer = setTimeout(() => {
          const el1 = document.getElementById('a1');
          const el2 = document.getElementById('a2');
          const el3 = document.getElementById('a3');
          const el4 = document.getElementById('a4');
          if (el1) el1.style.background = "#d5f5e3";
          if (el2) el2.style.background = "#fcf3cf";
          if (el3) el3.style.background = "#fadbd8";
          if (el4) el4.style.background = "#f4f6f7";
          currentTimer = null; // Clear ref
      }, 500);
      timerType = 'timeout';
  }

  function renderObjectiveFlow() {
      visualContent.innerHTML = `
          <div style="text-align:center;">
              <div style="font-family:serif; font-style:italic; font-size:1.2em; margin-bottom:20px;">
                  Loss = -Adv * min(ratio, clip) + β * KL
              </div>
              <div style="display:flex; justify-content:center; gap:30px;">
                  <div class="box" style="background:#e8f8f5;">概率比率 (Policy/Old)</div>
                  <div class="box" style="background:#fef9e7;">优势 (Advantage)</div>
                  <div class="box" style="background:#f4ecf7;">散度 (KL penalty)</div>
              </div>
          </div>
      `;
  }

  function renderUpdateLoop() {
      visualContent.innerHTML = `
          <div style="display:flex; flex-direction:column; align-items:center; gap:10px;">
              <div id="loop-box" style="padding:20px; border:3px solid #3498db; border-radius:50%; width:120px; height:120px; display:grid; place-items:center; text-align:center; transition:all 0.5s;">
                  Step 1/N
              </div>
              <div class="label-text">反复优化同一批经验</div>
          </div>
      `;
      let i = 1;
      currentTimer = setInterval(() => {
          const lb = document.getElementById('loop-box');
          if(lb) {
              lb.innerText = `Step ${i}/3`;
              lb.style.borderColor = i % 2 ? '#3498db' : '#e67e22';
              i = (i % 3) + 1;
          } else {
              // Safety clean if somehow element is gone but interval runs
              clearInterval(currentTimer);
              currentTimer = null;
          }
      }, 1500);
      timerType = 'interval';
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

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') goPrev();
    if (e.key === 'ArrowRight') goNext();
  });

  updateUI();
});
