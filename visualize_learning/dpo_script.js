document.addEventListener('DOMContentLoaded', () => {
  const prevBtn = document.getElementById('prevStep');
  const nextBtn = document.getElementById('nextStep');
  const resetBtn = document.getElementById('reset');
  const infoBox = document.getElementById('infoBox');
  const visualContent = document.getElementById('visualContent');
  const codeSnippet = document.getElementById('codeSnippet');

  if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet) {
    console.error("Required elements not found in DPO script");
    return;
  }

  let currentStep = 0;
  let currentTimeout = null; // Track timeout for cleanup

  // DPO 训练流程：从偏好数据到 DPO/IPO 损失
  const steps = [
    {
      title: 'Phase 1: 偏好数据对 (Preference Pairs)',
      description:
        'DPO 不需要外部奖励模型。它的输入是<b>偏好对</b>：对于同一个 Prompt，包含一个“更好的回答” (Chosen) 和一个“较差的回答” (Rejected)。<br><span class="step-badge">dataset.py:DPODataset</span>',
      code: `batch = { "prompt": "...", "chosen": "...", "rejected": "..." }`,
      render: () => renderDataPair(),
    },
    {
      title: 'Phase 2: 并行概率计算 (LogProbs Sum)',
      description:
        '为了提高效率，Cortex 将 Chosen 和 Rejected 拼接成一个大 Batch 送入模型。通过 <code>log_softmax</code> 取出每个生成 Token 在目标位置上的概率，并对序列求和（忽略 Padding）。<br><span class="step-badge">dpo_trainer.py:255, 186</span>',
      code: 'concat_inputs = torch.concat([chosen, rejected], dim=0)\nlogprobs = log_softmax(logits, labels).sum(-1)',
      render: () => renderLogProbCalc(),
    },
    {
      title: 'Phase 3: 隐含奖励差值 (Implicit Reward)',
      description:
        '计算模型对 Chosen 的偏好相对于 Ref 模型的提升程度。公式：<code>pi_logratios = pi_chosen - pi_reject</code>，<code>ref_logratios = ref_chosen - ref_reject</code>，最终 <code>logits = pi_logratios - ref_logratios</code>。<br><span class="step-badge">loss.py:193-198</span>',
      code: 'pi_logratios = policy_chosen_logps - policy_reject_logps\nlogits = pi_logratios - ref_logratios',
      render: () => renderImplicitReward(),
    },
    {
      title: 'Phase 4: DPO vs IPO 目标函数',
      description:
        '<b>DPO:</b> 使用 LogSigmoid，类似于分类器，通过拉大 Chosen 和 Rejected 的差距来优化。<br><b>IPO:</b> (Implicit Preference Optimization) 增加了一个二次项 <code>(logits - 1/(2*beta))**2</code>，对 Logits 大小进行正则，防止模型崩塌。<br><span class="step-badge">loss.py:201-210</span>',
      code: 'if self.ipo:\n    loss = (logits - 1/(2*self.beta)) ** 2\nelse:\n    loss = -F.logsigmoid(self.beta * logits) * (1 - smooth)',
      render: () => renderLossComparison(),
    },
    {
      title: 'Phase 5: NLL 辅助损失 (可选)',
      description:
        '在 DPO 优化时，有时会加入负对数似然 (NLL) 损失来保持模型在 Chosen 数据上的语言建模能力，防止由于偏好对齐导致模型“变笨”。<br><span class="step-badge">dpo_trainer.py:276</span>',
      code: 'nll_loss = -policy_logprobs_means[chosen].mean()\ntotal_loss = dpo_loss + nll_coef * nll_loss',
      render: () => renderNLLAux(),
    },
    {
      title: 'Phase 6: 梯度方向 (Direct Update)',
      description:
        '梯度会自动推着 Chosen 的 LogProb 上升，Rejected 的 LogProb 下降。由于不需要采样 (Rollout)，DPO 的显存开销通常比 PPO 小得多。',
      code: 'loss.backward()\noptimizer.step()',
      render: () => renderGradientFlow(),
    },
  ];

  function updateUI() {
    // Clear any pending timeouts from previous steps
    if (currentTimeout) {
      clearTimeout(currentTimeout);
      currentTimeout = null;
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
    
    // Guard against render errors
    try {
      step.render();
    } catch (e) {
      console.error("Render failed", e);
    }
    
    updateButtons();
  }

  function renderDataPair() {
    visualContent.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:10px; width:80%">
        <div class="box" style="background:#eee; border-left: 4px solid #95a5a6;"><b>Prompt:</b> "Explain Quantum Physics"</div>
        <div style="display:flex; gap:10px">
          <div class="box" style="border-color:#2ecc71; flex:1; background:#e8f8f5;"><b>Chosen (👍):</b> "It is about very small particles..."</div>
          <div class="box" style="border-color:#e74c3c; flex:1; background:#fdedec;"><b>Rejected (👎):</b> "It is magic."</div>
        </div>
      </div>
    `;
  }

  function renderLogProbCalc() {
      visualContent.innerHTML = `
          <div style="display:flex; flex-direction:column; align-items:center; gap:20px; width:100%;">
              <div style="display:flex; border:2px solid #34495e; border-radius:8px; overflow:hidden;">
                  <div style="padding:10px; background:#d5f5e3; width:120px; text-align:center;">Chosen</div>
                  <div style="padding:10px; background:#fadbd8; width:120px; text-align:center;">Rejected</div>
              </div>
              <div class="arrow">⬇ Concat & Forward ⬇</div>
              <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                  <div class="box" style="font-size:12px; border-color:#3498db; background:#ebf5fb;">
                      <b>Policy Model</b><br>
                      SUM(logP_chosen)<br>
                      SUM(logP_rejected)
                  </div>
                  <div class="box" style="font-size:12px; border-color:#95a5a6; background:#f4f6f7; opacity:0.8;">
                      <b>Ref Model (Frozen)</b><br>
                      SUM(logP_chosen)<br>
                      SUM(logP_rejected)
                  </div>
              </div>
          </div>
      `;
  }

  function renderImplicitReward() {
      visualContent.innerHTML = `
          <div style="text-align:center; width: 100%;">
              <div style="font-size:1.5em; margin-bottom:20px; font-family:serif;">
                  <span style="color:#27ae60">π<sub>θ</sub>(y<sub>w</sub>|x)</span> / <span style="color:#7f8c8d">π<sub>ref</sub>(y<sub>w</sub>|x)</span>
                  &nbsp; vs &nbsp;
                  <span style="color:#c0392b">π<sub>θ</sub>(y<sub>l</sub>|x)</span> / <span style="color:#7f8c8d">π<sub>ref</sub>(y<sub>l</sub>|x)</span>
              </div>
              
              <div style="display:flex; justify-content:center; align-items:center; gap:10px; margin-top:20px;">
                  <div class="box" style="background:#e8f8f5; border-color:#27ae60;">
                      Log Ratio (Chosen)<br>
                      <span style="font-size:1.2em;">+2.5</span>
                  </div>
                  <div style="font-size:20px; color:#aaa;">-</div>
                  <div class="box" style="background:#fdedec; border-color:#c0392b;">
                      Log Ratio (Rejected)<br>
                      <span style="font-size:1.2em;">-1.2</span>
                  </div>
                  <div style="font-size:20px;">=</div>
                  <div class="box active" style="background:#fff; border-color:#f1c40f;">
                      Implicit Reward<br>
                      <span style="font-size:1.2em;">+3.7</span>
                  </div>
              </div>
              <div class="label-text" style="margin-top:20px; color:#666;">模型不仅要选对，还要比 Ref 模型更自信</div>
          </div>
      `;
  }

  function renderLossComparison() {
      visualContent.innerHTML = `
          <div style="display:flex; gap:30px; align-items:flex-end; height:150px;">
              <div style="text-align:center;">
                  <div style="height:100px; width:80px; background:#3498db; margin:0 auto; display:flex; align-items:center; justify-content:center; color:white; border-radius:4px;">LogSigmoid</div>
                  <div style="font-size:12px; margin-top:5px; font-weight:bold;">DPO</div>
                  <div style="font-size:10px; color:#666;">-log(σ(r))</div>
              </div>
              <div style="text-align:center;">
                  <div style="height:120px; width:80px; background:#9b59b6; margin:0 auto; display:flex; align-items:center; justify-content:center; color:white; border-radius:4px;">Quadratic</div>
                  <div style="font-size:12px; margin-top:5px; font-weight:bold;">IPO</div>
                  <div style="font-size:10px; color:#666;">(r - 1/2β)²</div>
              </div>
          </div>
          <div class="label-text" style="margin-top:20px;">IPO 对 Logits 差值施加了更强的约束，防止过拟合</div>
      `;
  }

  function renderNLLAux() {
      visualContent.innerHTML = `
          <div class="math-flow">
              <div class="row" style="display:flex; gap:10px; align-items:center;">
                  <div class="box" style="background:#e67e22; color:white; border:none;">DPO Loss (Relative)</div>
                  <div style="font-size:20px;">+</div>
                  <div class="box" style="background:#2c3e50; color:white; border:none;">
                      NLL Loss (Chosen)<br>
                      <span style="font-size:0.8em; opacity:0.8;">-log P(chosen)</span>
                  </div>
              </div>
              <div class="label-text" style="margin-top:10px;">同时优化“偏好”与“生成能力”，防止模型为了迎合偏好而丧失语言通顺性</div>
          </div>
      `;
  }

  function renderGradientFlow() {
      visualContent.innerHTML = `
          <div style="position:relative; width:100%; height:120px; display:flex; justify-content:center; align-items:center; background:#fcfcfc; border-radius:8px;">
              <div id="g-chosen" style="position:absolute; left:20%; transition:all 1s; text-align:center;">
                  <div style="font-size:24px;">👍</div>
                  Chosen LogProb
              </div>
              <div id="g-rejected" style="position:absolute; right:20%; transition:all 1s; text-align:center;">
                  <div style="font-size:24px;">👎</div>
                  Rejected LogProb
              </div>
              <div style="width:60%; height:2px; background:#eee;"></div>
          </div>
      `;
      
      // Auto-trigger animation safely
      currentTimeout = setTimeout(() => {
          const c = document.getElementById('g-chosen');
          const r = document.getElementById('g-rejected');
          if(c && r) {
              c.style.transform = "translateY(-30px)";
              c.style.color = "#27ae60";
              c.innerHTML += "<div style='font-size:12px; font-weight:bold;'>UP</div>";
              
              r.style.transform = "translateY(30px)";
              r.style.color = "#c0392b";
              r.innerHTML += "<div style='font-size:12px; font-weight:bold;'>DOWN</div>";
          }
      }, 500);
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

  updateUI();
});
