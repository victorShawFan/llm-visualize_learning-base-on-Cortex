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

    // Simulation Data
    const promptText = "Question: 2+2=?";
    const completions = [
        { text: "It is 4.", score: 1.0 },
        { text: "Maybe 5.", score: 0.0 },
        { text: "Answer: 4", score: 1.0 },
        { text: "I don't know", score: 0.1 }
    ];
    // Calculate Mean/Std for simulation
    const scores = completions.map(c => c.score);
    const mean = scores.reduce((a,b)=>a+b,0)/4; // 0.525
    const std = Math.sqrt(scores.map(x => Math.pow(x-mean, 2)).reduce((a,b)=>a+b,0)/4) + 1e-6; // approx 0.47
    
    // Add advantages
    completions.forEach(c => {
        c.adv = ((c.score - mean) / std).toFixed(2);
    });

    function renderTree(state) {
        // State: 'prompt', 'group', 'rewards', 'stats', 'advantages', 'loss'
        let html = `<div class="tree-container">`;
        
        // No Critic Badge
        html += `<div class="critic-crossed">Critic Model</div>`;

        // Prompt Node
        html += `<div class="node prompt anim-pop-in" style="animation-delay:0s">${promptText}</div>`;

        // SVG Lines (simplified via CSS/HTML lines)
        if (state !== 'prompt') {
            html += `<div class="lines-svg" style="position:absolute; top:40px; width:100%; height:40px; opacity:0; animation:slideUp 0.5s 0.2s forwards;">
                <!-- Drawn via CSS borders usually, but here implied by structure -->
                <div style="width:2px; height:20px; background:#bdc3c7; margin:0 auto;"></div>
                <div style="width:360px; height:2px; background:#bdc3c7; margin:0 auto;"></div>
            </div>`;
        }

        // Branch Row
        if (state !== 'prompt') {
            html += `<div class="branch-row" style="margin-top:20px;">`;
            completions.forEach((c, idx) => {
                const showScore = ['rewards', 'stats', 'advantages', 'loss'].includes(state);
                const showAdv = ['advantages', 'loss'].includes(state);
                const isBad = parseFloat(c.adv) < 0;
                
                let borderStyle = '';
                if (state === 'loss') {
                    borderStyle = isBad ? 'border-color:#e74c3c; background:#fadbd8' : 'border-color:#2ecc71; background:#d5f5e3';
                }

                html += `
                <div class="node completion anim-pop-in" style="animation-delay:${0.3 + idx*0.1}s; ${borderStyle}">
                    <div style="font-style:italic">"${c.text}"</div>
                    ${showScore ? `<div class="score-badge visible" style="transition-delay:${1 + idx*0.1}s">R: ${c.score}</div>` : ''}
                    ${showAdv ? `<div class="adv-badge visible" style="background:${isBad?'#c0392b':'#27ae60'}; animation-delay:${1.5 + idx*0.1}s">Adv: ${c.adv}</div>` : ''}
                </div>`;
            });
            html += `</div>`;
        }

        // Stats Overlay
        if (['stats', 'advantages', 'loss'].includes(state)) {
            html += `<div class="stats-overlay visible" style="animation-delay:0.8s">
                <div>Mean (μ): ${mean.toFixed(3)}</div>
                <div>Std (σ): ${std.toFixed(3)}</div>
            </div>`;
        }

        html += `</div>`; // End container
        return html;
    }

    const steps = [
        {
            title: "Phase 1: Prompt Input",
            desc: "GRPO 不需要 Value Model (Critic)。一切从一个 Prompt 开始。我们将其复制 G 份（Group Size）。",
            code: "prompts = batch['prompt'] # Batch Size 1 for demo",
            render: () => renderTree('prompt')
        },
        {
            title: "Phase 2: Group Sampling",
            desc: "模型对同一个 Prompt 生成 G=4 个不同的回复。这里利用了采样的随机性（Temperature > 0）。",
            code: "outputs = model.generate(prompts, num_return_sequences=4)",
            render: () => renderTree('group')
        },
        {
            title: "Phase 3: Reward Scoring",
            desc: "Reward Model 对这 4 个回复分别打分。注意：这里有一些回复是错的（0分），有些是对的（1分）。",
            code: "rewards = reward_model(outputs) # [1.0, 0.0, 1.0, 0.1]",
            render: () => renderTree('rewards')
        },
        {
            title: "Phase 4: Group Statistics",
            desc: "计算这组回复的平均分 (Mean) 和标准差 (Std)。这是 GRPO 的核心：我们不与全局 Critic 比较，而是自己和自己组内的“平均水平”比较。",
            code: "mean = rewards.mean()\nstd = rewards.std()",
            render: () => renderTree('stats')
        },
        {
            title: "Phase 5: Advantage Calculation (Comparison)",
            desc: "现在我们来“论功行赏”。<br>基准线（平均分）是 <b>0.525</b>。<br>1. 'It is 4' 得了 1.0 分 -> <b>高于</b>平均 -> 优势为正 (+1.01)<br>2. 'Maybe 5' 得了 0.0 分 -> <b>低于</b>平均 -> 优势为负 (-1.11)<br>这实际上是在做组内排序 (Ranking)。",
            code: "# Formula: (Score - Mean) / Std\n# Case 1: (1.0 - 0.525) / 0.47 = +1.01 (Good!)\n# Case 2: (0.0 - 0.525) / 0.47 = -1.11 (Bad!)",
            render: () => renderTree('advantages')
        },
        {
            title: "Phase 6: Loss Calculation (Encourage/Suppress)",
            desc: "我们的目标是最小化 Loss。公式为：<b>Loss = -Advantage × ln(Probability)</b>。<br>1. <b>Case 1 (Adv=+1.01)</b>：Loss = -1.01 × ln(P)。为了最小化 Loss，ln(P) 必须尽可能大 -> <b>提高概率 (Encourage)</b>。<br>2. <b>Case 2 (Adv=-1.11)</b>：Loss = +1.11 × ln(P)。为了最小化 Loss，ln(P) 必须尽可能小（负无穷）-> <b>降低概率 (Suppress)</b>。",
            code: "loss = -advantage * torch.log(prob)\n\n# Case 1 (Good): -1.01 * ln(P) -> Maximize P to minimize Loss\n# Case 2 (Bad): +1.11 * ln(P) -> Minimize P to minimize Loss",
            render: () => renderTree('loss')
        },
        {
            title: "Why No Critic?",
            desc: "因为我们使用组内平均值 (Group Mean) 作为 Baseline，而不是 Critic 预测的 Value。这节省了一个巨大的 Critic 模型，大幅降低了显存开销。",
            code: "# No Critic Model needed!\n# Memory usage reduced by ~40%",
            render: () => `<div class="tree-container">
                <div class="node prompt active" style="transform:scale(1.2)">Efficiency 🚀</div>
                <div style="margin-top:20px; text-align:center;">
                    <p>Mean(Group) ≈ Value(State)</p>
                    <p style="color:#7f8c8d; font-size:12px;">The group average serves as a dynamic baseline.</p>
                </div>
            </div>`
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
