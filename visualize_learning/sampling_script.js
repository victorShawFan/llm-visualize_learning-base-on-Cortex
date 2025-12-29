document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const chart = document.getElementById('chart');
    const stats = document.getElementById('stats');
    const tempInput = document.getElementById('tempInput');
    const topKInput = document.getElementById('topKInput');
    const topPInput = document.getElementById('topPInput');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !codeSnippet || !chart || !stats) {
        console.error("Required elements not found in Sampling script");
        return;
    }

    let currentStep = 0;
    let currentInterval = null;

    let initialLogits = [
        { tok: "The", l: 4.5, p: 0.0 }, { tok: "Cat", l: 4.2, p: 0.0 }, 
        { tok: "Dog", l: 3.8, p: 0.0 }, { tok: "A", l: 3.1, p: 0.0 }, 
        { tok: "Sky", l: 2.5, p: 0.0 }, { tok: "Is", l: 1.8, p: 0.0 },
        { tok: "Run", l: 1.5, p: 0.0 }, { tok: "Eat", l: 0.5, p: 0.0 }
    ];
    let workingLogits = [];

    const steps = [
        {
            title: "Step 0: 原始 Logits (Model Output)",
            description: "模型对词表中的每个候选词输出原始得分。值越高，模型认为该词出现的可能性越大。目前处于未归一化状态。<span class='step-badge'>generate_utils.py:268</span>",
            code: "logits = model(input_ids).logits[:, -1, :]",
            action: () => resetData()
        },
        {
            title: "Step 1: 温度缩放 (Temperature)",
            description: "调整分布的“锐利”度。<b>T < 1.0</b> 放大高分优势（保守）；<b>T > 1.0</b> 缩小分差（多样）。<span class='step-badge'>generate_utils.py:59</span>",
            code: "logits = logits / temperature",
            action: () => applyTemperature()
        },
        {
            title: "Step 2: Top-K 截断 (Hard Truncation)",
            description: "只保留前 K 个最高分词。其余所有词的 Logits 被设为 <code>-inf</code>（对应概率为 0）。<span class='step-badge'>generate_utils.py:91</span>",
            code: "indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]\nlogits = logits.masked_fill(indices_to_remove, -float('Inf'))",
            action: () => applyTopK()
        },
        {
            title: "Step 3: Softmax 归一化 (Probabilities)",
            description: "将得分映射到 <code>(0, 1)</code> 区间且总和为 1。被截断的词概率降为 0。<span class='step-badge'>generate_utils.py:295</span>",
            code: "probs = F.softmax(logits, dim=-1)",
            action: () => applySoftmax()
        },
        {
            title: "Step 4: Top-P (Nucleus) 采样",
            description: "核采样：按概率升序排列，剔除累积概率低于 <code>1-p</code> 的“长尾”部分。这使得模型能根据预测置信度动态调整候选池大小。<span class='step-badge'>generate_utils.py:144</span>",
            code: "cumulative_probs = sorted_probs.cumsum(dim=-1)\nmask = cumulative_probs <= (1 - p)",
            action: () => applyTopP()
        },
        {
            title: "Step 5: 多项式随机采样 (Multinomial)",
            description: "最后在过滤后的分布中进行一次“投骰子”。高概率词更有可能被选中，但低概率词仍有一线生机。<span class='step-badge'>generate_utils.py:303</span>",
            code: "next_token = torch.multinomial(probs, num_samples=1)",
            action: () => applySampling()
        }
    ];

    function updateUI() {
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

        try {
            step.action();
        } catch(e) {
            console.error("Action failed", e);
        }

        updateButtons();
    }

    function resetData() {
        workingLogits = JSON.parse(JSON.stringify(initialLogits));
        stats.innerHTML = "Original scores from language model.";
        renderBars(workingLogits, false);
    }

    function applyTemperature() {
        const t = (tempInput ? parseFloat(tempInput.value) : 1.0) || 1.0;
        workingLogits = JSON.parse(JSON.stringify(initialLogits));
        workingLogits.forEach(d => d.l /= t);
        stats.innerHTML = `Temperature T=${t}: ${t < 1 ? 'Focusing on high scores' : 'Smoothing distribution'}`;
        renderBars(workingLogits, false);
    }

    function applyTopK() {
        applyTemperature();
        const k = (topKInput ? parseInt(topKInput.value) : 5) || 5;
        let sorted = [...workingLogits].sort((a,b) => b.l - a.l);
        let threshold = sorted[k-1].l;
        workingLogits.forEach(d => { if (d.l < threshold) d.masked = true; });
        stats.innerHTML = `Keeping top ${k} tokens. Others set to -∞.`;
        renderBars(workingLogits, false);
    }

    function applySoftmax() {
        applyTopK();
        let sumExp = 0;
        workingLogits.forEach(d => { if (!d.masked) sumExp += Math.exp(d.l); });
        workingLogits.forEach(d => { d.p = d.masked ? 0 : Math.exp(d.l) / sumExp; });
        stats.innerHTML = "Mapped to probability space (0-1).";
        renderBars(workingLogits, true);
    }

    function applyTopP() {
        applySoftmax();
        workingLogits.sort((a,b) => a.p - b.p);
        let cum = 0;
        const p = (topPInput ? parseFloat(topPInput.value) : 0.9) || 0.9;
        const q = 1 - p;
        workingLogits.forEach(d => {
            cum += d.p;
            d.cum = cum;
            if (cum <= q) { d.masked = true; d.p = 0; }
        });
        stats.innerHTML = `Nucleus P=${p}: Dynamic filter based on cumulative mass.`;
        renderBars(workingLogits, true, true);
    }

    function applySampling() {
        applyTopP();
        let r = Math.random();
        let acc = 0;
        let winner = null;
        for(let d of workingLogits) {
            if (d.masked) continue;
            acc += d.p;
            if (r <= acc) { winner = d; break; }
        }
        if (!winner) winner = workingLogits.find(d => !d.masked);

        const cols = document.querySelectorAll('.bar-col');
        let count = 0;
        currentInterval = setInterval(() => {
            if (!document.querySelectorAll('.bar-col').length) {
                if(currentInterval) clearInterval(currentInterval);
                return;
            }
            cols.forEach(c => c.children[1].style.opacity = '0.5');
            cols[Math.floor(Math.random()*cols.length)].children[1].style.opacity = '1';
            if(++count > 10) {
                if(currentInterval) clearInterval(currentInterval);
                currentInterval = null;
                cols.forEach((c, i) => {
                    const isWinner = workingLogits[i].tok === winner.tok;
                    c.children[1].style.opacity = isWinner ? '1' : '0.2';
                    if(isWinner) c.children[1].style.background = '#2ecc71';
                });
                stats.innerHTML = `Sampled: <strong style="color:#2ecc71; font-size:1.4em;">${winner.tok}</strong>`;
            }
        }, 80);
    }

    function renderBars(data, showProb, showLine=false) {
        chart.innerHTML = '';
        // Fix: Set grid columns dynamically to prevent stacking
        chart.style.gridTemplateColumns = `repeat(${data.length}, 1fr)`;

        data.forEach(d => {
            const col = document.createElement('div');
            col.className = 'bar-col';
            const h = showProb ? d.p * 250 : (d.l + 1) * 40;
            col.innerHTML = `
                <div class="val-tag" style="font-size:9px;">${showProb ? (d.p*100).toFixed(1)+'%' : d.l.toFixed(1)}</div>
                <div class="prob-bar ${d.masked?'masked':''}" style="height:${Math.max(h, 4)}px; transition: all 0.5s;"></div>
                <div class="token-label" style="font-size:10px; margin-top:5px;">${d.tok}</div>
            `;
            chart.appendChild(col);
        });

        if (showLine) {
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.style.cssText = "position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;";
            let points = "";
            data.forEach((d, i) => {
                const x = (i + 0.5) * (chart.offsetWidth / data.length);
                const y = 300 - (d.cum * 250);
                points += `${x},${y} `;
            });
            svg.innerHTML = `<polyline points="${points}" fill="none" stroke="#f1c40f" stroke-width="2" stroke-dasharray="4" />`;
            chart.appendChild(svg);
        }
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
