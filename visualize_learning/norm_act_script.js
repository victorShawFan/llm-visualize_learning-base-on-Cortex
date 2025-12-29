document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    const gammaInput = document.getElementById('gammaInput');
    
    // Guard
    if (!visualContent) {
        console.error("Required elements not found in norm_act_script");
        return;
    }

    let currentStep = 0;
    let gamma = 1.5;

    const steps = [
        {
            title: "RMSNorm 步骤 1: 平方与均值 (Mean Square)",
            description: "RMSNorm 相比 LayerNorm 更轻量。它不减去均值，而是直接计算向量元素的平方均值（RMS 的平方），衡量向量的整体“能量”。<span class='step-badge'>llm_model.py:45</span>",
            code: `variance = hidden_states.pow(2).mean(-1, keepdim=True)`,
            state: "rms_sq"
        },
        {
            title: "RMSNorm 步骤 2: 倒数平方根归一化 (rsqrt)",
            description: "使用 <code>torch.rsqrt(variance + eps)</code> 得到缩放因子。将原向量除以其 RMS 长度，使得每个 Token 向量都被映射到超球面上。<span class='step-badge'>llm_model.py:47</span>",
            code: `hidden_states = hidden_states * torch.rsqrt(variance + eps)`,
            state: "rms_norm"
        },
        {
            title: "RMSNorm 步骤 3: 可学习缩放 (Gamma)",
            description: "归一化后，乘以可学习的 <code>weight</code>。这允许模型自主调节每一维度的重要性，恢复特征的表达能力。<span class='step-badge'>llm_model.py:49</span>",
            code: `return self.weight * hidden_states`,
            state: "rms_scale"
        },
        {
            title: "Residual 步骤 4: 残差连接 (Skip Connection)",
            description: "Transformer 的核心设计之一。将原始输入 <code>residual</code> 直接加到子模块（如 Attention 或 MLP）的输出上。这开辟了梯度的“高速公路”，有效解决了深层网络的梯度消失问题。<span class='step-badge'>llm_model.py:446</span>",
            code: `hidden_states = hidden_states + residual`,
            state: "residual"
        },
        {
            title: "SwiGLU 步骤 5: 双分支门控投影",
            description: "Llama 系列弃用了简单的 ReLU，改用 SwiGLU。它将输入投影到两个分支：Gate 和 Up。Gate 分支负责决定哪些信息可以通过。<span class='step-badge'>llm_model.py:122</span>",
            code: `gate = self.gate_proj(x)\nup = self.up_proj(x)`,
            state: "swiglu_proj"
        },
        {
            title: "SwiGLU 步骤 6: SiLU 激活与哈达玛积",
            description: "对 Gate 分支应用 SiLU 激活函数，然后与 Up 分支做逐元素相乘。这种机制比传统激活函数具有更强的非线性表达能力。<span class='step-badge'>llm_model.py:122</span>",
            code: `return self.down_proj(F.silu(gate) * up)`,
            state: "swiglu_down"
        }
    ];

    // Interactive State
    let inputVec = [2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0]; // Larger vector for better viz

    function init() {
        currentStep = 0;
        if (gammaInput) gamma = parseFloat(gammaInput.value) || 1.5;
        updateUI();
    }

    function updateUI() {
        const step = steps[currentStep];
        if (infoBox) infoBox.innerHTML = `<strong>${step.title}</strong><p style="margin-top:10px; font-size:0.95em;">${step.description}</p>`;
        if (codeSnippet) {
            codeSnippet.textContent = step.code;
            if(window.hljs) hljs.highlightElement(codeSnippet);
        }
        render(step.state);
        updateButtons();
    }

    function render(state) {
        if (!visualContent) return;
        visualContent.innerHTML = '';
        
        // RMS Calculation
        const sq = inputVec.map(v => v*v);
        const meanSq = sq.reduce((a,b)=>a+b,0) / sq.length;
        const rms = Math.sqrt(meanSq);
        const normed = inputVec.map(v => v / (rms + 1e-6));
        const scaled = normed.map(v => v * gamma);

        if (state === "rms_sq") {
            visualContent.appendChild(createBarChart("Input Vector", inputVec, "#3498db"));
            
            const calcBox = document.createElement('div');
            calcBox.className = 'tensor-box';
            calcBox.style.marginTop = '20px';
            calcBox.innerHTML = `
                <div>Squares: [${sq.map(v=>v.toFixed(1)).join(', ')}]</div>
                <div>Sum: ${sq.reduce((a,b)=>a+b,0).toFixed(1)}</div>
                <div style="font-weight:bold; color:#e74c3c;">Mean Square (Variance) = ${meanSq.toFixed(2)}</div>
                <div style="font-weight:bold; color:#e67e22;">RMS = ${rms.toFixed(2)}</div>
            `;
            visualContent.appendChild(calcBox);
        }
        else if (state === "rms_norm") {
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '20px'; row.style.alignItems = 'flex-start';
            
            row.appendChild(createBarChart("Input (High Variance)", inputVec, "#3498db"));
            
            const arrow = document.createElement('div');
            arrow.style.alignSelf = 'center';
            arrow.innerHTML = `<div style="text-align:center;">÷ ${rms.toFixed(2)}</div><div style="font-size:24px;">➡</div>`;
            row.appendChild(arrow);
            
            row.appendChild(createBarChart("Normalized (Unit RMS)", normed, "#2ecc71"));
            
            visualContent.appendChild(row);
            
            const statBox = document.createElement('div');
            statBox.className = 'op-label';
            statBox.innerText = `New RMS ≈ 1.00`;
            visualContent.appendChild(statBox);
        }
        else if (state === "rms_scale") {
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '20px'; row.style.alignItems = 'flex-start';
            
            row.appendChild(createBarChart("Normalized", normed, "#2ecc71"));
            
            const arrow = document.createElement('div');
            arrow.style.alignSelf = 'center';
            arrow.innerHTML = `<div style="text-align:center;">× ${gamma}</div><div style="font-size:24px;">➡</div>`;
            row.appendChild(arrow);
            
            row.appendChild(createBarChart("Scaled Output", scaled, "#9b59b6"));
            
            visualContent.appendChild(row);
        }
        else if (state === "residual") {
            // Visualize x + f(x)
            const residual = [1, 1, 1, 1, 1, 1, 1, 1];
            const output = inputVec.map((v, i) => v + residual[i]);
            
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '10px'; row.style.alignItems = 'center';
            
            row.appendChild(createBarChart("Input (Residual)", residual, "#95a5a6", 100));
            row.appendChild(document.createTextNode("+"));
            row.appendChild(createBarChart("Block Output", inputVec, "#3498db", 100));
            row.appendChild(document.createTextNode("="));
            row.appendChild(createBarChart("Result", output, "#34495e", 100));
            
            visualContent.appendChild(row);
        }
        else if (state === "swiglu_proj") {
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '40px'; row.style.justifyContent = 'center';
            
            // Mock projections
            const gate = inputVec.map(v => v * 0.5 + 1); // Mock W_g
            const up = inputVec.map(v => v * -0.5); // Mock W_u
            
            row.appendChild(createBarChart("Gate Projection (x W_g)", gate, "#f1c40f"));
            row.appendChild(createBarChart("Up Projection (x W_u)", up, "#3498db"));
            
            visualContent.appendChild(row);
        }
        else if (state === "swiglu_down") {
            // Mock calcs
            const gate = inputVec.map(v => v * 0.5 + 1);
            const up = inputVec.map(v => v * -0.5);
            
            // SiLU = x * sigmoid(x)
            const silu = (x) => x / (1 + Math.exp(-x));
            const activated_gate = gate.map(v => silu(v));
            const combined = activated_gate.map((v, i) => v * up[i]);
            
            const row = document.createElement('div');
            row.style.display = 'flex'; row.style.gap = '10px'; row.style.alignItems = 'center';
            
            row.appendChild(createBarChart("SiLU(Gate)", activated_gate, "#e67e22"));
            row.appendChild(document.createTextNode("⊗"));
            row.appendChild(createBarChart("Up", up, "#3498db"));
            row.appendChild(document.createTextNode("="));
            row.appendChild(createBarChart("Element-wise Mul", combined, "#27ae60"));
            
            visualContent.appendChild(row);
        }
    }

    function createBarChart(title, data, color, heightScale=150) {
        const container = document.createElement('div');
        container.style.textAlign = 'center';
        container.innerHTML = `<div style="font-size:0.8em; margin-bottom:5px;">${title}</div>`;
        
        const chart = document.createElement('div');
        chart.style.display = 'flex';
        chart.style.alignItems = 'flex-end';
        chart.style.justifyContent = 'center';
        chart.style.gap = '2px';
        chart.style.height = '100px';
        chart.style.borderBottom = '1px solid #ccc';
        chart.style.padding = '5px';
        
        // Find max magnitude for scaling
        const maxVal = 6.0; // Fixed scale for stability
        
        data.forEach(val => {
            const bar = document.createElement('div');
            const h = Math.abs(val) / maxVal * 80;
            bar.style.height = `${Math.min(h, 100)}%`;
            bar.style.width = '15px';
            bar.style.background = color;
            bar.style.opacity = val < 0 ? '0.5' : '1'; // Distinguish negative
            bar.title = val.toFixed(2);
            chart.appendChild(bar);
        });
        
        container.appendChild(chart);
        return container;
    }

    function updateButtons() {
        if(prevBtn) prevBtn.disabled = currentStep === 0;
        if(nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
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

    if(nextBtn) nextBtn.addEventListener('click', goNext);
    if(prevBtn) prevBtn.addEventListener('click', goPrev);
    if(resetBtn) resetBtn.addEventListener('click', init);
    if(gammaInput) {
        gammaInput.addEventListener('change', init);
        gammaInput.addEventListener('input', init); // Live update
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    init();
});
