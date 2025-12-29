document.addEventListener('DOMContentLoaded', () => {
    const posInput = document.getElementById('posInput');
    const seqLenInput = document.getElementById('seqLenInput');
    const ropeTypeSelect = document.getElementById('ropeType');
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet || !posInput || !seqLenInput || !ropeTypeSelect) {
        console.error("Required elements not found in RoPE script");
        return;
    }

    let currentStep = 0;
    let position = 1;
    let seqLen = 2048;
    const maxPos = 2048;
    const dim = 64; 
    const base = 10000;
    let autoPlayInterval = null; // Fix: Define global variable

    // Helper for creating dynamic visualizations
    const createPhasor = (id, label, freq, currentPos, color) => {
        const container = document.createElement('div');
        container.className = 'phasor-box';
        const angle = (currentPos * freq) % (2 * Math.PI);
        const deg = (angle * 180 / Math.PI).toFixed(1);
        
        container.innerHTML = `
            <div class="circle-viz-enhanced">
                <div class="axis-x"></div><div class="axis-y"></div>
                <div class="phasor-vec" style="transform: rotate(${-deg}deg); background: ${color}; box-shadow: 0 0 10px ${color};"></div>
                <div class="phasor-trail" style="border-color: ${color}"></div>
                <div class="angle-label" style="color:${color}">${deg}°</div>
            </div>
            <div class="phasor-meta">
                <strong>${label}</strong><br>
                Freq: ${freq.toFixed(4)}<br>
                Period: ${(2*Math.PI/freq).toFixed(1)} pos
            </div>
        `;
        return container;
    };

    const steps = [
        {
            title: "Step 0: 小学级直观理解 (The 'Hello World' of RoPE)",
            desc: "想象你是一个时钟的指针。<b>RoPE 的规则就是：你站在第几个格子(Position)，就顺时针转几个刻度。</b><br>这比“加法位置编码”更高级：它是通过<b>旋转角度</b>来标记位置的。",
            code: `# 伪代码演示：向量如何随位置旋转
vec = [0, 1]  # 初始向量（指向12点钟）
pos1 = rotate(vec, 30°)  # 位置1：转到1点钟
pos2 = rotate(vec, 60°)  # 位置2：转到2点钟
# 向量长度不变，唯有角度改变。`,
            render: () => {
                return `
                <div style="display:flex; flex-direction:column; align-items:center; gap:20px; height:350px;">
                    <div style="position:relative; width:200px; height:200px; border:4px solid #34495e; border-radius:50%; background:white; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                        <!-- Clock markings -->
                        <div style="position:absolute; top:10px; left:95px; font-weight:bold; color:#ccc;">12</div>
                        <div style="position:absolute; top:95px; right:10px; font-weight:bold; color:#ccc;">3</div>
                        <div style="position:absolute; bottom:10px; left:95px; font-weight:bold; color:#ccc;">6</div>
                        <div style="position:absolute; top:95px; left:10px; font-weight:bold; color:#ccc;">9</div>
                        
                        <!-- The Vector Hand -->
                        <div id="demoHand" style="position:absolute; top:50%; left:50%; width:4px; height:80px; background:#e74c3c; transform-origin:bottom center; transform:translate(-50%, -100%) rotate(0deg); transition: transform 1s cubic-bezier(0.68, -0.55, 0.27, 1.55); border-radius:4px;"></div>
                        <div style="position:absolute; top:50%; left:50%; width:12px; height:12px; background:#34495e; transform:translate(-50%, -50%); border-radius:50%;"></div>
                    </div>
                    
                    <div style="display:flex; gap:10px;">
                        <button class="demo-btn" data-pos="0">位置 0 (0°)</button>
                        <button class="demo-btn" data-pos="1">位置 1 (30°)</button>
                        <button class="demo-btn" data-pos="2">位置 2 (60°)</button>
                        <button class="demo-btn" data-pos="3">位置 3 (90°)</button>
                    </div>
                    <div id="demoText" style="font-size:18px; font-weight:bold; color:#2c3e50; min-height:24px;">我是单词向量，我在位置 0</div>
                </div>
                `;
            }
        },
        {
            title: "Phase 1: 频率谱 (Frequency Spectrum)",
            desc: "RoPE 的第一步是为每个维度对 (Dim i) 计算旋转频率 <code>theta_i</code>。频率随维度指数衰减：低维转得快（关注局部），高维转得慢（关注全局）。",
            code: `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))`,
            render: () => {
                let html = `<div style="display:flex; align-items:flex-end; height:150px; gap:2px; width:100%; justify-content:center;">`;
                for(let i=0; i<32; i++) {
                    const freq = 1.0 / Math.pow(base, (2*i)/dim);
                    const height = Math.max(5, freq * 140);
                    const color = i < 4 ? '#e74c3c' : (i > 28 ? '#3498db' : '#95a5a6');
                    html += `<div class="freq-bar anim-grow-up" style="height:${height}px; background:${color}; width:10px; opacity:0.8;" title="Dim ${2*i}: ${freq.toFixed(4)}"></div>`;
                }
                html += `</div>
                <div style="display:flex; justify-content:space-between; width:80%; margin-top:10px; font-size:12px; color:#666;">
                    <span>Dim 0 (High Freq)</span>
                    <span>Dim 64 (Low Freq)</span>
                </div>`;
                return html;
            }
        },
        {
            title: "Phase 2: 相量旋转 (Phasor Rotation)",
            desc: "在 Position = <b>" + position + "</b> 时，不同维度的向量旋转角度不同。左边是低维度（快速旋转），右边是高维度（慢速旋转）。",
            code: `angles = position * inv_freq\n# Low Dim rotates fast, High Dim rotates slow`,
            render: () => {
                const container = document.createElement('div');
                container.style.display = 'flex';
                container.style.gap = '40px';
                container.style.justifyContent = 'center';
                
                // Dim 0 (Fast)
                const freq0 = 1.0; 
                const p1 = createPhasor('p1', 'Dim 0 (High Freq)', freq0, position, '#e74c3c');
                
                // Dim 32 (Medium)
                const freqMid = 1.0 / Math.pow(base, 32/64); 
                const p2 = createPhasor('p2', 'Dim 32 (Mid Freq)', freqMid, position, '#f1c40f');

                // Dim 62 (Slow)
                const freqEnd = 1.0 / Math.pow(base, 62/64);
                const p3 = createPhasor('p3', 'Dim 62 (Low Freq)', freqEnd, position, '#3498db');

                container.appendChild(p1);
                container.appendChild(p2);
                container.appendChild(p3);
                
                // Add Auto Play Control
                const controls = document.createElement('div');
                controls.style.marginTop = '20px';
                controls.innerHTML = `<button id="autoPlayBtn" style="padding:8px 16px; background:#2ecc71; color:white; border:none; border-radius:4px; cursor:pointer;">▶ Auto Rotate</button>`;
                container.appendChild(controls);
                
                return container;
            }
        },
        {
            title: "Phase 3: 相对位置编码 (Relative Attention)",
            desc: "Attention Score 取决于 Query 和 Key 的相对角度差。无论它们绝对位置在哪，只要相对距离 <code>m-n</code> 相同，点积结果就相同（旋转不变性）。",
            code: `Score = Q · K = |Q||K| cos(θ_q - θ_k)\n# Δθ depends only on (pos_q - pos_k)`,
            render: () => {
                const posQ = position;
                const posK = Math.max(0, position - 5); // K is 5 steps behind
                const freq = 0.5; // Example freq
                
                const angleQ = posQ * freq;
                const angleK = posK * freq;
                const diff = angleQ - angleK;
                
                return `
                <div style="display:flex; gap:30px; align-items:center;">
                    <div class="circle-viz-enhanced" style="width:120px; height:120px;">
                        <div class="axis-x"></div><div class="axis-y"></div>
                        <div class="phasor-vec" style="transform: rotate(${-angleQ}rad); background: #e74c3c; height:50%;"></div>
                        <div class="phasor-vec" style="transform: rotate(${-angleK}rad); background: #3498db; height:50%;"></div>
                        <div class="angle-arc" style="transform: rotate(${-angleQ}rad) scaleX(-1);"></div>
                    </div>
                    <div style="font-family:monospace; font-size:14px;">
                        <div>Pos Q: ${posQ} <span style="color:#e74c3c">■</span></div>
                        <div>Pos K: ${posK} <span style="color:#3498db">■</span></div>
                        <hr>
                        <div>Diff: ${(posQ-posK)}</div>
                        <div>Dot: <b>${Math.cos(diff).toFixed(3)}</b></div>
                    </div>
                </div>
                <p style="font-size:12px; color:#666; margin-top:10px;">拖动下方 Position 滑块，观察虽然 Q 和 K 都在转，但夹角（Dot Product）保持不变！</p>
                `;
            }
        },
        {
            title: "Phase 4: 远程衰减 (Long-term Decay)",
            desc: "在实际中，由于多频率叠加，Attention Score 会随着相对距离增加而震荡衰减。这使得模型天然偏向关注附近的 Token。",
            code: `# Interaction of multiple frequencies naturally causes decay`,
            render: () => {
                let svg = `<svg width="400" height="100" style="background:#f8f9fa; border:1px solid #ddd;">`;
                // Draw decay curve
                let path = "M 0 50 ";
                for(let x=0; x<400; x++) {
                    let y = 0;
                    // Sum of a few freqs
                    for(let k=0; k<5; k++) {
                        y += Math.cos(x * 0.1 * (k+1));
                    }
                    y = y/5 * 40; // scale
                    path += `L ${x} ${50 - y} `;
                }
                svg += `<path d="${path}" fill="none" stroke="#2c3e50" stroke-width="1.5" />`;
                svg += `</svg>`;
                return `<div style="text-align:center;">${svg}<p>Attention Score vs Distance</p></div>`;
            }
        },
        {
            title: "Phase 5: YaRN (Long Context) - \"混合频率\"对策",
            desc: "当我们要把上下文窗口从 2k 拉长到 128k 时，直接外推会让模型“发晕”。<br><b>YaRN (Yet another RoPE extension)</b> 使用了一种<b>冷热混合策略</b>：<br>1. <b>高频维度（细节）</b>：保持原样，不做线性拉伸（Interpolation），防止分辨率丢失。<br>2. <b>低频维度（宏观）</b>：进行线性插值，像拉长橡皮筋一样适应更长的距离。<br>3. <b>中间维度</b>：使用 Ramp 函数平滑过渡。",
            code: `# YaRN Logic:
# high_freq: no change (extrapolate)
# low_freq: linear interpolation (stretch)
ramp = linear_ramp(dim)
freq = freq_interp * (1-ramp) + freq_extrap * ramp`,
            render: () => {
                // Visualization of Frequency Handling
                return `
                <div style="display:flex; flex-direction:column; gap:15px; width:100%;">
                    
                    <!-- Concept Visual -->
                    <div style="display:flex; align-items:center; gap:10px; justify-content:center;">
                        <div style="text-align:center;">
                            <div style="font-weight:bold; color:#e74c3c; margin-bottom:5px;">High Freq (细节)</div>
                            <div class="yarn-wave" style="width:100px; height:40px; background:repeating-linear-gradient(90deg, #e74c3c 0, #e74c3c 2px, transparent 2px, transparent 10px);"></div>
                            <div style="font-size:12px; color:#666;">不拉伸 (Extrapolation)</div>
                        </div>
                        <div style="font-size:24px; color:#aaa;">+</div>
                        <div style="text-align:center;">
                            <div style="font-weight:bold; color:#3498db; margin-bottom:5px;">Low Freq (宏观)</div>
                            <div class="yarn-wave" style="width:100px; height:40px; background:repeating-linear-gradient(90deg, #3498db 0, #3498db 2px, transparent 2px, transparent 40px);"></div>
                            <div style="font-size:12px; color:#666;">拉伸 (Interpolation)</div>
                        </div>
                    </div>

                    <!-- Spectrum Gradient -->
                    <div style="position:relative; width:100%; height:80px; background:#f0f0f0; border-radius:8px; overflow:hidden; border:1px solid #ccc; margin-top:10px;">
                        <div style="position:absolute; top:0; left:0; height:100%; width:100%; background: linear-gradient(90deg, rgba(231,76,60,0.2) 0%, rgba(231,76,60,0.2) 30%, rgba(52,152,219,0.2) 70%, rgba(52,152,219,0.2) 100%);"></div>
                        
                        <!-- Labels -->
                        <div style="position:absolute; top:50%; left:10%; transform:translateY(-50%); font-weight:bold; color:#c0392b;">High Freq<br>(No Scale)</div>
                        <div style="position:absolute; top:50%; right:10%; transform:translateY(-50%); font-weight:bold; color:#2980b9;">Low Freq<br>(Scale L/L_train)</div>
                        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:12px; color:#555; background:white; padding:2px 8px; border-radius:10px; border:1px solid #999;">Ramp Mixing Region</div>
                        
                        <!-- Dashed Lines -->
                        <div style="position:absolute; top:0; left:33%; height:100%; border-left:2px dashed #999;"></div>
                        <div style="position:absolute; top:0; right:33%; height:100%; border-right:2px dashed #999;"></div>
                    </div>
                    
                    <p style="font-size:13px; color:#555; text-align:center; margin:0;">
                        这就好比看一张超长的全景图：近处的细节（高频）我们还是用原来的放大镜看；<br>远处的大轮廓（低频）我们把它缩小（拉伸）放进视野里。
                    </p>
                </div>`;
            }
        },
        {
            title: "Step 6: 实战演练 - 初始化 (Example Setup)",
            desc: "我们用一个极简向量来模拟 Token 'Data'。维度 Dim=4（包含 2 个旋转对）。<br>初始向量 <b>x</b> = [1, 0, 1, 0]。",
            code: `x = torch.tensor([1.0, 0.0, 1.0, 0.0])
# Dim 0,1 (Pair 1): High Freq
# Dim 2,3 (Pair 2): Low Freq`,
            render: () => {
                return `
                <div style="display:flex; justify-content:center; gap:40px; align-items:center; height:200px;">
                    <div class="matrix-box" style="padding:20px; border:2px solid #2c3e50; border-radius:8px; background:white;">
                        <div style="font-weight:bold; margin-bottom:10px; border-bottom:1px solid #eee; padding-bottom:5px;">Input Vector x</div>
                        <div style="display:flex; gap:10px; font-family:monospace; font-size:18px;">
                            <div style="color:#e74c3c">[1.0, 0.0]</div>
                            <div style="color:#3498db">[1.0, 0.0]</div>
                        </div>
                        <div style="display:flex; gap:10px; font-size:12px; color:#999; margin-top:5px;">
                            <div style="width:90px;">Pair 1 (High)</div>
                            <div style="width:90px;">Pair 2 (Low)</div>
                        </div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Step 7: 标准 RoPE 旋转 (Standard Rotation)",
            desc: "假设当前位置 Pos = 10。我们看看这两个对分别转了多少度。<br>Pair 1 (高频) 转得快，Pair 2 (低频) 转得慢。",
            code: `pos = 10
theta_0 = 1.0;  theta_1 = 0.1
# Pair 1 angle: 10 * 1.0 = 10 rad ≈ 573° (1.6 圈)
# Pair 2 angle: 10 * 0.1 = 1.0 rad ≈ 57°`,
            render: () => {
                return `
                <div style="display:flex; justify-content:center; gap:50px; align-items:center;">
                    <!-- Pair 1 -->
                    <div style="text-align:center;">
                        <div style="font-weight:bold; color:#e74c3c; margin-bottom:10px;">Pair 1 (High Freq)</div>
                        <div class="circle-viz-enhanced" style="width:120px; height:120px;">
                            <div class="axis-x"></div><div class="axis-y"></div>
                            <div class="phasor-vec" style="transform: rotate(-213deg); background: #e74c3c; height:50%;"></div>
                            <div class="phasor-trail" style="border-color: #e74c3c"></div>
                            <div class="angle-label" style="color:#e74c3c">573°</div>
                        </div>
                        <div style="font-size:12px; margin-top:10px;">Angle = 10 rad</div>
                    </div>

                    <!-- Pair 2 -->
                    <div style="text-align:center;">
                        <div style="font-weight:bold; color:#3498db; margin-bottom:10px;">Pair 2 (Low Freq)</div>
                        <div class="circle-viz-enhanced" style="width:120px; height:120px;">
                            <div class="axis-x"></div><div class="axis-y"></div>
                            <div class="phasor-vec" style="transform: rotate(-57deg); background: #3498db; height:50%;"></div>
                            <div class="phasor-trail" style="border-color: #3498db"></div>
                            <div class="angle-label" style="color:#3498db">57°</div>
                        </div>
                        <div style="font-size:12px; margin-top:10px;">Angle = 1.0 rad</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Step 8: 长文本危机 (Long Context Issue)",
            desc: "现在位置来到了 Pos = 100 (超出训练长度)！<br>Pair 2 (低频) 本来应该转得慢，现在也转了 10 弧度，模型从未见过这种“狂转”的低频特征，直接懵圈。",
            code: `pos = 100 (Out of Distribution)
# Pair 2 angle: 100 * 0.1 = 10 rad
# Model panic: "Low freq features shouldn't spin this fast!"`,
            render: () => {
                return `
                <div style="display:flex; justify-content:center; gap:50px; align-items:center;">
                    <!-- Pair 1 -->
                    <div style="text-align:center; opacity:0.5;">
                        <div style="font-weight:bold; color:#e74c3c; margin-bottom:10px;">Pair 1 (High Freq)</div>
                        <div class="circle-viz-enhanced" style="width:100px; height:100px;">
                            <div class="phasor-vec" style="animation: spin 0.5s linear infinite; background: #e74c3c;"></div>
                        </div>
                        <div>Too fast anyway</div>
                    </div>

                    <!-- Pair 2 -->
                    <div style="text-align:center;">
                        <div style="font-weight:bold; color:#3498db; margin-bottom:10px;">Pair 2 (Low Freq) 😱</div>
                        <div class="circle-viz-enhanced" style="width:120px; height:120px;">
                             <div class="axis-x"></div><div class="axis-y"></div>
                            <div class="phasor-vec" style="transform: rotate(-213deg); background: #3498db; height:50%; box-shadow:0 0 15px red;"></div>
                            <div class="angle-label" style="color:red; font-weight:bold;">573° (10 rad)</div>
                        </div>
                        <div style="font-size:12px; margin-top:10px; color:red; font-weight:bold;">OOD Error: Angle too large!</div>
                    </div>
                </div>`;
            }
        },
        {
            title: "Step 9: YaRN 救场 (YaRN Correction)",
            desc: "YaRN 介入：强制把低频对的频率除以扩充倍数 (Scale=2)。<br>现在 Pair 2 只转了 5 弧度（看起来像是在 Pos 50），模型觉得“这我熟”，于是成功理解长文。",
            code: `# YaRN Strategy: Scale = 2
theta_1_new = theta_1 / 2 = 0.05
# New Angle: 100 * 0.05 = 5.0 rad
# Effective Pos: 50 (Within distribution)`,
            render: () => {
                return `
                <div style="display:flex; justify-content:center; gap:50px; align-items:center;">
                    <!-- Pair 1 -->
                    <div style="text-align:center;">
                        <div style="font-weight:bold; color:#e74c3c; margin-bottom:10px;">Pair 1 (High Freq)</div>
                        <div style="background:#eee; padding:5px; border-radius:4px; font-size:12px;">No Change</div>
                    </div>

                    <div style="font-size:30px;">→</div>

                    <!-- Pair 2 -->
                    <div style="text-align:center;">
                        <div style="font-weight:bold; color:#3498db; margin-bottom:10px;">Pair 2 (Low Freq) ✅</div>
                        <div class="circle-viz-enhanced" style="width:120px; height:120px;">
                             <div class="axis-x"></div><div class="axis-y"></div>
                            <div class="phasor-vec" style="transform: rotate(-286deg); background: #2ecc71; height:50%; box-shadow:0 0 15px #2ecc71;"></div>
                            <div class="angle-label" style="color:#2ecc71; font-weight:bold;">286° (5 rad)</div>
                        </div>
                        <div style="font-size:12px; margin-top:10px; color:#27ae60; font-weight:bold;">Scaled Down!<br>Effective Pos = 50</div>
                    </div>
                </div>`;
            }
        }
    ];

    function updateUI() {
        const step = steps[currentStep];
        // Fix: Use step.desc if step.description is missing
        const description = step.desc || step.description;
        infoBox.innerHTML = `<h3>${step.title}</h3><p>${description}</p>`;
        codeSnippet.textContent = step.code;
        
        if (window.hljs) hljs.highlightElement(codeSnippet);

        const content = step.render();
        if (typeof content === 'string') {
            visualContent.innerHTML = content;
        } else {
            visualContent.innerHTML = '';
            visualContent.appendChild(content);
        }

        prevBtn.disabled = currentStep === 0;
        nextBtn.disabled = currentStep === steps.length - 1;

        // --- Event Binding for Interactive Elements ---
        
        // Step 0: Clock Demo
        if (currentStep === 0) {
            const btns = document.querySelectorAll('.demo-btn');
            btns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const pos = parseInt(e.target.dataset.pos);
                    const hand = document.getElementById('demoHand');
                    const text = document.getElementById('demoText');
                    if(hand && text) {
                        const angle = pos * 30;
                        hand.style.transform = `translate(-50%, -100%) rotate(${angle}deg)`;
                        text.innerText = `我是单词向量，我在位置 ${pos} (旋转 ${angle}°)`;
                    }
                });
            });
        }

        // Phase 2: Auto Play
        if (currentStep === 2) { // Note: Array index 2 corresponds to "Phase 2" because we added Step 0 at index 0
             const autoBtn = document.getElementById('autoPlayBtn');
             if(autoBtn) {
                 autoBtn.addEventListener('click', () => {
                     if (autoPlayInterval) {
                         clearInterval(autoPlayInterval);
                         autoPlayInterval = null;
                         autoBtn.textContent = "▶ Auto Rotate";
                         autoBtn.style.background = "#2ecc71";
                     } else {
                         autoBtn.textContent = "⏸ Stop";
                         autoBtn.style.background = "#e74c3c";
                         autoPlayInterval = setInterval(() => {
                             position = (position + 1) % 100;
                             posInput.value = position;
                             // Re-render only the phasors, not the whole UI to avoid flickering?
                             // Ideally yes, but for simplicity let's just trigger updateUI or partial update.
                             // Calling updateUI() will kill the interval because it re-renders button.
                             // So we should manually update the phasors here.
                             
                             // Update logic for Phase 2 Phasors
                             const p1 = document.querySelector('#visualContent > div > div:nth-child(1)'); // P1 container
                             const p2 = document.querySelector('#visualContent > div > div:nth-child(2)'); // P2 container
                             const p3 = document.querySelector('#visualContent > div > div:nth-child(3)'); // P3 container
                             
                             const updatePhasor = (el, freq, pos) => {
                                 if(!el) return;
                                 const angle = (pos * freq) % (2 * Math.PI);
                                 const deg = (angle * 180 / Math.PI).toFixed(1);
                                 const vec = el.querySelector('.phasor-vec');
                                 const lbl = el.querySelector('.angle-label');
                                 if(vec) vec.style.transform = `rotate(${-deg}deg)`;
                                 if(lbl) lbl.innerText = `${deg}°`;
                             };

                             const freq0 = 1.0;
                             const freqMid = 1.0 / Math.pow(base, 32/64);
                             const freqEnd = 1.0 / Math.pow(base, 62/64);

                             updatePhasor(p1, freq0, position);
                             updatePhasor(p2, freqMid, position);
                             updatePhasor(p3, freqEnd, position);

                         }, 50);
                     }
                 });
             }
        } else {
            // Clear interval if leaving Phase 2
            if (autoPlayInterval) {
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
            }
        }
    }

    // Navigation Logic
    function goNext() {
        if(currentStep < steps.length - 1) { 
            currentStep++; 
            updateUI(); 
        }
    }

    function goPrev() {
        if(currentStep > 0) { 
            currentStep--; 
            updateUI(); 
        }
    }

    // Event Listeners
    nextBtn.addEventListener('click', goNext);
    prevBtn.addEventListener('click', goPrev);
    
    // Position slider creates live animation
    posInput.addEventListener('input', (e) => {
        position = parseInt(e.target.value);
        // Only update if current step uses position (Phase 2 & 3)
        // Step 0 is index 0
        // Phase 1 is index 1 (Spectrum)
        // Phase 2 is index 2 (Phasor) -> Uses position
        // Phase 3 is index 3 (Relative) -> Uses position
        if (currentStep === 2 || currentStep === 3) {
            updateUI();
        }
    });

    resetBtn.addEventListener('click', () => {
        currentStep = 0;
        position = 1;
        posInput.value = 1;
        updateUI();
    });
    
    // Remove old auto-play logic at bottom
    
    updateUI();

    // Fix: Global Keyboard Navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });
});
