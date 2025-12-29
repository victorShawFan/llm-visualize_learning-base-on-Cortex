document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const codeSnippet = document.getElementById('codeSnippet');
    const chartContainer = document.getElementById('chart');
    const optState = document.getElementById('optState');
    const warmupInput = document.getElementById('warmupInput');
    const periodMulInput = document.getElementById('periodMulInput');
    
    // Guard
    if (!chartContainer) {
        console.error("Required elements not found in scheduler_script");
        return;
    }

    let currentStep = 0;
    
    // Config
    const TOTAL_TRAIN_STEPS = 400;
    const CONFIG = {
        warmupIters: 20,
        initialLR: 0.0001,
        minLR: 0.0001,
        maxLR: 0.001,
        cosinePeriod: 50,
        periodMul: 2
    };

    let LR_TIMELINE = [];

    const steps = [
        {
            title: "Step 0: LRScheduler 抽象接口总览",
            description: "所有学习率策略都实现同一个接口：\n<code>cur_steps / cur_lr / step() / can_clip_grad() / get_ckpt_dict() / restore_ckpt_dict()</code>，这样 Trainer 只需要持有一个调度器实例即可。<span class='step-badge'>scheduler.py:6-25</span>",
            code: `class LRScheduler(ABC):\n    @property\n    def cur_steps(self): ...\n    @property\n    def cur_lr(self): ...\n    def step(self): ...\n    def can_clip_grad(self): ...`,
            focusStep: 0,
            phase: "overview"
        },
        {
            title: "Step 1: WarmupCosineAnnealingLR 初始化参数",
            description: "调度器接收 <code>warmup_iters</code>、<code>initial_lr</code>、<code>min_lr</code>、<code>max_lr</code>、<code>cosine_annealing_period</code> 和 <code>cosine_annealing_period_mul</code> 等配置，并绑定到一个优化器上。<span class='step-badge'>scheduler.py:28-51</span>",
            code: `scheduler = WarmupCosineAnnealingLRScheduler(\n    optimizer=optim,\n    warmup_iters=warmup,\n    initial_lr=initial_lr, min_lr=min_lr, max_lr=max_lr,\n    cosine_annealing_period=period,\n    cosine_annealing_period_mul=period_mul\n)`,
            focusStep: 0,
            phase: "config"
        },
        {
            title: "Step 2: 线性 Warmup 阶段 (warmup_iters)",
            description: "前 <code>warmup_iters</code> 步使用线性上升：<code>initial_lr → max_lr</code>。源码中预先计算 <code>_lr_increment=(max_lr-initial_lr)/warmup_iters</code>，每步叠加一次。<span class='step-badge'>scheduler.py:55-58,88-93</span>",
            code: `if steps <= warmup_iters:\n    lr = initial_lr + steps * _lr_increment`,
            focusStep: 10,
            phase: "warmup"
        },
        {
            title: "Step 3: can_clip_grad 何时允许裁剪梯度",
            description: "在 warmup 阶段不做梯度裁剪，只有当 <code>steps &gt; warmup_iters</code> 时 <code>can_clip_grad()</code> 才返回 True，从而保护前期训练稳定性。<span class='step-badge'>scheduler.py:75-81</span>",
            code: `def can_clip_grad(self):\n    return self._steps > self._warmup_iters`,
            focusStep: CONFIG.warmupIters,
            phase: "clip"
        },
        {
            title: "Step 4: 进入余弦阶段并记录 base_lr",
            description: "一旦越过 warmup，调度器在第一次进入余弦区间时将 <code>_cosine_annealing_base_lr</code> 设为当前 lr，后续所有周期都围绕这个基准 lr 在 <code>[min_lr, base_lr]</code> 区间内做余弦震荡。<span class='step-badge'>scheduler.py:95-97</span>",
            code: `if not self._cosine_annealing_base_lr:\n    self._cosine_annealing_base_lr = self.cur_lr`,
            focusStep: CONFIG.warmupIters + 1,
            phase: "base_lr"
        },
        {
            title: "Step 5: 单周期内的 Cosine Annealing",
            description: "在当前周期内，调度器使用 <code>T_max</code> 步完成一次从 base_lr → min_lr → base_lr 的余弦退火：<code>lr = min_lr + (base_lr - min_lr) * (1 + cos(π·t/T_max))/2</code>。<span class='step-badge'>scheduler.py:98-117</span>",
            code: `T_max = period * (max(period_mul, 1) ** cycle)\nT_cur += 1\ncos_factor = (1 + cos(pi * T_cur / T_max)) / 2\nlr = min_lr + (base_lr - min_lr) * cos_factor`,
            focusStep: CONFIG.warmupIters + Math.floor(CONFIG.cosinePeriod / 2),
            phase: "cosine0"
        },
        {
            title: "Step 6: 多周期与周期倍增 (cycle, T_cur)",
            description: "每当 <code>T_cur ≥ T_max</code>，如果 <code>period_mul &gt; 0</code>，就将 <code>cycle += 1</code> 并重置 <code>T_cur=0</code>，下一周期的 <code>T_max</code> 会按倍数增长，实现 Cosine with Restarts。<span class='step-badge'>scheduler.py:100-113</span>",
            code: `T_max = period * (max(period_mul, 1) ** cycle)\nif T_cur >= T_max:\n    cycle += 1\n    T_cur = 0`,
            focusStep: CONFIG.warmupIters + CONFIG.cosinePeriod + 5,
            phase: "cycle"
        },
        {
            title: "Step 7: period_mul = 0 的单周期退火模式",
            description: "当 <code>cosine_annealing_period_mul = 0</code> 时，调度器退化成『单周期余弦退火』：超过 <code>warmup + cosine_annealing_period</code> 后，lr 恒为 <code>min_lr</code>。<span class='step-badge'>scheduler.py:83-86,106-110</span>",
            code: `if period_mul == 0 and steps >= warmup_iters + period:\n    lr = min_lr  # 之后不再上升`,
            focusStep: CONFIG.warmupIters + CONFIG.cosinePeriod + 30,
            phase: "single_cycle"
        },
        {
            title: "Step 8: 完整曲线：Warmup + 多轮 Cosine",
            description: "把以上逻辑串起来，就得到一条先线性升高、再多周期余弦震荡、最终在低 lr 带内缓慢探索的学习率曲线。图中所有橙色点即为对 Python 调度器逐步模拟的结果。",
            code: `for step in range(total_steps):\n    scheduler.step()\n    lrs.append(scheduler.cur_lr)`,
            focusStep: TOTAL_TRAIN_STEPS,
            phase: "full_curve"
        },
        {
            title: "Step 9: Checkpoint 保存与恢复",
            description: "为了在中断后继续训练，调度器会将当前 lr、已走步数、base_lr、T_cur 与 cycle 序列化到 checkpoint 中，并在恢复时完整还原内部状态后再调用一次 <code>_update_lr()</code>。<span class='step-badge'>scheduler.py:127-153</span>",
            code: `ckpt = scheduler.get_ckpt_dict()\n# 保存到 ckpt\n...\n# 恢复\nscheduler.restore_ckpt_dict(ckpt)`,
            focusStep: Math.floor(TOTAL_TRAIN_STEPS * 0.6),
            phase: "ckpt"
        },
        {
            title: "Step 10: NoneLRScheduler – 固定学习率模式",
            description: "当不需要调度时，使用 <code>NoneLRScheduler</code>：始终返回固定 lr，<code>cur_steps=-1</code>，<code>can_clip_grad()</code> 永远 True，仅负责把当前 lr 写入 checkpoint。<span class='step-badge'>scheduler.py:155-177</span>",
            code: `scheduler = NoneLRScheduler(initial_lr)\n# step() 空实现，cur_lr 永远不变`,
            focusStep: 0,
            phase: "none"
        }
    ];

    // Logic
    function simulateSchedulerTimeline(totalSteps, cfg) {
        let steps = -1;
        let currentLR = cfg.initialLR;
        let T_cur = 0;
        let cycle = 0;
        let baseLR = null;
        const lrIncrement = cfg.warmupIters !== 0 ? (cfg.maxLR - cfg.initialLR) / cfg.warmupIters : 0;
        const timeline = [];

        for (let i = 0; i <= totalSteps; i++) {
            steps += 1;
            let lr;
            let phase = '';

            if (cfg.periodMul === 0 && steps >= cfg.cosinePeriod + cfg.warmupIters) {
                lr = cfg.minLR;
                phase = 'min_locked';
            } else if (steps <= cfg.warmupIters) {
                lr = cfg.initialLR + steps * lrIncrement;
                phase = 'warmup';
            } else {
                if (baseLR === null) baseLR = currentLR;
                const periodMul = Math.max(cfg.periodMul, 1);
                let T_max = cfg.cosinePeriod * Math.pow(periodMul, cycle);
                T_cur += 1;
                let calc_t = T_cur;
                if (T_cur >= T_max) {
                    if (cfg.periodMul === 0) { T_cur = T_max; calc_t = T_max; }
                    else { cycle += 1; T_cur = 0; calc_t = T_max; }
                }
                phase = 'cosine';
                const cosFactor = (1 + Math.cos(Math.PI * calc_t / T_max)) / 2;
                lr = cfg.minLR + (baseLR - cfg.minLR) * cosFactor;
            }
            currentLR = lr;
            timeline.push({ step: steps, lr, T_cur, cycle, phase });
        }
        return timeline;
    }

    // Canvas
    let canvas = null;
    let ctx = null;

    function initCanvas() {
        if (!chartContainer) return;
        chartContainer.innerHTML = '';
        canvas = document.createElement('canvas');
        // Force some default if container hidden
        const w = chartContainer.clientWidth || 500;
        const h = chartContainer.clientHeight || 300;
        
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        chartContainer.appendChild(canvas);
        ctx = canvas.getContext('2d');
        
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);
    }

    function drawCanvas() {
        if (!ctx) initCanvas();
        if (!canvas) return;
        
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.width / dpr;
        const height = canvas.height / dpr;
        
        ctx.clearRect(0, 0, width, height);
        
        // Grid
        ctx.beginPath();
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        for(let i=0; i<=5; i++) {
            const y = height - (i/5) * (height-20);
            ctx.moveTo(30, y);
            ctx.lineTo(width, y);
        }
        ctx.stroke();

        // Curve
        if (LR_TIMELINE.length === 0) return;
        ctx.beginPath();
        ctx.lineWidth = 2;
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#e74c3c');
        gradient.addColorStop(CONFIG.warmupIters / TOTAL_TRAIN_STEPS, '#f39c12');
        gradient.addColorStop(1, '#3498db');
        ctx.strokeStyle = gradient;

        const xScale = (width - 40) / TOTAL_TRAIN_STEPS;
        const yScale = (height - 40) / CONFIG.maxLR;

        LR_TIMELINE.forEach((p, i) => {
            const x = 30 + p.step * xScale;
            const y = height - 20 - p.lr * yScale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // Fill
        ctx.lineTo(30 + TOTAL_TRAIN_STEPS * xScale, height - 20);
        ctx.lineTo(30, height - 20);
        ctx.fillStyle = 'rgba(241, 196, 15, 0.1)';
        ctx.fill();
        
        // Labels
        ctx.fillStyle = '#666';
        ctx.font = '10px Menlo';
        ctx.fillText('0', 10, height - 20);
        ctx.fillText(CONFIG.maxLR.toExponential(1), 0, height - 20 - CONFIG.maxLR * yScale);
        ctx.fillText(TOTAL_TRAIN_STEPS, width - 30, height - 5);
    }

    function drawScanner(stepIndex) {
        if (!ctx) return;
        drawCanvas(); // naive redraw
        
        const p = LR_TIMELINE[Math.min(stepIndex, LR_TIMELINE.length - 1)];
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.width / dpr;
        const height = canvas.height / dpr;
        const xScale = (width - 40) / TOTAL_TRAIN_STEPS;
        const yScale = (height - 40) / CONFIG.maxLR;
        
        const x = 30 + p.step * xScale;
        const y = height - 20 - p.lr * yScale;
        
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#2ecc71';
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#2ecc71';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    function updateParams() {
        if(warmupInput) CONFIG.warmupIters = parseInt(warmupInput.value) || 0;
        if(periodMulInput) CONFIG.periodMul = parseInt(periodMulInput.value) || 0;
        LR_TIMELINE = simulateSchedulerTimeline(TOTAL_TRAIN_STEPS, CONFIG);
        drawCanvas();
        render();
    }

    function render() {
        const stepMeta = steps[currentStep];
        const idx = Math.min(stepMeta.focusStep, LR_TIMELINE.length - 1);
        const p = LR_TIMELINE[idx];
        
        drawScanner(idx);

        let T_max = 0;
        if (p.step > CONFIG.warmupIters) {
            const periodMul = Math.max(CONFIG.periodMul, 1);
            T_max = CONFIG.cosinePeriod * Math.pow(periodMul, p.cycle);
        }

        if(infoBox) {
            infoBox.innerHTML = `
                <div class="step-badge">Step ${currentStep} / ${steps.length - 1}</div>
                <strong>${stepMeta.title}</strong><br>
                <div style="margin-top:8px; font-size:0.95em;">${stepMeta.description}</div>
            `;
        }

        if(codeSnippet) {
            codeSnippet.textContent = stepMeta.code;
            if(window.hljs) hljs.highlightElement(codeSnippet);
        }

        let extra = "";
        if (stepMeta.phase === "warmup") {
            extra = `<div style="color:#e74c3c">Phase: Linear Warmup (Rising)</div>`;
        } else if (stepMeta.phase === "cosine0" || stepMeta.phase === "cycle") {
            extra = `<div style="color:#3498db">Phase: Cosine Annealing (Decaying)</div>
                     <div>Cycle: ${p.cycle} | T_cur: ${p.T_cur} / ${T_max.toFixed(0)}</div>`;
        } else if (stepMeta.phase === "single_cycle") {
            extra = `<div>说明：当 period_mul=0 时，超过 warmup+period 后 lr 恒为 min_lr。</div>`;
        } else if (stepMeta.phase === "ckpt") {
            extra = `<div>ckpt 字段：{cur_lr, lr_steps, cosine_annealing_base_lr, t_cur, cycle}</div>`;
        } else if (stepMeta.phase === "none") {
            extra = `<div>NoneLRScheduler：cur_steps=-1，step() 空实现，始终可以 clip_grad。</div>`;
        }

        if(optState) {
            optState.innerHTML = `
                <div class="state-box"><b>Train Step</b>: ${p.step}</div>
                <div class="state-box"><b>LR</b>: ${p.lr.toExponential(4)}</div>
                ${extra}
            `;
        }

        updateButtons();
    }

    function updateButtons() {
        if(prevBtn) prevBtn.disabled = currentStep === 0;
        if(nextBtn) nextBtn.disabled = currentStep === steps.length - 1;
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

    // Bind events
    if (warmupInput) warmupInput.addEventListener('input', updateParams);
    if (periodMulInput) periodMulInput.addEventListener('input', updateParams);
    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);
    if (resetBtn) resetBtn.addEventListener('click', () => { currentStep = 0; render(); });

    window.addEventListener('resize', drawCanvas);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    // Init
    try {
        LR_TIMELINE = simulateSchedulerTimeline(TOTAL_TRAIN_STEPS, CONFIG);
        initCanvas();
        render();
    } catch(e) {
        console.error("Scheduler Init Failed", e);
    }
});
