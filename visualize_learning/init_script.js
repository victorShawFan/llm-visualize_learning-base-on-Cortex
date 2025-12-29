document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('init-container');
    const currentStepSpan = document.getElementById('current-step');
    const totalStepSpan = document.getElementById('total-step');
    const stepTitle = document.getElementById('step-title');
    const stepDesc = document.getElementById('step-desc');
    const stepBadge = document.querySelector('.step-badge');
    const nextBtn = document.getElementById('next-btn');
    const prevBtn = document.getElementById('prev-btn');
    const resetBtn = document.getElementById('reset-btn');

    if (!container || !currentStepSpan || !totalStepSpan || !stepTitle || !stepDesc || !stepBadge || !nextBtn || !prevBtn) {
        console.error("Required elements not found in Init script");
        return;
    }

    const steps = [
        {
            title: "Step 0: TrainConfig 统一入口 (Initialization Entry)",
            desc: "<code>TrainConfig</code> 是整个 Cortex 系统的‘神经中枢’。它将模型架构、数据源、优化器策略、并行模式 (DeepSpeed/DDP) 等所有子配置聚合在一起，确保训练的一致性。<span class='step-badge'>train_configs.py:289</span>",
            badge: "train_configs.py:289",
            state: "train_config"
        },
        {
            title: "Step 1: 分布式环境引导 (Bootstrap & Seed)",
            desc: "调用 <code>set_seed(42)</code> 锁定随机性。这在分布式训练中至关重要，确保所有 GPU 上的模型权重初始化完全一致，防止参数分叉导致训练失败。<span class='step-badge'>utils.py:11</span>",
            badge: "utils.py:11",
            state: "seed"
        },
        {
            title: "Step 2: 全局单例检测 (TrainerTools)",
            desc: "检测环境变量 <code>PARALLEL_TYPE</code>。根据环境自动实例化并行控制器（DsParallel / DdpParallel / NoneParallel），并绑定 Tokenizer 作为全局唯一文本入口。<span class='step-badge'>tools.py:14</span>",
            badge: "tools.py:14",
            state: "tools"
        },
        {
            title: "Step 3: 权重矩阵构建 (Model Blueprint)",
            desc: "根据 Config 堆叠 Decoder 层。每层包含注意力投影和 FFN。权重按正态分布 $\\mathcal{N}(0, 0.02)$ 预填充，此时显存开始被大量占用。<span class='step-badge'>llm_model.py:477</span>",
            badge: "llm_model.py:477",
            state: "layers"
        },
        {
            title: "Step 4: 数据规模预估 (Data Estimation)",
            desc: "在正式启动前，系统会遍历数据集文件列表，根据任务类型估算总样本数。这一步用于计算精确的学习率 Warmup 步数和总 Epoch 数。<span class='step-badge'>tools.py:87</span>",
            badge: "tools.py:87",
            state: "data_size"
        },
        {
            title: "Step 5: 系统就绪与同步 (Barrier)",
            desc: "调用 <code>parallel.wait()</code>。所有进程在 Barrier 处集合，确保初始化无误。此时系统输出「Cortex System Ready」，正式进入主循环。<span class='step-badge'>parallel.py:194</span>",
            badge: "parallel.py:194",
            state: "ready"
        }
    ];

    let currentStep = 0;

    function render() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        currentStepSpan.innerText = currentStep + 1;
        totalStepSpan.innerText = steps.length;
        stepTitle.innerText = step.title;
        stepDesc.innerHTML = step.desc;
        stepBadge.innerText = step.badge;

        container.innerHTML = '';
        
        try {
            if (step.state === 'train_config') renderTrainConfig();
            else if (step.state === 'seed') renderSeed();
            else if (step.state === 'tools') renderTools();
            else if (step.state === 'layers') renderLayers();
            else if (step.state === 'data_size') renderDataSize();
            else if (step.state === 'ready') renderReady();
        } catch(e) {
            console.error("Render failed", e);
        }

        prevBtn.disabled = currentStep === 0;
        nextBtn.innerText = currentStep === steps.length - 1 ? "Start Training" : "Next Step";
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    function renderTrainConfig() {
        container.innerHTML = `
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; width:100%;">
                <div class="singleton-box"><strong>ModelConfig</strong><br><small>Hidden: 4096, Layers: 32</small></div>
                <div class="singleton-box"><strong>OptimConfig</strong><br><small>LR: 3e-4, AdamW</small></div>
                <div class="singleton-box"><strong>DsConfig</strong><br><small>ZeRO-3, Offload</small></div>
                <div class="singleton-box"><strong>DatasetConfig</strong><br><small>Batch: 4, Shuffle: T</small></div>
            </div>
        `;
    }

    function renderSeed() {
        container.innerHTML = `
            <div style="display:flex; gap:10px; justify-content:center;">
                ${[0,1,2,3].map(i => `<div class="rank-node">GPU ${i}<br><span style="font-size:10px; color:#27ae60;">Seed: 42</span></div>`).join('')}
            </div>
            <div style="margin-top:20px; font-weight:bold; color:#2980b9;">Determinism Locked across all ranks</div>
        `;
    }

    function renderTools() {
        container.innerHTML = `
            <div class="singleton-box" style="width:80%; margin:0 auto; border:2px solid #8e44ad;">
                <div style="font-weight:bold; color:#8e44ad; border-bottom:1px solid #eee; margin-bottom:10px;">TrainerTools (Singleton)</div>
                <div style="font-size:12px; text-align:left;">
                    • ParallelBackend: DeepSpeed<br>
                    • GlobalTokenizer: Llama-3-8B<br>
                    • AMP: Enabled (BFloat16)
                </div>
            </div>
        `;
    }

    function renderLayers() {
        container.innerHTML = `
            <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; width:100%;">
                ${[0,1,2,31].map(i => `
                    <div class="layer-block">
                        <div style="font-size:10px;">Layer ${i}</div>
                        <div class="weight-matrix" style="height:40px; background:linear-gradient(45deg, #eee, #ddd);"></div>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top:15px; font-size:11px; color:#666;">Allocating VRAM for Weight Matrices...</div>
        `;
    }

    function renderDataSize() {
        container.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; gap:10px;">
                <div style="display:flex; gap:5px;">
                    <div class="file-icon">Part_0.npy</div>
                    <div class="file-icon">Part_1.npy</div>
                </div>
                <div class="arrow">↓</div>
                <div class="tensor-row" style="background:#fef9e7;">Estimated: 1.2M Samples</div>
            </div>
        `;
    }

    function renderReady() {
        container.innerHTML = `
            <div style="text-align:center; padding:20px;">
                <div style="font-size:48px; margin-bottom:10px;">🚀</div>
                <div style="color:#27ae60; font-weight:bold; font-size:20px;">System Core Initialized</div>
                <div style="font-size:12px; color:#666; margin-top:10px;">All parameters synced. Starting main loop...</div>
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

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    render();
});
