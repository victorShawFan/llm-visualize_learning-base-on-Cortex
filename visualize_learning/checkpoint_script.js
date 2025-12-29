document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('ckpt-container');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const resetBtn = document.getElementById('reset-btn');
    const currentStepSpan = document.getElementById('current-step');
    const totalStepSpan = document.getElementById('total-step');
    const stepTitle = document.getElementById('step-title');
    const stepDesc = document.getElementById('step-desc');
    const codeRef = document.querySelector('.code-reference code');

    if (!container || !prevBtn || !nextBtn || !resetBtn || !currentStepSpan || !totalStepSpan || !stepTitle || !stepDesc || !codeRef) {
        console.error("Required elements not found in Checkpoint script");
        return;
    }

    let currentStep = 0;

    const steps = [
        {
            title: "Step 1: 分布式分片保存 (ZeRO Sharding)",
            desc: "在 ZeRO-3 模式下，权重被切分在不同 GPU 上。保存时，每个 Rank 独立写入其持有的参数分片（如 <code>mp_rank_0X_model_states.pt</code>），避免了单节点 I/O 瓶颈。<span class='step-badge'>ds_checkpoint.py:27</span>",
            badge: "model.save_checkpoint(save_dir=ckpt_dir)",
            state: "save"
        },
        {
            title: "Step 2: 磁盘目录结构 (Storage Layout)",
            desc: "文件系统呈现层级化结构：<code>latest</code> 文件指向当前最先进步数；每个 <code>global_step_*</code> 文件夹内存储了所有 Rank 的状态及优化器元数据。",
            badge: "ls checkpoint/global_step_*/",
            state: "fs_view"
        },
        {
            title: "Step 3: 自动化滚动清理 (Pruning Logic)",
            desc: "为了节省磁盘空间，Rank 0 会监控文件夹数量。当超过 <code>CKPT_MAX_TO_KEEP</code> 时，系统自动识别并删除时间戳最旧的快照，保持存储处于稳态。<span class='step-badge'>ds_checkpoint.py:38</span>",
            badge: "shutil.rmtree(oldest_ckpt)",
            state: "prune"
        },
        {
            title: "Step 4: 权重离线合并 (Consolidation)",
            desc: "<b>关键工程：</b>为了方便推理或上传 ModelScope，需要将零散的 ZeRO 分片合并。工具遍历所有 pt 文件，按原模型参数映射表还原出完整的单文件 <code>state_dict</code>。<span class='step-badge'>ds_checkpoint.py:63</span>",
            badge: "get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)",
            state: "merge"
        },
        {
            title: "Step 5: 容错恢复与断点续训 (Resume Flow)",
            desc: "系统在 <code>steps.pt</code> 中记录 global_steps、Scheduler 周期等元数据。启动时先加载元数据对齐进度，再由 DeepSpeed 恢复分片权重，实现真正的无缝续训。<span class='step-badge'>checkpoint.py:140</span>",
            badge: "ckpt.update(lr_scheduler.get_ckpt_dict())",
            state: "train_resume"
        }
    ];

    function updateUI() {
        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        currentStepSpan.innerText = currentStep + 1;
        totalStepSpan.innerText = steps.length;
        stepTitle.innerText = step.title;
        stepDesc.innerHTML = step.desc;
        codeRef.innerText = step.badge;
        
        render(step.state);
        
        prevBtn.disabled = currentStep === 0;
        nextBtn.innerText = currentStep === steps.length - 1 ? "Finish" : "Next Step";
        nextBtn.disabled = currentStep === steps.length - 1;
    }

    function render(state) {
        container.innerHTML = '';
        if (state === "save") {
            const grid = document.createElement('div');
            grid.style.display = 'grid'; grid.style.gridTemplateColumns = 'repeat(4, 1fr)'; grid.style.gap = '15px';
            for(let i=0; i<4; i++) {
                grid.innerHTML += `
                    <div class="gpu-box" style="border:2px solid #4299e1; padding:10px; border-radius:10px; background:#ebf8ff;">
                        <div style="font-size:10px; font-weight:bold;">Rank ${i}</div>
                        <div style="background:#fff; border:1px solid #ddd; font-size:8px; margin-top:5px; padding:4px;">shard_${i}.pt</div>
                    </div>
                `;
            }
            container.appendChild(grid);
        }
        else if (state === "fs_view") {
            container.innerHTML = `
                <div style="background:#2d3748; color:#fff; padding:15px; border-radius:10px; font-family:monospace; font-size:11px; text-align:left;">
                    checkpoint/<br>
                    ├── latest ("global_step_200")<br>
                    ├── global_step_100/ (Rank Shards 0..N)<br>
                    └── global_step_200/ (Rank Shards 0..N)
                </div>
            `;
        }
        else if (state === "prune") {
            container.innerHTML = `
                <div style="display:flex; gap:10px; justify-content:center; align-items:center;">
                    <div style="opacity:0.4; filter:grayscale(1);">📁 step_100<br><span style="color:#e53e3e; font-size:9px;">[Deleted]</span></div>
                    <div class="arrow">→</div>
                    <div style="border:2px solid #48bb78; padding:10px; border-radius:8px;">📁 step_200<br><span style="color:#48bb78; font-size:9px;">[Retained]</span></div>
                    <div style="border:2px solid #3182ce; padding:10px; border-radius:8px;">📁 step_300<br><span style="color:#3182ce; font-size:9px;">[New]</span></div>
                </div>
            `;
        }
        else if (state === "merge") {
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; gap:15px;">
                    <div style="display:flex; gap:5px;">
                        ${[0,1,2,3].map(i => `<div style="width:25px; height:25px; background:#3182ce; opacity:0.6;"></div>`).join('')}
                    </div>
                    <div style="font-size:18px;">⬇️</div>
                    <div style="width:120px; height:30px; background:linear-gradient(to right, #3182ce, #63b3ed); border:2px solid #2c5282; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:10px; color:#fff; font-weight:bold;">Consolidated.bin</div>
                </div>
            `;
        }
        else if (state === "train_resume") {
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:10px; align-items:center;">
                    <div class="tensor-row" style="background:#edf2f7;">1. Load steps.pt → global_steps=200</div>
                    <div class="tensor-row" style="background:#edf2f7;">2. Load Layer Shards 0..N</div>
                    <div class="tensor-row" style="background:#e6fffa; border-color:#38a169;">3. Start from step 201</div>
                </div>
            `;
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

    nextBtn.addEventListener('click', goNext);
    prevBtn.addEventListener('click', goPrev);
    
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
