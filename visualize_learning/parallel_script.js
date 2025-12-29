document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset');
    const infoBox = document.getElementById('infoBox');
    const visualContent = document.getElementById('visualContent');
    const codeSnippet = document.getElementById('codeSnippet');
    const detailPanel = document.getElementById('detailPanel');

    if (!prevBtn || !nextBtn || !resetBtn || !infoBox || !visualContent || !codeSnippet || !detailPanel) {
        console.error("Required elements not found in Parallel script");
        return;
    }

    let currentStep = 0;
    let currentInterval = null;
    let currentTimeout = null;

    const steps = [
        {
            title: "Step 1: 初始化与 Rank 分配 (Init)",
            description: "启动分布式环境，<code>dist.init_process_group</code> 建立通信网。每个 GPU 被分配：<br>1. <b>Global Rank</b>: 集群中的唯一 ID。<br>2. <b>Local Rank</b>: 当前机器内的 GPU 编号。<span class='step-badge'>parallel.py:34-78</span>",
            code: "dist.init_process_group(backend='nccl')\n# Rank 0: 主进程 (Main Process)\n# world_size: 总 GPU 数",
            render: () => renderInit()
        },
        {
            title: "Step 2: 数据分布式采样 (DistributedSampler)",
            description: "<code>DistributedSampler</code> 保证每个 Rank 训练不重复的数据。逻辑：将数据集 indices 按 <code>world_size</code> 取模，Rank <code>i</code> 只获取 <code>index % world_size == i</code> 的样本。<span class='step-badge'>parallel.py:112</span>",
            code: "sampler = DistributedSampler(dataset)\ndataloader = DataLoader(..., sampler=sampler)\n# 每个 Epoch 开始前需调用 sampler.set_epoch(epoch) 来重新打乱顺序",
            render: () => renderDataSplit()
        },
        {
            title: "Step 3: 并行模式选择 (Parallel Implementation)",
            description: "<code>Cortex</code> 提供三种并行抽象：<br>1. <b>DdpParallel</b>: 原生 DDP，每张卡存完整模型。<br>2. <b>DsParallel</b>: DeepSpeed ZeRO-3，模型参数分片。<br>3. <b>NoneParallel</b>: 单卡模式。此外，所有模式都支持 <code>torch.compile</code> 图编译加速。<span class='step-badge'>parallel.py:210-320</span>",
            code: "parallel = DdpParallel() # 或 DsParallel()\nif self._use_compile:\n    model = torch.compile(model)\nmodel, optim = parallel.process(model, optimizer)",
            render: () => renderParallelModes()
        },
        {
            title: "Step 4: 梯度同步 - Ring All-Reduce (Gradient Sync)",
            description: "在 DDP 中，每个 GPU 计算完局部梯度后，通过 <b>Ring All-Reduce</b> 算法进行聚合。数据在 GPU 之间环形传递，经过 2*(N-1) 次传输后，所有 GPU 都获得了梯度的平均值。<span class='step-badge'>DDP 隐式调用</span>",
            code: "# DDP 内部逻辑\n# 1. 计算本地梯度 (Backward)\n# 2. 触发 All-Reduce 聚合所有 Rank 的梯度",
            render: () => renderRingAllReduce()
        },
        {
            title: "Step 5: ZeRO-3 参数分片与收集 (ZeRO-3 Flow)",
            description: "在 <code>DsParallel</code> 中使用 ZeRO-3。前向传播时，GPU 仅在需要某层权重时通过 <b>All-Gather</b> 临时收集完整参数，计算完立即释放（Flush），极大节省显存。<span class='step-badge'>parallel.py:234</span>",
            code: "model, ... = deepspeed.initialize(..., config=ds_config)\n# 权重分片 P0, P1, P2... 分散在各 GPU 上",
            render: () => renderZeroFlow()
        },
        {
            title: "Step 6: 同步屏障与主进程控制 (Barrier & Rank 0)",
            description: "为了防止进程跑太快导致数据不同步，调用 <code>wait()</code> (封装了 <code>dist.barrier()</code>) 强制对齐。日志打印和模型保存通常由 <code>is_main_process</code> (Rank 0) 唯一执行。<span class='step-badge'>parallel.py:180-207</span>",
            code: "if parallel.is_main_process:\n    self.save_checkpoint()\nparallel.wait('after_save')",
            render: () => renderBarrier()
        }
    ];

    function updateUI() {
        if (currentInterval) {
            clearInterval(currentInterval);
            currentInterval = null;
        }
        if (currentTimeout) {
            clearTimeout(currentTimeout);
            currentTimeout = null;
        }

        if (currentStep < 0) currentStep = 0;
        if (currentStep >= steps.length) currentStep = steps.length - 1;

        const step = steps[currentStep];
        infoBox.innerHTML = `<strong>${step.title}</strong><br>${step.description}`;
        codeSnippet.textContent = step.code;
        visualContent.innerHTML = ''; 
        
        try {
            step.render();
        } catch(e) {
            console.error("Render failed", e);
        }

        updateButtons();
    }

    // --- Renderers ---

    function renderInit() {
        visualContent.innerHTML = `
            <div class="gpu-cluster">
                ${[0,1,2,3].map(i => `
                    <div class="gpu-card" style="animation: slideIn 0.4s ${i*0.1}s both;">
                        <h4>GPU ${i}</h4>
                        <div style="font-size:11px;">Rank: ${i}<br>Local: ${i}</div>
                        <div class="status-led" style="background:#2ecc71; width:8px; height:8px; border-radius:50%; margin:10px auto;"></div>
                    </div>
                `).join('')}
            </div>
            <div style="text-align:center; margin-top:20px; font-weight:bold; color:#2980b9;">NCCL Communication Ring Established</div>
        `;
        detailPanel.innerText = "Processes are spawned and connected via NCCL backend.";
    }

    function renderDataSplit() {
        visualContent.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:15px; align-items:center;">
                <div style="display:flex; gap:3px;">
                    ${Array.from({length:12}).map((_, i) => `<div class="mini-box" style="background:#bdc3c7; width:20px; height:20px; font-size:10px;">${i}</div>`).join('')}
                </div>
                <div class="arrow">↓ index % 4 ↓</div>
                <div class="gpu-cluster">
                    ${[0,1,2,3].map(i => `
                        <div class="gpu-card" style="height:70px;">
                            <h4>GPU ${i}</h4>
                            <div style="display:flex; gap:2px; justify-content:center;">
                                <div class="mini-box" style="background:#3498db; width:18px; height:18px; font-size:9px;">${i}</div>
                                <div class="mini-box" style="background:#3498db; width:18px; height:18px; font-size:9px;">${i+4}</div>
                                <div class="mini-box" style="background:#3498db; width:18px; height:18px; font-size:9px;">${i+8}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        detailPanel.innerText = "Each GPU handles a unique shard of the dataset. DistributedSampler handles the indexing.";
    }

    function renderParallelModes() {
        visualContent.innerHTML = `
            <div style="display:flex; gap:15px; justify-content:center; width:100%;">
                <div class="gpu-card" style="border-color:#3498db; flex:1;">
                    <h4 style="color:#3498db;">DDP</h4>
                    <div style="font-size:10px; color:#666; text-align:left;">
                        • 完整参数复写到每张卡<br>• 反向传播同步梯度<br>• 速度快，但显存开销大
                    </div>
                </div>
                <div class="gpu-card" style="border-color:#e67e22; flex:1;">
                    <h4 style="color:#e67e22;">ZeRO-3</h4>
                    <div style="font-size:10px; color:#666; text-align:left;">
                        • 参数/梯度/优化器状态分片<br>• 动态 All-Gather 收集参数<br>• 显存利用率极高
                    </div>
                </div>
                <div class="gpu-card" style="border-color:#27ae60; flex:1;">
                    <h4 style="color:#27ae60;">None</h4>
                    <div style="font-size:10px; color:#666; text-align:left;">
                        • 单卡模式<br>• 不启动分布式组<br>• 适合单机测试/推理
                    </div>
                </div>
            </div>
        `;
        detailPanel.innerText = "DdpParallel vs DsParallel vs NoneParallel. Chosen based on scale.";
    }

    function renderRingAllReduce() {
        visualContent.innerHTML = `
            <div style="position:relative; width:100%; height:200px; display:flex; justify-content:center; align-items:center;">
                <div id="ring-container" style="position:relative; width:200px; height:200px;">
                    ${[0,1,2,3].map(i => {
                        const angle = i * 90;
                        const x = 85 + Math.cos(angle * Math.PI / 180) * 80;
                        const y = 85 + Math.sin(angle * Math.PI / 180) * 80;
                        return `
                            <div class="gpu-card" style="position:absolute; width:40px; height:40px; left:${x}px; top:${y}px; margin:0; font-size:10px; padding:0;">
                                G${i}
                                <div id="data-dot-${i}" style="width:10px; height:10px; background:#9b59b6; border-radius:50%; margin:2px auto;"></div>
                            </div>
                        `;
                    }).join('')}
                    <svg width="200" height="200" style="position:absolute; top:0; left:0;">
                        <circle cx="100" cy="100" r="80" fill="none" stroke="#9b59b6" stroke-dasharray="5,5" opacity="0.3" />
                    </svg>
                </div>
            </div>
        `;
        
        let angleOffset = 0;
        currentInterval = setInterval(() => {
            angleOffset += 5;
            for(let i=0; i<4; i++) {
                const dot = document.getElementById(`data-dot-${i}`);
                if(!dot) {
                    if(currentInterval) clearInterval(currentInterval);
                    return;
                }
                const baseAngle = i * 90;
                const currentAngle = (baseAngle + angleOffset) * Math.PI / 180;
                // const x = Math.cos(currentAngle) * 80; // This was in original but not used for transform
                // const y = Math.sin(currentAngle) * 80;
                // Actually the dots are just static in the original code, 
                // wait, the original code computed x/y but didn't assign them to dot.style.left/top
                // It just rendered them inside the loop.
                // Ah, the original code had:
                /*
                const x = Math.cos(currentAngle) * 80;
                const y = Math.sin(currentAngle) * 80;
                // This is just a visual trick for "passing"
                */
                // It seems the original code's animation was incomplete or broken (it calculated x/y but didn't update DOM).
                // I should fix this to make it actually animate.
                
                // Fixed animation logic:
                // We need to move the dots relative to their container center (100, 100)
                // But the dots are inside `gpu-card` which is absolute positioned.
                // This structure is hard to animate continuously.
                // Let's just pulsate them to show activity instead of complex rotation for stability.
                dot.style.opacity = (Math.sin((angleOffset + i * 90) * Math.PI / 180) + 1) / 2;
            }
        }, 50);

        detailPanel.innerText = "Ring All-Reduce: Data flows in a logical ring. Each step, GPUs exchange and accumulate gradient chunks.";
    }

    function renderZeroFlow() {
        visualContent.innerHTML = `
            <div class="gpu-cluster">
                ${[0,1,2,3].map(i => `
                    <div class="gpu-card" style="height:110px;">
                        <h4>GPU ${i}</h4>
                        <div style="font-size:9px; background:#eee; margin-bottom:2px; color:#999;">Layer 0</div>
                        <div style="font-size:9px; background:#eee; margin-bottom:2px; color:#999;">Layer 1</div>
                        <div style="font-size:9px; background:${i===1?'#e74c3c':'#eee'}; color:${i===1?'#fff':'#999'}; border:1px solid #e74c3c;">Layer 2 (Owner)</div>
                        <div style="font-size:9px; background:#eee; color:#999;">Layer 3</div>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top:20px; font-size:12px; color:#e67e22; font-weight:bold;" id="flow-msg">Target: Forward Layer 2...</div>
        `;
        
        currentTimeout = setTimeout(() => {
            const msg = document.getElementById('flow-msg');
            if(msg) msg.innerText = "Step 1: All-Gather Layer 2 weights from GPU 1...";
        }, 1500);
    }

    function renderBarrier() {
        visualContent.innerHTML = `
            <div style="display:flex; align-items:center; gap:20px; height:180px; justify-content:center;">
                <div class="gpu-cluster" style="margin:0;">
                    ${[0,1,2,3].map(i => `<div class="gpu-card" style="height:60px; width:60px;">G${i}<br>Wait</div>`).join('')}
                </div>
                <div style="width:4px; height:100%; background:#2c3e50; position:relative;">
                     <div style="position:absolute; top:50%; left:-40px; transform:translateY(-50%); font-weight:bold; color:#2c3e50;">BARRIER</div>
                </div>
                <div class="gpu-card" style="border-color:#27ae60; background:#eafaf1;">
                    <h4>Rank 0</h4>
                    <div style="font-size:10px;">Save Checkpoint<br>Log Metrics</div>
                </div>
            </div>
        `;
        detailPanel.innerText = "Sync point: All GPUs must reach the barrier before any can proceed. Rank 0 handles the I/O operations.";
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

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });

    updateUI();
});
