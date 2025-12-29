document.addEventListener('DOMContentLoaded', () => {
    const prevBtn = document.getElementById('prevStep');
    const nextBtn = document.getElementById('nextStep');
    const resetBtn = document.getElementById('reset-btn'); // Matches HTML ID in read content
    const container = document.getElementById('unwrap-container');
    const infoBox = document.getElementById('infoBox'); // Assuming standard layout, but HTML seems to use step-desc
    
    // In unwrap_script.js, some IDs were different in the original code:
    // next-btn, prev-btn, reset-btn
    // But standard is nextStep, prevStep.
    // Let's check the read content again carefully.
    // content used: document.getElementById('prev-btn'), document.getElementById('next-btn')
    // I should probably standardise or respect existing IDs.
    // The previous file content used 'prev-btn', 'next-btn'.
    // I will use the IDs found in the file: 'prev-btn', 'next-btn', 'reset-btn'.

    const prevButton = document.getElementById('prev-btn');
    const nextButton = document.getElementById('next-btn');
    const resetButton = document.getElementById('reset-btn');

    // Guard
    if (!container) {
        console.error("Required elements not found in unwrap_script");
        return;
    }

    const steps = [
        {
            title: "ZeRO-3 分片状态 (Sharded State)",
            desc: "训练中，模型参数 P 被切分为 P0, P1, P2, P3 存储在 4 个 GPU 中。此时无法直接进行完整的矩阵乘法（全量推理）。",
            badge: "partition_utils.py:101",
            state: "sharded"
        },
        {
            title: "临时参数聚合 (Gathering)",
            desc: "进入 <code>unwrap_model_for_generation</code> 后，在 ZeRO-3 场景中通过 <code>deepspeed.zero.GatheredParameters(model.parameters())</code> 临时聚合参数，相当于在内部完成一次 All-Gather。<span class='step-badge'>partition_utils.py:30-40</span>",
            badge: "partition_utils.py:35",
            state: "gathering"
        },
        {
            title: "Hook 挂起 (Remove Hooks)",
            desc: "DeepSpeed 的自动 offload 机制（Hooks）会干扰自回归生成。系统会暂时移除这些 Hook，确保生成过程不受内存置换影响。<span class='step-badge'>partition_utils.py:242-270</span>",
            badge: "partition_utils.py:242-270",
            state: "no_hooks"
        },
        {
            title: "全量推理 (Full Generation)",
            desc: "此时模型处于 unwrap 状态，<code>with unwrap_model_for_generation(model) as eval_model:</code> 中的 <code>eval_model</code> 是一个完整的 nn.Module，可直接执行 forward/generate，这对于 KV Cache 的连续更新至关重要。<span class='step-badge'>trainer.py:665-699</span>",
            badge: "partition_utils.py:13-44",
            state: "gen"
        },
        {
            title: "状态恢复 (Cleanup)",
            desc: "生成结束，退出上下文管理器。Gathered 内存被释放，参数重新回到 Sharded 状态，并重新挂载 Optimizer Hooks。<span class='step-badge'>partition_utils.py:218-236</span>",
            badge: "partition_utils.py:218-236",
            state: "cleanup"
        },
        {
            title: "DdpParallel / NoneParallel 的轻量解包路径",
            desc: "当使用 <code>DdpParallel</code> 或 <code>NoneParallel</code> 时，<code>unwrap_model_for_generation</code> 不需要做 ZeRO 聚合：DDP 场景下直接返回 <code>model.module</code>，非分布式场景下则原样返回 <code>model</code>，保证 API 一致但开销最小。<span class='step-badge'>partition_utils.py:41-44,101-117</span>",
            badge: "partition_utils.py:41-44",
            state: "light_unwrap"
        },
        {
            title: "多种包装类型的解包 (unwrap_model)",
            desc: "无论是 DeepSpeedEngine 还是 DDP，最终都可以通过 <code>unwrap_model</code> 拿到内部的基础模型 (nn.Module)。这一步是参数对齐、权重导出和评估前的统一入口。<span class='step-badge'>partition_utils.py:101-117</span>",
            badge: "partition_utils.py:101-117",
            state: "unwrap"
        },
        {
            title: "sync_model_params - 完全对齐参数",
            desc: "在 PPO/GRPO 等场景中，需要把训练中的策略模型参数同步到一个评估/参考模型上。<code>sync_model_params(_from, _to, mixup_alpha=1.0)</code> 会在所有 Rank 上用 state_dict 覆盖 _to 的参数。<span class='step-badge'>partition_utils.py:56-75</span>",
            badge: "partition_utils.py:56-75",
            state: "sync"
        },
        {
            title: "sync_model_params - 参数混合 (mixup_alpha)",
            desc: "当 <code>mixup_alpha &lt; 1.0</code> 时，不是直接覆盖，而是做线性插值：<code>target = (1-α)*target + α*from</code>。这可以在不同策略之间做平滑融合，避免训练震荡。<span class='step-badge'>partition_utils.py:76-83,177-185</span>",
            badge: "partition_utils.py:76-83,177-185",
            state: "mix"
        },
        {
            title: "_get_ds_model_params + 广播 (Rank 0 汇总 + 同步)",
            desc: "ZeRO-3 场景下，只有 Rank 0 在 <code>_get_ds_full_state_dict_on_rank0</code> 中拿到完整 state_dict，其余 Rank 得到 None。随后通过 <code>dist.broadcast_object_list</code> 把这个 state_dict 广播出去，让所有进程共享同一份参数快照。<span class='step-badge'>partition_utils.py:120-174</span>",
            badge: "partition_utils.py:120-174",
            state: "broadcast"
        },
        {
            title: "_add_hooks 与 _remove_hooks (恢复 Offload 行为)",
            desc: "生成阶段前使用 <code>_remove_hooks</code> 清空 DeepSpeed optimizer 的 forward/backward hooks 与 <code>ds_active_sub_modules</code>，避免触发不必要的参数 offload。生成结束后再通过 <code>_add_hooks</code> 重新注册这些 hooks，恢复正常的 ZeRO-3 训练行为。<span class='step-badge'>partition_utils.py:218-236,242-270</span>",
            badge: "partition_utils.py:218-236,242-270",
            state: "rehook"
        }
    ];

    let currentStep = 0;

    function render() {
        const step = steps[currentStep];
        const currentStepSpan = document.getElementById('current-step');
        if (currentStepSpan) currentStepSpan.innerText = currentStep + 1;
        
        const totalSpan = document.getElementById('total-step');
        if (totalSpan) totalSpan.innerText = steps.length;
        
        const stepTitle = document.getElementById('step-title');
        if (stepTitle) stepTitle.innerText = step.title;
        
        const stepDesc = document.getElementById('step-desc');
        if (stepDesc) stepDesc.innerHTML = step.desc;
        
        const badgeEl = document.querySelector('.code-reference .step-badge');
        if (badgeEl) badgeEl.innerText = step.badge;

        container.innerHTML = '';
        
        if (step.state === 'sharded') {
            renderSharded();
        } else if (step.state === 'gathering') {
            renderGathering();
        } else if (step.state === 'no_hooks') {
            renderNoHooks();
        } else if (step.state === 'gen') {
            renderGen();
        } else if (step.state === 'cleanup') {
            renderCleanup();
        } else if (step.state === 'light_unwrap') {
            renderLightUnwrap();
        } else if (step.state === 'unwrap') {
            renderUnwrap();
        } else if (step.state === 'sync') {
            renderSync();
        } else if (step.state === 'mix') {
            renderMix();
        } else if (step.state === 'broadcast') {
            renderBroadcast();
        } else if (step.state === 'rehook') {
            renderRehook();
        }

        if (prevButton) prevButton.disabled = currentStep === 0;
        if (nextButton) nextButton.innerText = currentStep === steps.length - 1 ? "完成" : "下一步";
    }

    function renderSharded() {
        container.innerHTML = `
            <div class="sharded-view">
                <div class="gpu-row">
                    <div class="gpu-shard">P0</div><div class="gpu-shard">P1</div><div class="gpu-shard">P2</div><div class="gpu-shard">P3</div>
                </div>
                <div class="status-msg">State: Training (Memory Optimized)</div>
            </div>
        `;
    }

    function renderGathering() {
        container.innerHTML = `
            <div class="gather-view">
                <div class="ncc-ring">NCCL All-Gather</div>
                <div class="full-weight-box pulsing">P0 + P1 + P2 + P3</div>
                <div class="status-msg">Gathering Parameters into VRAM...</div>
            </div>
        `;
    }

    function renderNoHooks() {
        container.innerHTML = `
            <div class="hooks-view">
                <div class="hook disabled">Offload Hook X</div>
                <div class="hook disabled">Grad Hook X</div>
                <div class="hook disabled">Param Hook X</div>
                <div class="status-msg">Optimizer Hooks Suspended</div>
            </div>
        `;
    }

    function renderGen() {
        container.innerHTML = `
            <div class="gen-view">
                <div class="input-seq">"Hello" -> [LLM] -> "World"</div>
                <div class="active-model">Unwrapped Module (Full Params)</div>
                <div class="kv-cache-status">KV Cache Active</div>
            </div>
        `;
    }

    function renderCleanup() {
        container.innerHTML = `
            <div class="cleanup-view">
                <div class="gpu-row">
                    <div class="gpu-shard">P0</div><div class="gpu-shard">P1</div><div class="gpu-shard">P2</div><div class="gpu-shard">P3</div>
                </div>
                <div class="status-msg">State: Re-Sharded & Hooks Restored</div>
            </div>
        `;
    }

    function renderLightUnwrap() {
        container.innerHTML = `
            <div class="unwrap-view">
                <div class="gpu-row">
                    <div class="gpu-shard">DDP(model)</div>
                    <div class="arrow">unwrap_model_for_generation</div>
                    <div class="gpu-shard">model.module</div>
                </div>
                <div class="gpu-row" style="margin-top:10px;">
                    <div class="gpu-shard">Plain nn.Module</div>
                    <div class="arrow">unwrap_model_for_generation</div>
                    <div class="gpu-shard">Same instance (no-op)</div>
                </div>
                <div class="status-msg">无 ZeRO 分片时的轻量路径：不做参数聚合，只解包一层封装或直接返回原模型。</div>
            </div>
        `;
    }

    function renderUnwrap() {
        container.innerHTML = `
            <div class="unwrap-view">
                <div class="gpu-row">
                    <div class="gpu-shard">DeepSpeedEngine</div>
                    <div class="arrow">unwrap_model()</div>
                    <div class="gpu-shard">nn.Module</div>
                </div>
                <div class="gpu-row" style="margin-top:10px;">
                    <div class="gpu-shard">DDP(model)</div>
                    <div class="arrow">unwrap_model()</div>
                    <div class="gpu-shard">model.module</div>
                </div>
                <div class="status-msg">统一入口：无论封装方式如何，最终都拿到基础模型来做 state_dict / forward / export。</div>
            </div>
        `;
    }

    function renderSync() {
        container.innerHTML = `
            <div class="sync-view">
                <div class="gpu-row">
                    <div class="gpu-shard">_from (训练中)</div>
                    <div class="arrow">state_dict → broadcast</div>
                    <div class="gpu-shard">_to (评估/参考模型)</div>
                </div>
                <div class="status-msg">mixup_alpha = 1.0：_to 参数在所有 Rank 上与 _from 完全对齐。</div>
            </div>
        `;
    }

    function renderMix() {
        container.innerHTML = `
            <div class="sync-view">
                <div class="gpu-row">
                    <div class="gpu-shard">_to = 0.8·old + 0.2·from</div>
                </div>
                <div class="status-msg">参数混合：通过 mixup_alpha 在旧参数与新参数之间做平滑插值，减缓策略切换带来的震荡。</div>
            </div>
        `;
    }

    function renderBroadcast() {
        container.innerHTML = `
            <div class="broadcast-view">
                <div class="gpu-row">
                    <div class="gpu-shard" style="background:#f6e05e;">Rank 0\n(full state_dict)</div>
                    <div class="gpu-shard" style="background:#e2e8f0;">Rank 1\nNone</div>
                    <div class="gpu-shard" style="background:#e2e8f0;">Rank 2\nNone</div>
                    <div class="gpu-shard" style="background:#e2e8f0;">Rank 3\nNone</div>
                </div>
                <div class="arrow">dist.broadcast_object_list(object_list, src=0)</div>
                <div class="gpu-row">
                    <div class="gpu-shard">Rank 0\nstate_dict</div>
                    <div class="gpu-shard">Rank 1\nstate_dict</div>
                    <div class="gpu-shard">Rank 2\nstate_dict</div>
                    <div class="gpu-shard">Rank 3\nstate_dict</div>
                </div>
                <div class="status-msg">ZeRO-3 下：只在 Rank 0 汇总参数，再广播到所有进程，保证多机视图一致。</div>
            </div>
        `;
    }

    function renderRehook() {
        container.innerHTML = `
            <div class="hooks-view">
                <div class="hook">Offload Hook ✓</div>
                <div class="hook">Grad Hook ✓</div>
                <div class="hook">Param Hook ✓</div>
                <div class="status-msg">生成结束：通过 _add_hooks 重新注册 DeepSpeed optimizer hooks，恢复 ZeRO-3 正常训练行为。</div>
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

    if (nextButton) nextButton.addEventListener('click', goNext);
    if (prevButton) prevButton.addEventListener('click', goPrev);
    if (resetButton) resetButton.addEventListener('click', () => {
        currentStep = 0;
        render();
    });

    render();

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goPrev();
        if (e.key === 'ArrowRight') goNext();
    });
});
