for balance in 0.4 0.7 1.0; do
    for gamma in 0.1 0.4 0.7 1.0; do
        /workspace/examples/python_rs/llm/vllm/kv-router-run.sh 8 prefix $gamma $balance deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        # Wait for GPU memory usage to reach high utilization (indicating model is loaded)
        while true; do
            mem_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
            mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
            mem_percent=$(( mem_usage * 100 / mem_total ))
            echo "GPU memory usage: $mem_percent%"
            if [ "$mem_percent" -gt 80 ]; then
                break
            fi
            sleep 1
        done
        # Additional wait to ensure stability
        sleep 40
        python3 -m common.client_bench --gamma $gamma --balance_threshold $balance
        tmux ls | grep 'v-' | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
        sleep 10   # Sleep for 30 seconds after cleanup
    done
done