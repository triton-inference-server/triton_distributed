for gamma in $(seq 0.1 0.1 0.5); do
    for balance in $(seq 0.1 0.1 0.5); do
        /workspace/examples/python_rs/llm/vllm/kv-router-run.sh 8 prefix $gamma $balance deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        sleep 180  # Sleep for 3 minutes after server startup
        python3 -m common.client_bench --gamma $gamma --balance_threshold $balance
        tmux ls | grep 'v-' | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
        sleep 30   # Sleep for 30 seconds after cleanup
    done
done