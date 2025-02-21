# Run specific (balance, gamma) pairs
declare -a balances=(0.4 0.2 0.3 0.4 0.5)
declare -a gammas=(0.1 0.5 0.5 0.5 0.5)

for i in {0..4}; do
    balance=${balances[$i]}
    gamma=${gammas[$i]}
    /workspace/examples/python_rs/llm/vllm/kv-router-run.sh 8 prefix $gamma $balance deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    sleep 180  # Sleep for 3 minutes after server startup
    python3 -m common.client_bench --gamma $gamma --balance_threshold $balance
    tmux ls | grep 'v-' | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
    sleep 30   # Sleep for 30 seconds after cleanup
done