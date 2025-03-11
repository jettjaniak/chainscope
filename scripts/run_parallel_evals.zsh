#!/bin/zsh

MAX_SCREENS=30

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting parallel evaluations with max $MAX_SCREENS screens"

for file in chainscope/data/cot_responses/instr-wm/T0.7_P0.9_M2000/**/*.yaml; do
    # :h gets parent directory, :t gets its name
    parent=${file:h:t}
    # :t gets filename, :r removes extension
    basename=${file:t:r}

    # if [[ $basename != "openai__gpt-4o" ]]; then
    #     continue
    # fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing file: $file"

    # Wait if we have too many screens
    while true; do
        screen_count=$(screen -ls | grep -c "\.")
        if (( screen_count < MAX_SCREENS )); then
            break
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for available screen slot (current: $screen_count/$MAX_SCREENS)"
        sleep 10  # Wait 10 seconds before checking again
    done

    screen_name="${parent}_${basename}"
    log_file="logs/${parent}_${basename}.log"
    # Create logs directory if it doesn't exist
    mkdir -p logs
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting screen '$screen_name'"
    screen -dmS "$screen_name" bash -c "./scripts/eval_cots.py \"$file\" -m deepseek/deepseek-chat --api ds"

    sleep 5  # Wait 5 seconds before starting next screen
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All evaluation jobs have been queued" 