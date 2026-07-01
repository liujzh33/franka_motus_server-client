#!/bin/bash

printf '%-5s | %-10s | %-8s | %-28s | %-70s\n' "GPU" "显存占用" "利用率" "使用账号" "任务"
printf '%s\n' "--------------------------------------------------------------------------------------------------------------------------------"

for gpu in 0 1 2 3 4 5 6 7; do
  mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u)

  if [ -z "$mem" ]; then
    printf '%-5s | %-10s | %-8s | %-28s | %-70s\n' "$gpu" "N/A" "N/A" "无法读取" "nvidia-smi 查询失败"
    continue
  fi

  mem_show="${mem}MB"
  util_show="${util}%"

  if [ -z "$pids" ]; then
    if [ "$mem" -gt 1000 ]; then
      printf '%-5s | %-10s | %-8s | %-28s | %-70s\n' "$gpu" "$mem_show" "$util_show" "其他容器(PID不可见)" "显存被占用，但当前容器看不到进程PID"
    else
      printf '%-5s | %-10s | %-8s | %-28s | %-70s\n' "$gpu" "$mem_show" "$util_show" "空闲" "-"
    fi
    continue
  fi

  accounts=""
  tasks=""

  for pid in $pids; do
    cmd=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | sed 's/[[:space:]]\+/ /g' | head -c 220)
    envs=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n')
    cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null)
    ps_user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')

    all_text="$cmd $cwd $envs"

    account=$(echo "$all_text" | grep -oE 'w{0,2}x[0-9]{7,}' | head -1)

    if [ -z "$account" ]; then
      account="$ps_user"
    fi

    if [ -z "$account" ]; then
      account="未知账号"
    fi

    proc_mem=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v p="$pid" '$1 ~ p {gsub(/ /,"",$2); print $2"MB"; exit}')

    task_name=$(echo "$cmd" | grep -oE '[^ /]+\.py|torchrun|python|eval_policy|train[^ ]*' | head -1)
    if [ -z "$task_name" ]; then
      task_name=$(echo "$cmd" | cut -c1-60)
    fi
    if [ -z "$task_name" ]; then
      task_name="PID可见但命令不可读"
    fi

    accounts="${accounts}${account}; "
    tasks="${tasks}PID=${pid} ${proc_mem} ${task_name}; "
  done

  accounts=$(echo "$accounts" | sed 's/; $//' | sed 's/;/, /g')
  tasks=$(echo "$tasks" | sed 's/; $//')

  printf '%-5s | %-10s | %-8s | %-28s | %-70s\n' "$gpu" "$mem_show" "$util_show" "$accounts" "$tasks"
done
