import time
import subprocess

# 要运行的脚本列表
scripts = [
    "run_sjtu.py",
    "run_china.py", 
    ]
total_start_time = time.time()

for script in scripts:
    try:
        subprocess.run(["python", script], )  
    except subprocess.CalledProcessError as e:
        print(f"脚本 {script} 运行失败，错误信息如下：")
        print(e.stderr)

# 记录整个实验的结束时间并计算总运行时间
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"All rounds completed in {total_time / 3600:.2f} hours.")